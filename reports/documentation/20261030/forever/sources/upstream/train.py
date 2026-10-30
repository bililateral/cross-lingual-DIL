import os
os.environ['TRANSFORMERS_CACHE'] = '/workspace/CL_Memory/cache'
import sys
from typing import List
from peft import PeftModel
import fire
import torch
import transformers
import shutil
from datasets import load_dataset, concatenate_datasets
from transformers import AutoConfig
import gc
import math
import time

AutoConfig.default_cache_dir = '/workspace/CL_Memory/cache'

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.prompter import Prompter
from transformers import set_seed
from utils.dataset_order import get_dataset_order
from utils.load_data import load_current_task_data, load_memory_buffer, load_validation_set

set_seed(42)

class DriftRecorderCallback(transformers.TrainerCallback):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.drift_log_path = os.path.join(output_dir, "drift_log.csv")

        self.tau = 0.0
        self.step = 0
        self.prev_vec = None
        self._header_written = False

        # new
        # for LEAR-style intensity control
        self.mu = None
        self.mu0 = None
        self.ema_alpha = None

        if os.path.exists(self.drift_log_path):
            with open(self.drift_log_path, 'r') as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last_line = lines[-1].strip().split(',')
                    try:
                        self.step = int(last_line[0])
                        self.tau = float(last_line[2])
                        self._header_written = True
                        print(f"[DriftRecorder] Resumed from step={self.step}, tau={self.tau}")
                    except Exception:
                        pass

    def _get_trainable_params_vector(self, model) -> torch.Tensor:
        params = []
        for name, p in model.named_parameters():
            if p is not None and p.requires_grad:
                params.append(p.detach().float().view(-1).cpu())
        if not params:
            return torch.zeros(1)
        return torch.cat(params, dim=0)

    def update_reference_point(self, model):
        with torch.no_grad():
            self.prev_vec = self._get_trainable_params_vector(model)
        print(f"[DriftRecorder] Reference point updated at step {self.step} (Silent Update after Replay).")

    def on_train_begin(self, args, state, control, **kwargs):
        if self.prev_vec is None:
            model = kwargs["model"]
            with torch.no_grad():
                self.prev_vec = self._get_trainable_params_vector(model)

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        with torch.no_grad():
            curr_vec = self._get_trainable_params_vector(model)
            if self.prev_vec is None:
                self.prev_vec = curr_vec
                return

            delta_vec = curr_vec - self.prev_vec
            delta_t = torch.norm(delta_vec, p=2).item()

        self.tau += float(delta_t)
        self.step += 1
        self.prev_vec = curr_vec

        # new
        # update instantaneous update intensity mu_t if calibrated
        if self.mu0 is not None and self.ema_alpha is not None:
            if self.mu is None:
                self.mu = self.mu0
            self.mu = (1.0 - self.ema_alpha) * self.mu + self.ema_alpha * float(delta_t)

        os.makedirs(self.output_dir, exist_ok=True)
        with open(self.drift_log_path, "a", encoding="utf-8") as f:
            if (not self._header_written) and f.tell() == 0:
                f.write("step,delta_t,tau\n")
                self._header_written = True
            f.write(f"{self.step},{delta_t},{self.tau}\n")

class TauTriggerStopCallback(transformers.TrainerCallback):
    def __init__(self, drift_recorder: DriftRecorderCallback, target_tau: float):
        self.drift_recorder = drift_recorder
        self.target_tau = target_tau

    def on_step_end(self, args, state, control, **kwargs):
        if self.target_tau is not None and self.drift_recorder.tau >= self.target_tau:
            control.should_training_stop = True

class MyTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

# new
class ReplayTrainer(MyTrainer):
    """Trainer used for memory replay with parameter-level regularization."""
    def __init__(self, beta_t: float, theta_star: dict, importance: dict, *args, **kwargs):
        self.beta_t = beta_t
        self.theta_star = theta_star
        self.importance = importance

        # new 12.15
        # new: cache for logging
        self.last_base_loss = None
        self.last_loss_reg = None

        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        base_out = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        if isinstance(base_out, tuple):
            base_loss, outputs = base_out
        else:
            base_loss, outputs = base_out, None

        loss_reg = None
        for name, param in model.named_parameters():
            if name not in self.theta_star:
                continue
            ref_param = self.theta_star[name].to(param.device)
            w = self.importance.get(name, 1.0)
            diff = param - ref_param
            reg = w * diff.pow(2).sum()
            if loss_reg is None:
                loss_reg = reg
            else:
                loss_reg = loss_reg + reg

        # new 12.15
        # cache for logging (detach!)
        self.last_base_loss = base_loss.detach().item()
        self.last_loss_reg = loss_reg.detach().item()
        
        if loss_reg is not None:
            total_loss = base_loss + 0.001 * self.beta_t * loss_reg 
            
        else:
            total_loss = base_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss
    
def train(
    # model/data params
    base_model: str = "",
    #data_dir: str = "../data_longsequence",
    data_dir: str = "../data_superni",
    output_path: str = "../checkpoint_files",
    cache_dir: str = "/workspace/CL_Memory/cache",

    # training hyperparams
    batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 1,
    learning_rate: float = 3e-4,
    max_input_length: int = 1024,
    val_set_size: int = 20,
    # lora hyperparams
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = (
        "q_proj",
        "v_proj",
    ),
    # llm hyperparams
    train_on_inputs: bool = False,
    add_eos_token: bool = True,
    group_by_length: bool = False,
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",
    wandb_log_model: str = "",
    resume_from_checkpoint: str = None,
    prompt_template_name: str = "alpaca",
    dataset_id: int = 1,
    task_id: int = 0,
    memory_data_ratio: int = 2,
    first_task_epochs: int = None,
    # Replay params
    memory_epochs: int = 2,
    steps_per_day: int = 24,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(f"Training CL Model Task-{task_id}")

    if first_task_epochs is not None and task_id == 0:
        effective_epochs = first_task_epochs
    else:
        effective_epochs = num_epochs

    dataset_order = get_dataset_order(dataset_id)
    print(f"Current service: {dataset_order[task_id]}")

    model_name = base_model.split("/")[-1] + "lora"
    output_dir = os.path.join(
        output_path,
        model_name + "_cl_dataset_id_" + str(dataset_id),
        str(task_id) + "-" + dataset_order[task_id],
    )
    print(f"output_dir: {output_dir}")

    if task_id == 0:
        lora_weights = ""
    else:
        last_service_name = dataset_order[task_id - 1]
        last_checkpoint_dir = os.path.join(
            output_path,
            model_name + "_cl_dataset_id_" + str(dataset_id),
            str(task_id - 1) + "-" + last_service_name,
        )
        lora_weights = last_checkpoint_dir
        if not os.path.exists(lora_weights):
            print(f"Error: lora_weights dir {lora_weights} not found!")
            sys.exit(1)

    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.padding_side = "left"

    prompter = Prompter(prompt_template_name)

    def tokenize(prompt, add_eos_token=True):
        result = tokenizer(
            prompt,
            truncation=True,
            max_length=max_input_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < max_input_length
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        result["labels"] = result["input_ids"].copy()
        return result

    def generate_and_tokenize_prompt(data_point):
        full_prompt = prompter.generate_prompt(
            data_point["instruction"], data_point["input"], data_point["output"]
        )
        tokenized_full_prompt = tokenize(full_prompt)
        if not train_on_inputs:
            user_prompt = prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            if add_eos_token:
                user_prompt_len -= 1
            tokenized_full_prompt["labels"] = (
                [-100] * user_prompt_len
                + tokenized_full_prompt["labels"][user_prompt_len:]
            )
        return tokenized_full_prompt

    current_data_raw = load_current_task_data(
        dataset_id, task_id, data_dir, cache_dir, model_name
    )
    current_data = current_data_raw.shuffle().map(generate_and_tokenize_prompt)

    memory_data = None
    if task_id > 0:
        memory_data_raw = load_memory_buffer(
            dataset_id, task_id, data_dir, memory_data_ratio, cache_dir, model_name
        )
        memory_data = memory_data_raw.shuffle().map(generate_and_tokenize_prompt)
        print(f"Memory Data loaded: {len(memory_data)} samples.")

    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=list(lora_target_modules),
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    if task_id == 0:
        model = get_peft_model(model, config)
        print("Fine tune lora from scratch!")
    else:
        model = PeftModel.from_pretrained(model, lora_weights, is_trainable=True)
        print("Continual fine tune lora!")

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if not os.path.exists(checkpoint_name):
            checkpoint_name = os.path.join(resume_from_checkpoint, "adapter_model.bin")
        if os.path.exists(checkpoint_name):
            print(f"Restarting from {checkpoint_name}")
            adapters_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, adapters_weights)

    model.print_trainable_parameters()

    # new
    # snapshot reference parameters for LEAR-style replay
    theta_star = {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    importance = {
        name: 1.0
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    gradient_accumulation_steps = batch_size // micro_batch_size
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    num_samples = len(current_data)
    if num_samples == 0:
        print("No samples in current_data, exit.")
        return

    total_steps_current = math.ceil(num_samples / batch_size * effective_epochs)
    print(f"Total estimated steps for Current Task: {total_steps_current}")

    drift_recorder = DriftRecorderCallback(output_dir)

    # new 12.15
    replay_strength_log_path = os.path.join(output_dir, "replay_strength_log.csv")
    replay_loss_log_path = os.path.join(output_dir, "replay_loss_log.csv")
    # header flags
    replay_strength_header_written = False
    replay_loss_header_written = False


    steps_per_day = max(1, int(steps_per_day))
    trigger_days_human = [1, 2, 4, 7, 15, 30, 60, 90, 120]

    calibrated = False           # 是否用前 steps_per_day 校准过 model_day
    model_day = None
    trigger_thresholds: List[float] = []
    next_trigger_idx = 0         # 当前要看的阈值下标
    phase_idx = 0                # 日志用
    steps_done = drift_recorder.step

    # new
    # LEAR-style intensity control hyperparameters
    ema_alpha = 0.05
    beta_base = 1.0
    beta_scale = 1.0
    beta_min, beta_max = 0.5, 3.0

    print(
        f"[Scheduler] steps_per_day={steps_per_day}, "
        f"effective_epochs={effective_epochs}, total_steps_current={total_steps_current}"
    )

    # ========== 主循环：直到当前任务的 total_steps_current 都跑完 ==========
    while steps_done < total_steps_current:
        remaining_steps = total_steps_current - steps_done

        # ====== 第一次：校准 model_day，用前 steps_per_day 步的 tau ======
        if not calibrated:
            calib_steps = min(steps_per_day, remaining_steps)
            phase_idx += 1
            print(
                f"\n=== Phase {phase_idx}: CALIBRATION CURRENT, "
                f"steps={calib_steps}, steps_done(before)={steps_done}/{total_steps_current} ==="
            )

            calib_args = transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=10,
                max_steps=calib_steps,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                save_strategy="steps",
                save_steps=40,
                save_total_limit=2,
                output_dir=output_dir,
                report_to="wandb" if (len(wandb_project) > 0) else None,
                run_name=f"{wandb_run_name}_phase_{phase_idx}"
                if (len(wandb_run_name) > 0)
                else None,
            )

            trainer_calib = MyTrainer(
                model=model,
                train_dataset=current_data,
                eval_dataset=None,
                args=calib_args,
                callbacks=[drift_recorder],
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    padding=True,
                ),
            )
            trainer_calib.train()

            steps_done = drift_recorder.step
            current_tau = drift_recorder.tau
            print(
                f"[Scheduler] After CALIBRATION, tau={current_tau:.6f}, "
                f"steps_done={steps_done}/{total_steps_current}"
            )

            # 用前 steps_per_day 步的 tau 作为模型的一天
            model_day = max(current_tau, 1e-8)
            trigger_thresholds = [d * model_day for d in trigger_days_human]
            calibrated = True
            print(
                f"[Scheduler] Calibrated model_day={model_day:.6f}. "
                f"trigger_thresholds={trigger_thresholds}"
            )

            # new
            # initialize mu0 and mu based on calibration phase
            if steps_done > 0:
                drift_recorder.mu0 = current_tau / float(steps_done)
                drift_recorder.mu = drift_recorder.mu0
                drift_recorder.ema_alpha = ema_alpha
                print(f"[Scheduler] Initialized mu0={drift_recorder.mu0:.6f}")

            # 校准阶段结束后兜底保存一次
            model.save_pretrained(output_dir)
            del trainer_calib
            gc.collect()
            torch.cuda.empty_cache()

            # 进入下一轮循环（下一轮开始时会先检查是否已经跨过第一个阈值）
            continue

        # ====== 已校准：先检查当前 tau 是否已经跨过下一个阈值（即时触发回放） ======
        steps_done = drift_recorder.step
        current_tau = drift_recorder.tau
        print(
            f"\n[Scheduler] Loop start: tau={current_tau:.6f}, "
            f"steps_done={steps_done}/{total_steps_current}, "
            f"next_trigger_idx={next_trigger_idx}"
        )

        if (
            memory_data is not None
            and next_trigger_idx < len(trigger_thresholds)
            and current_tau >= trigger_thresholds[next_trigger_idx]
        ):
            phase_idx += 1
            print(
                f"=== Phase {phase_idx}: MEMORY REPLAY (immediate), "
                f"trigger_threshold={trigger_thresholds[next_trigger_idx]:.6f}, "
                f"tau={current_tau:.6f} ==="
            )
            
            # new
            # compute replay strength beta_t based on current update intensity
            mu = drift_recorder.mu if drift_recorder.mu is not None else drift_recorder.mu0
            mu0 = drift_recorder.mu0 if drift_recorder.mu0 is not None else 1.0
            if mu is None:
                mu = mu0
            instability_ratio = mu / (mu0 + 1e-12)
            scale = 1.0 + beta_scale * (instability_ratio - 1.0)

            # new 12.15
            # ============================
            # Log replay strength
            # ============================
            os.makedirs(output_dir, exist_ok=True)
            with open(replay_strength_log_path, "a", encoding="utf-8") as f:
                if (not replay_strength_header_written) and f.tell() == 0:
                    f.write("step,scale,beta_min,beta_max,mu,mu0\n")
                    replay_strength_header_written = True

                f.write(
                    f"{drift_recorder.step},"
                    f"{scale:.6f},"
                    f"{beta_min:.6f},{beta_max:.6f},"
                    f"{mu:.6e},{mu0:.6e}\n"
                )

            if scale < beta_min:
                scale = beta_min
            elif scale > beta_max:
                scale = beta_max
            beta_t = beta_base * scale
            print(
                f"[Scheduler] Replay strength beta_t={beta_t:.6f} "
                f"(mu={mu:.6e}, mu0={mu0:.6e}, r={instability_ratio:.4f})"
            )

            replay_args = transformers.TrainingArguments(
                per_device_train_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=5,
                num_train_epochs=memory_epochs,
                learning_rate=learning_rate,
                fp16=True,
                logging_steps=10,
                optim="adamw_torch",
                save_strategy="no",
                output_dir=output_dir,
                report_to="wandb" if (len(wandb_project) > 0) else None,
            )

            # new
            trainer_memory = ReplayTrainer(
                beta_t=beta_t,
                theta_star=theta_star,
                importance=importance,
                model=model,
                train_dataset=memory_data,
                eval_dataset=None,
                args=replay_args,
                callbacks=[],  # 回放不统计 drift
                data_collator=transformers.DataCollatorForSeq2Seq(
                    tokenizer,
                    pad_to_multiple_of=8,
                    return_tensors="pt",
                    padding=True,
                ),
            )
            trainer_memory.train()
            drift_recorder.update_reference_point(model)

            # ============================
            # Log replay loss
            # ============================
            with open(replay_loss_log_path, "a", encoding="utf-8") as f:
                if (not replay_loss_header_written) and f.tell() == 0:
                    f.write("step,base_loss,loss_reg,beta_t,beta_loss_reg\n")
                    replay_loss_header_written = True

                base_loss = trainer_memory.last_base_loss
                loss_reg = trainer_memory.last_loss_reg
                beta_loss_reg = beta_t * loss_reg if loss_reg is not None else None

                f.write(
                    f"{drift_recorder.step},"
                    f"{base_loss:.6f},{loss_reg:.6e},"
                    f"{beta_t:.6f},{beta_loss_reg:.6e}\n"
                )



            del trainer_memory
            gc.collect()
            torch.cuda.empty_cache()

            next_trigger_idx += 1
            print(
                f"[Scheduler] One replay finished. "
                f"next_trigger_idx={next_trigger_idx}, tau(still)={drift_recorder.tau:.6f}"
            )

            # 回放后继续下一轮（不推进 steps_done）
            continue

        remaining_steps = total_steps_current - steps_done
        if remaining_steps <= 0:
            break

        phase_idx += 1
        print(
            f"=== Phase {phase_idx}: CONTINUE CURRENT, "
            f"remaining_steps={remaining_steps}, "
            f"next_trigger_idx={next_trigger_idx} ==="
        )

        target_tau = None
        tau_callback = None
        if memory_data is not None and next_trigger_idx < len(trigger_thresholds):
            target_tau = trigger_thresholds[next_trigger_idx]
            tau_callback = TauTriggerStopCallback(drift_recorder, target_tau)
            print(f"[Scheduler] Will watch tau to reach target_tau={target_tau:.6f}")

        current_args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=10,
            max_steps=remaining_steps,
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="steps",
            save_steps=40,
            save_total_limit=2,
            output_dir=output_dir,
            report_to="wandb" if (len(wandb_project) > 0) else None,
            run_name=f"{wandb_run_name}_phase_{phase_idx}"
            if (len(wandb_run_name) > 0)
            else None,
        )

        callbacks = [drift_recorder]
        if tau_callback is not None:
            callbacks.append(tau_callback)

        trainer_current = MyTrainer(
            model=model,
            train_dataset=current_data,
            eval_dataset=None,
            args=current_args,
            callbacks=callbacks,
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )
        trainer_current.train()

        steps_done = drift_recorder.step
        current_tau = drift_recorder.tau
        print(
            f"[Scheduler] After CURRENT phase, tau={current_tau:.6f}, "
            f"steps_done={steps_done}/{total_steps_current}"
        )

        model.save_pretrained(output_dir)
        del trainer_current
        gc.collect()
        torch.cuda.empty_cache()


    # ====================== 所有 current 阶段完成 ======================
    print("\nAll CURRENT phases completed for this task.")

    if memory_data is None:
        # 第一个任务或者没有 memory_data 的情况，不做任何回放
        print("[Scheduler] memory_data is None (e.g., task_id == 0), so no replay was performed.")
    else:
        # ====== 统一的最终回放：保证每个任务结束时，至少回放一次 memory_data ======
        phase_idx += 1
        print(
            f"=== Phase {phase_idx}: FINAL MEMORY REPLAY after all current steps, "
            f"memory_epochs={memory_epochs} ==="
        )

        # 复用与阈值触发回放一致的 beta_t 设计
        mu = drift_recorder.mu if drift_recorder.mu is not None else drift_recorder.mu0
        mu0 = drift_recorder.mu0 if drift_recorder.mu0 is not None else 1.0
        if mu is None:
            mu = mu0
        instability_ratio = mu / (mu0 + 1e-12)
        scale = 1.0 + beta_scale * (instability_ratio - 1.0)
        if scale < beta_min:
            scale = beta_min
        elif scale > beta_max:
            scale = beta_max
        beta_t = beta_base * scale
        print(
            f"[Scheduler] FINAL replay strength beta_t={beta_t:.6f} "
            f"(mu={mu:.6e}, mu0={mu0:.6e}, r={instability_ratio:.4f})"
        )

        final_replay_args = transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=memory_epochs,   
            learning_rate=learning_rate,
            fp16=True,
            logging_steps=10,
            optim="adamw_torch",
            save_strategy="no",
            output_dir=output_dir,
            report_to="wandb" if (len(wandb_project) > 0) else None,
        )

        final_trainer_memory = ReplayTrainer(
            beta_t=beta_t,
            theta_star=theta_star,
            importance=importance,
            model=model,
            train_dataset=memory_data,
            eval_dataset=None,
            args=final_replay_args,
            callbacks=[],  # 回放不统计 drift，避免 memory 影响 tau
            data_collator=transformers.DataCollatorForSeq2Seq(
                tokenizer,
                pad_to_multiple_of=8,
                return_tensors="pt",
                padding=True,
            ),
        )

        final_trainer_memory.train()

        drift_recorder.update_reference_point(model)

        del final_trainer_memory
        gc.collect()
        torch.cuda.empty_cache()

        model.save_pretrained(output_dir)
        print("[Scheduler] Final memory replay completed and model saved.")

if __name__ == "__main__":
    fire.Fire(train)
