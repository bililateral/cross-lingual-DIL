// Mechanical transcription of already-opened research results. No model,
// labels, prediction arrays, metric recomputation or resampling is used.
import fs from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const root = process.cwd();
const out = path.join(root, 'outputs/20261022_continual_review');
const here = path.join(root, 'reports/documentation/20261022/continual_review');
const read = async p => fs.readFile(path.join(root, p), 'utf8');
const json = async p => JSON.parse((await read(p)).replace(/^\uFEFF/, ''));
const metrics = ['average_precision','trapezoidal_pr_auc','roc_auc','recall_at_fpr_1pct','brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc','map','mrr','recall_at_1','recall_at_3','recall_at_5','recall_at_10','ndcg_at_1','ndcg_at_3','ndcg_at_5','ndcg_at_10'];
const labels = ['AP','梯形PR-AUC','ROC-AUC','Recall@FPR≤1%','Brier↓','Log-loss↓','Precision','Recall','F1','Specificity','Balanced accuracy','MCC','MAP','MRR','Recall@1','Recall@3','Recall@5','Recall@10','NDCG@1','NDCG@3','NDCG@5','NDCG@10'];
const configs = [
 ['E01','09-07','问题形成','固定特征四臂','57→32→1；冻结／SEQ／512KiB ER／累计重训','同源500训练＋500开发世界；5批；到达种子11/23/37','51,015物理更新；编码器不训练','没有稳定域级遗忘；ER相对SEQ AP−0.005869','docs/SELLER_ALIAS_CONTINUAL_RESULT.zh.md'],
 ['E02','09-08～09','问题形成','LaBSE文本四臂','LaBSE及文本头续训；冻结／SEQ／512KiB ER／累计','同源500训练＋500开发世界；5批；三顺序','完整42评分点／12模型','SEQ未持续退化；ER−SEQ AP−0.003249250','docs/SELLER_ALIAS_TEXT_RESULT.zh.md'],
 ['E03','09-09','问题形成','新群体三域四臂','LaBSE；冻结／SEQ／6群ER／累计；每阶段3轮','主题比例变化，表达机制共享；ABC/BCA/CAB','5,400物理更新','SEQ旧域AP平均上升；ER−SEQ AP−0.064915177','docs/SELLER_ALIAS_POPULATION_RESULT.zh.md'],
 ['E04','09-10','机制诊断','训练表达重分配','原路径与后续训练表达重分配路径；共同首域','原主题域，三顺序；复合干预','2,700更新／15点','首域下降且伤新域；不是自然遗忘成功或唯一原因证明','docs/SELLER_ALIAS_CAUSE_TRAIN_RESULT.zh.md'],
 ['E05','09-10','LaBSE表达域','SEQ与单域／联合诊断','LaBSE＋BCE；单域3轮、SEQ各3轮、联合3轮','主题＋表达机制变化；60 train/20 valid每域；三顺序','2,160更新；11评分点','首域AP遗忘0.058932；联合兼顾失败','docs/SELLER_ALIAS_EXPRESSION_RESULT.zh.md'],
 ['E06','09-10','LaBSE表达域','联合延长6轮','从共同初始重建540步再延长540步，Adam连续','同E05；联合可见全部三域，离线能力参照','1,080新物理更新','三域AP兼顾点估计门通过；原3轮失败保留','docs/SELLER_ALIAS_JOINT_RESULT.zh.md'],
 ['E07','09-15','LaBSE表达域','随机完整群ER','BCE(current)+BCE(history)；6群／1MiB','共享E05首域；三顺序；每阶段3轮','1,620更新含首域重建','首域AP低于SEQ；完整缓解条件未过','docs/SELLER_ALIAS_REPLAY_RESULT.zh.md'],
 ['E08','09-15','LaBSE诊断','缓存内外固定模型探针','不训练；共同起点、SEQ与ER固定模型','曾重放6群／未重放54群／原valid20群','固定模型评价','ER缓存MAP0.924936、valid首域MAP0.390481；不单独确证因果','docs/SELLER_ALIAS_REPLAY_PROBE_RESULT.zh.md'],
 ['E09','09-15','LaBSE表达域','历史logit保持','BCE(current)+BCE(history)+0.5历史logit MSE','共享首域，6群／1MiB，三顺序各3轮','1,620更新含首域重建','首域AP保持胜SEQ和ER；新域AP对SEQ下降0.059000','docs/SELLER_ALIAS_DISTILLATION_RESULT.zh.md'],
 ['B01','09-30～10-01','BGE三域','frozen／SEQ／ER1／LOGIT1','L=BCE+rank+0.5hard；ER=Lc+Lh；LOGIT再加0.5D','s0；ABC/BCA/CAB；48fit/12cal/20valid；6群／1MiB','6轮×48群=288更新/阶段；6,048物理更新','遗忘观察通过；ER−SEQ4/23；LOGIT−ER23/23','docs/SELLER_ALIAS_BGE_RESULT.zh.md'],
 ['B02','10-01～02','BGE三域','ER0.5／ER0.25','Lc+λLh，λ=0.5或0.25；共享首域','同B01；只新训后两阶段','3,456新更新','两臂对ER1均23/23；对SEQ均4/23；按原规则选0.25','docs/SELLER_ALIAS_ER_WEIGHT_RESULT.zh.md'],
 ['B03','10-02','BGE三域','ER0.1','Lc+0.1Lh；共享首域','同B01','1,728新更新','对ER0.25为23/23；对SEQ10/23','docs/SELLER_ALIAS_ER_LOW_RESULT.zh.md'],
 ['B04','10-02','BGE三域','LOGIT0.25','Lc+0.25Lh+0.5D；MSE不再乘0.25','同B01；共享首域','1,728新更新','对匹配ER0.25为23/23；对SEQ16/23','docs/SELLER_ALIAS_LOGIT_WEIGHT_RESULT.zh.md'],
 ['B05','10-03','BGE三域','LOGIT0.1','Lc+0.1Lh+0.5D','同B01；共享首域','1,728新更新','对ER0.1及LOGIT0.25均23/23；对SEQ20/23','docs/SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md'],
 ['B06','10-03～04','BGE三域','risk','Lc+0.1Lh+0.5 mean(D_rank,D_positive,D_negative)','同B01；分层、排序、单向风险分布保持','1,728新更新','对LOGIT0.1仅4/23；O/N/Z MAP负向区间','docs/SELLER_ALIAS_LOGIT_RISK_RESULT.zh.md'],
 ['B07','10-05','BGE三域','关系目标记忆原版','d32、ridge0.001；当前/历史平方识别＋排序各权重1；关闭dropout','s0三顺序；新头从预训练独立开始；6群／1MiB','2,592更新','对LOGIT0.1继续条件0/6；不是同起点单机制消融','docs/SELLER_ALIAS_RELATION_MEMORY_RESULT.zh.md'],
 ['B08','10-06','BGE三域','关系目标记忆修订','当前B＋历史Q＋0.1R；d32、ridge0.001；关闭dropout','三顺序独立首域；6群／1MiB','2,592更新','继续条件0/6；该修改版活动路线关闭','docs/SELLER_ALIAS_RELATION_REVISION_RESULT.zh.md'],
 ['B09','10-06～07','BGE三域','函数目标累计记忆','8192→128→1；d129、ridge0.001；Bc+0.1Bh_det+0.5D累计函数统计','共享LOGIT首域；6群／1MiB','1,728新更新','继续条件0/5；O/N/Z MAP均负向','docs/SELLER_ALIAS_FUNCTION_MEMORY_RESULT.zh.md'],
 ['B10','10-07～08','BGE三域记录','记录C／S','记录头4096→128→1；A0/A1监督均值；0.1历史；C=0.25D0+0.25D1，S=0.5D0','共同记录首域；s0三顺序；6群／1MiB','4,320更新／15状态','C−LOGIT原5/5，C−S4/5；整体未过；事后23项另列','reports/seller_alias_continual/20261008/record_replay_result/report.zh.txt'],
 ['B11','10-08～09','BGE三域记录','五臂：LOGIT／C／S／C_plus／S_strong','C_plus=0.5D0+0.25D1；S_strong=0.75D0；监督相同','四记录臂共同首域；LOGIT独立重训；有历史R1差异','10,368更新／36物理状态','C_plus继续条件false；相对S新域损失；该增强配置关闭','reports/seller_alias_continual/20261009/replay_improvement_result/report.zh.txt'],
 ['B12','10-09～11','BGE三域归因','LOGIT0.1／R0／完整S','R0记录网络无重组；S同网络加A0/A1监督；均0.1历史＋0.5D0','各臂独立首域、同预训练与供给；s0三顺序','7,776更新／27状态','R0旧域收益；S−R0的O/N/Z MAP区间均正；无新总门','reports/seller_alias_continual/20261011/record_attribution_result/report.zh.txt'],
 ['A01','10-12','固定A→B','S／C_cf／C_perm','C_cf替换D0为全域与折外共同支持的单边间隔；C_perm一次可行错配','s0；共享A；U0/U1各用另一折24群；6群／1MiB','1,440更新含两来源各144；4端点','C_cf−S及C_cf−C_perm的A/B MAP四区间跨零','reports/seller_alias_continual/20261013/crossfit_retention_result/report.zh.txt'],
 ['A02','10-13','固定A→B','S／Q／C_q／C_b','Q自然查询锚定历史增强；C_q=Q+0.25R；C_b固定替代增强＋同R；保留0.5D0','s0；共享A；四支B；当前均完整S','1,440更新／5端点','四比较八个MAP区间均跨零；无整体全过门','reports/seller_alias_continual/20261014/query_replay_result/report.zh.txt'],
 ['A03','10-15','固定A→B','S／G／F','S均值；G=logmeanexp(K),τ1；F均匀双边际熵正则最优值；监督及自身D0一致','s0；各臂独立A；6群／1MiB','1,728更新／6端点；旧数值失败另保留','F主要收益未建立；G旧A局部正向，新B未确定','reports/seller_alias_continual/20261015/balanced_pooling_result/report.zh.txt'],
 ['A04','10-16','固定A→B','H=G_half','H=0.5mean(K)+0.5logmeanexp(K)，τ1；无新教师项','s0；只新训H；复用S/G开放保存结果','576更新／2端点','对S旧A正向，新B跨零；双域门false','reports/seller_alias_continual/20261016/g_half_result/report.zh.txt'],
 ['A05','10-16','固定A→B','H_cons','H加自然/重组查询分布JS，λ1、τ1；当前C＋0.1历史C；0.5D0','s0；只新训H_cons；复用S/H保存结果','576更新／2端点','相对H新B改善但旧A受损；对S双域门false','reports/seller_alias_continual/20261016/h_consistency_result/report.zh.txt'],
 ['A06','10-18','固定A→B','S_pos／H_pos','只将rank换逐正例log(1+Σneg exp(z_n−z_p))；完整监督两视图、历史均替换','s0；各自独立A；复用S/H保存结果','1,152更新／4端点','S_pos−S A下界略负、B跨零；H_pos也未双域超过S','reports/seller_alias_continual/20261018/positive_ranking_result/report.zh.txt'],
 ['O01','10-19起','Optuna开发','S／S_pos／H_pos三臂','各12搜索：rank λ∈[0.25,2]、LR倍率∈[0.5,2]对数；各3新seed复验','搜索s0；复验s1/2/3；同预算与U=min(ΔA MAP,ΔB MAP)','36搜索＋9复验；272小时现行总限','本回顾不读取运行中或未开放结果；已存10-21 17:05记录25/36、0/9，非实时','docs/SELLER_ALIAS_OPTUNA.zh.md']
];
const sourceRows=[];
async function sourceId(p) {
 let n=sourceRows.findIndex(r=>r[1]===p); if(n>=0)return sourceRows[n][0];
 const b=await fs.readFile(path.join(root,p)); const id=`S${String(sourceRows.length+1).padStart(3,'0')}`;
 sourceRows.push([id,p,b.length,createHash('sha256').update(b).digest('hex')]); return id;
}
const long=[];
// Columns: experiment, kind, arm/contrast, role, endpoint/domain, metric,
// mean/lo/hi, per-order ABC/BCA/CAB, original JSON key, source identity.
function leafRows(obj, p=[], dest=[]) {
 if(!obj || typeof obj!=='object')return dest;
 if('mean' in obj && metrics.includes(p.at(-1))) { dest.push([p,obj]); return dest; }
 for(const[k,v]of Object.entries(obj))leafRows(v,[...p,k],dest); return dest;
}
function pushLeaf(id,kind,name,role,ep,metric,v,key,sid) {
 const ci=v.conditional_95pct_interval??v.ci95??[null,null]; const po=v.per_order??{};
 long.push([id,kind,name,role,ep,metric,v.mean,ci[0]??null,ci[1]??null,po.ABC??null,po.BCA??null,po.CAB??null,key,sid]);
}
const jSources=[
 ['B01','20261001/bge_continual_result/returned/job/evaluation/evaluation.json', ['frozen','seq','er','logit']],
 ['B02','20261002/er_weight_result/returned/job/evaluation/evaluation.json',['half','quarter']],
 ['B03','20261003/er_low_result/job/evaluation/evaluation.json',['tenth']],
 ['B04','20261003/logit_weight_result/job/evaluation/evaluation.json',['logit_quarter']],
 ['B05','20261004/logit_low_result/job/evaluation/evaluation.json',['logit_tenth']],
 ['B06','20261004/risk_result/job/evaluation/evaluation.json',['risk']],
 ['B07','20261006/relation_result/job/evaluation/evaluation.json',['relation']],
 ['B08','20261006/relation_revision_result/job/evaluation/evaluation.json',['relation']],
 ['B09','20261007/function_memory_result/job/evaluation/evaluation.json',['function_memory']],
 ['B11','20261009/replay_improvement_result/job/evaluation/evaluation.json',null]
];
for (const[id,rel,keep]of jSources) {
 const p='reports/seller_alias_continual/'+rel,j=await json(p),sid=await sourceId(p);
 for(const [keys,v]of leafRows(j.endpoints)) {
  const[arm,role,ep,metric]=keys; if(keep&&!keep.includes(arm))continue;
  if(keys.length!==4)throw Error('Unexpected endpoint '+keys);
  pushLeaf(id,'absolute',arm,role,ep,metric,v,'endpoints.'+keys.join('.'),sid);
 }
 for(const[keys,v]of leafRows(j.comparisons)) {
  const metric=keys.at(-1),ep=keys.at(-2),role=keys.slice(1,-2).join('/');
  pushLeaf(id,'contrast',keys[0],role,ep,metric,v,'comparisons.'+keys.join('.'),sid);
 }
 for(const[keys,v]of leafRows(j.delta))pushLeaf(id,'contrast','candidate-LOGIT0.1','primary',keys[0],keys.at(-1),v,'delta.'+keys.join('.'),sid);
}
{
 const p='reports/seller_alias_continual/20261008/record_replay_result/common_evaluation/calculation/statistics.json',j=await json(p),sid=await sourceId(p);
 for(const[keys,v]of leafRows(j.results)) {
  const[name,role,ep,metric]=keys;if(name==='LOGIT0.1')continue;
  pushLeaf('B10',name.includes('-')?'contrast':'absolute',name,role,ep,metric,v,'results.'+keys.join('.'),sid);
 }
}
function csvRows(t) {
 const a=[];let row=[],s='',q=false;
 for(let i=0;i<t.length;i++){const c=t[i];if(c==='"'){if(q&&t[i+1]==='"'){s+='"';i++;}else q=!q;}else if(c===','&&!q){row.push(s);s='';}else if(c==='\n'&&!q){row.push(s.replace(/\r$/,''));a.push(row);row=[];s='';}else s+=c;}
 if(s||row.length){row.push(s.replace(/\r$/,''));a.push(row);}const h=a.shift();return a.filter(r=>r.length===h.length).map(r=>Object.fromEntries(h.map((k,i)=>[k.replace(/^\uFEFF/,''),r[i]])));
}
function num(v) { if(v==null||v===''||/^(N\/A|nan|null)$/i.test(v))return null;const n=Number(v);if(!Number.isFinite(n))throw Error('Invalid number '+v);return n; }
{
 const p='reports/seller_alias_continual/20261011/record_attribution_result/analysis/all_endpoints.csv',sid=await sourceId(p);
 for(const r of csvRows(await read(p)))long.push(['B12',r.type,r.method_or_comparison,r.role,r.endpoint,r.metric,num(r.mean),num(r.lo95),num(r.hi95),num(r.ABC),num(r.BCA),num(r.CAB),'CSV',sid]);
}
const ab=[];
const abSources=[['A01','20261013/crossfit_retention_result'],['A02','20261014/query_replay_result'],['A03','20261015/balanced_pooling_result'],['A04','20261016/g_half_result'],['A05','20261016/h_consistency_result'],['A06','20261018/positive_ranking_result']];
for(const[id,rel]of abSources){const p='reports/seller_alias_continual/'+rel+'/analysis/all_metrics.csv',sid=await sourceId(p);for(const r of csvRows(await read(p)))ab.push([id,r.kind,r.point_or_contrast,r.role,r.domain,r.metric,num(r.mean),num(r.lower_95),num(r.upper_95),'原已开放保存统计',sid]);}
// Apply only the already-reviewed, saved 18-row sign correction. Keep the
// original source and corrected-source references; never mutate old files.
{
 const p='reports/seller_alias_continual/20261015/balanced_pooling_result/review/loss_forgetting_direction.csv',sid=await sourceId(p);
 let count=0;for(const r of csvRows(await read(p))){const dst=ab.find(x=>x[0]==='A03'&&x[1]===r.kind&&x[2]===r.contrast&&x[3]===r.role&&x[4]===r.domain&&x[5]===r.metric);if(!dst)throw Error('Missing reviewed correction');dst[6]=num(r.loss_worsening_mean_BminusA);dst[7]=num(r.loss_worsening_low);dst[8]=num(r.loss_worsening_high);dst[9]='已审勘误：损失型遗忘为B末−A末；原件保持';dst[10]+=';'+sid;count++;}if(count!==18)throw Error('Correction count '+count);
}
const early=[];
for(const cfg of configs.filter(r=>r[0].startsWith('E'))){const p=cfg[8],sid=await sourceId(p),lines=(await read(p)).split(/\r?\n/);let section='',header=null;for(let i=0;i<lines.length;i++){const l=lines[i];if(l.startsWith('#'))section=l.replace(/^#+\s*/, '');if(!l.trim().startsWith('|')){header=null;continue;}const c=l.trim().split('|').slice(1,-1).map(x=>x.trim());if(c.every(x=>/^:?-+:?$/.test(x)))continue;if(!header){header=c;continue;}for(let k=1;k<c.length;k++){const v=c[k].replace(/−/g,'-').replace(/[＋]/g,'+');if(/^[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+\-]?\d+)?$/i.test(v))early.push([cfg[0],section,c[0],header[k]??String(k),Number(v),null,sid]);else if(/\d/.test(v)&&!/^[-]+$/.test(v))early.push([cfg[0],section,c[0],header[k]??String(k),null,c[k],sid]);}}}
// Preserve every source's stated metric definitions; no zero-to-N/A retrofit
// for old contracts that explicitly used zero_division=0.
const mapNames={frozen:'frozen',seq:'SEQ',er:'ER1',logit:'LOGIT1',half:'ER0.5',quarter:'ER0.25',tenth:'ER0.1',logit_quarter:'LOGIT0.25',logit_tenth:'LOGIT0.1',relation:'关系目标记忆',function_memory:'函数目标记忆',G_half:'H=G_half'};
for(const r of long)if(r[1]==='absolute')r[2]=r[0]==='B08'?'关系目标记忆修订':(mapNames[r[2]]??r[2]);
const triSummary=[];
for(const id of [...new Set(long.filter(r=>r[1]==='absolute').map(r=>r[0]))])for(const arm of [...new Set(long.filter(r=>r[0]===id&&r[1]==='absolute').map(r=>r[2]))]){
 const get=(ep,m)=>{const a=long.filter(r=>r[0]===id&&r[1]==='absolute'&&r[2]===arm&&r[3]==='primary'&&r[4]===ep&&r[5]===m);if(a.length!==1)throw Error(`Expected unique ${id}/${arm}/${ep}/${m}, got ${a.length}`);return a[0][6];};
 triSummary.push([id,arm,...['O','N','Z','final_all','F_first','F','G'].map(e=>get(e,'map')),...['O','N','Z'].map(e=>get(e,'recall_at_5'))]);
}
triSummary.sort((a,b)=>a[0].localeCompare(b[0]));
const abSummary=[];
for(const[id]of abSources)for(const pt of [...new Set(ab.filter(r=>r[0]===id&&r[1]==='absolute'&&r[2].startsWith('B_')).map(r=>r[2]))]){
 const g=(dom,m,role='raw')=>{const a=ab.filter(r=>r[0]===id&&r[1]==='absolute'&&r[2]===pt&&r[3]===role&&r[4]===dom&&r[5]===m);if(a.length!==1)throw Error(`AB unique ${id}/${pt}/${role}/${dom}/${m} ${a.length}`);return a[0][6];};
 abSummary.push([id,pt,...['A','B'].map(d=>g(d,'map')),...['A','B'].map(d=>g(d,'recall_at_5')),...['A','B'].map(d=>g(d,'average_precision')),...['A','B'].map(d=>g(d,'brier','stage-cal')),...['A','B'].map(d=>g(d,'log_loss','stage-cal'))]);
}
for(const c of configs)await sourceId(c[8]);
await sourceId('reports/seller_alias_continual/20261011/record_attribution_result/review/primary_disposition.zh.txt');
const definitions=[
 ['统计覆盖','既有已开放结果的22项指标、合同端点、三种输出角色/primary、已保存条件区间及配对差。早期固定特征/文本有规则外12项，原表一并保留。逐群载荷、逐更新日志、全部六轮曲线仍在原报告链接，不复制进工作簿。'],
 ['本次操作','只读保存JSON/CSV/报告，转录和筛选；不重读正式文本/标签，不载模，不训练，不重算指标或bootstrap。不改原验收；无需新的结果外审。'],
 ['共同BGE设置','BAAI/bge-large-zh-v1.5；输入仅屏蔽标题/描述；每群28账号，378无向对，20正对；每查询27候选且1/2正例。48fit+12cal+20valid/域。'],
 ['训练','s0初始化20260909、日程根20260918；每阶段6轮×48群=288次AdamW调用。编码器峰值1e-5，29步warmup后降至0；head1e-3、clip1。记录表方案每阶段本身接续Adam。'],
 ['历史缓存','Algorithm R最多6完整群，原始数据及全部附属状态≤1MiB；成员固定到阶段末。归档的多份Memory不能合并成为学习器历史。'],
 ['O','三阶段最终两个旧域的均值：(R31+R32)/2，再三顺序平均。'],
 ['N / Z / A2','N=(R22+R33)/2；Z=R33；A2=R22。都是各顺序的到达域位置，不是固定真实域名字。'],
 ['final_all','(R31+R32+R33)/3=(2O+Z)/3。'],
 ['F_first / F / G','高值好指标：F_first=R11−R31；F=(R11−R31+R22−R32)/2；G=(R22−R12+R33−R23)/2。损失型按合同反向。'],
 ['A→B','A末与B末分别为完成A/B训练后的端点；主最终旧域为B末A、主新域为B末B。不能与三域O/N/Z当成同一口径直接排行。'],
 ['raw / stage-cal / first-cal','raw排序；stage-cal由当前域12校准群拟合正斜率仿射映射，用于主概率/分类；first-cal固定首映射，仅诊断。primary=raw14项+stage-cal8项。'],
 ['不确定性','BGE为按实际域20群配对的5,000次条件bootstrap；三顺序复用同一60群，且不是3个训练seed。区间不覆盖多次开发选择、重训或真实市场。早期抽样规则按各源报告。'],
 ['缺失与历史定义','空数值单元格=原结果未定义/未提供，不等于0。10月15日起precision/MCC严格传播N/A；更早合同零分母记0，保留原口径，跨批不可直接比较这些项。'],
 ['勘误','S/G/F的18行损失型遗忘采用既有review/loss_forgetting_direction.csv。C_q的24个violations_total导出0属于N/A；它们不是本表22项指标，不计零违例。'],
 ['归因报告已有勘误','B12 R0的G_MAP按保存统计及原主审为0.05243518626921002；原正文0.052435519为既有抄录误差。本表使用原始统计，不改冻结正文。'],
 ['同名方法','B10/B11与B12的S属于不同冻结批次；B11 LOGIT有历史复现差异。A01/A02的S保存分数相同不是独立seed。重复显示的对照不能当额外样本。'],
 ['验收','23项是预定条件集合，包含均值保护与区间条件，不是23个指标或23个显著性检验。区间跨零不证明等价。'],
 ['Optuna','仅列方案，未汇入进行中搜索/未开放复验成绩。10月21日17:05:21交接25/36搜索、0/9复验是历史时点，本回顾未查实时服务器。'],
 ['范围','主线问题形成、LaBSE表达域缓解、BGE三域有限历史、记录重组归因及A→B改进。基础编码器/联合排序预训练筛选、静态未执行提案和GPU资格不是抗遗忘方法新结果。']
];
for(let i=0;i<metrics.length;i++)definitions.push([metrics[i],labels[i]+(metrics[i]==='map'?'；逐查询AP再群宏平均':metrics[i]==='average_precision'?'；群内378无向账号对AP':metrics[i]==='recall_at_fpr_1pct'?'；经验ROC低FPR读数，不是已选部署阈值':'' )]);

await fs.mkdir(out,{recursive:true});
const fmt=x=>x==null?'N/A':typeof x==='number'?x.toFixed(6):String(x).replaceAll('|','／').replaceAll('\n',' ');
const mdTable=(h,rows)=>'| '+h.join(' | ')+' |\n| '+h.map(()=> '---').join(' | ')+' |\n'+rows.map(r=>'| '+r.map(fmt).join(' | ')+' |').join('\n')+'\n';
let md='# 持续学习抗遗忘主线实验回顾\n\n整理日期：2026-10-22。仅转录既有已开放结果，不产生新实验或新统计结论。\n\n';
md+=mdTable(['ID','日期','阶段','实验','配置','数据与路径','执行量','原结论','来源'],configs.map(r=>[...r.slice(0,8),`[原报告](<${path.join(root,r[8]).replaceAll('\\','/')}>)`]));
md+='\n## 三域主要结果\n\n'+mdTable(['ID','方法','O MAP','N MAP','Z MAP','最终全域MAP','F_first MAP','F MAP','G MAP','O R5','N R5','Z R5'],triSummary);
md+='\n## A→B主要结果\n\n'+mdTable(['ID','B末方法','A MAP','B MAP','A R5','B R5','A AP','B AP','A Brier(stage-cal)','B Brier(stage-cal)','A log-loss(stage-cal)','B log-loss(stage-cal)'],abSummary);
md+='\n## 所有22项指标：三域最终全域，primary\n\n';
for(const slice of [metrics.slice(0,12),metrics.slice(12)]){const rows=triSummary.map(s=>[s[0],s[1],...slice.map(m=>long.find(r=>r[0]===s[0]&&r[1]==='absolute'&&r[2]===s[1]&&r[3]==='primary'&&r[4]==='final_all'&&r[5]===m)[6])]);md+=mdTable(['ID','方法',...slice],rows)+'\n';}
md+='\n## 所有22项指标：A→B各域B末，primary\n\n排序及曲线14项使用raw；概率和阈值分类8项使用stage-cal。其余输出角色完整保留于工作簿。\n\n';
const calibrated=new Set(['brier','log_loss','precision','recall','f1','specificity','balanced_accuracy','mcc']);
for(const slice of [metrics.slice(0,12),metrics.slice(12)]){const rows=abSummary.flatMap(s=>['A','B'].map(d=>[s[0],s[1],d,...slice.map(m=>ab.find(r=>r[0]===s[0]&&r[1]==='absolute'&&r[2]===s[1]&&r[3]===(calibrated.has(m)?'stage-cal':'raw')&&r[4]===d&&r[5]===m)[6])]));md+=mdTable(['ID','方法','域',...slice],rows)+'\n';}
md+='\n## 口径\n\n'+mdTable(['项目','说明'],definitions);
await fs.writeFile(path.join(here,'tables.zh.md'),md);
console.log(JSON.stringify({rows:{config:configs.length,threeDomain:long.length,AB:ab.length,early:early.length},sources:sourceRows.length}));
if(!process.argv.includes('--xlsx'))process.exit(0);
const require=createRequire(path.join(here,'build.mjs'));
const {Workbook,SpreadsheetFile}=await import(pathToFileURL(require.resolve('@oai/artifact-tool')).href);
const wb=Workbook.create();
let tableNumber=0;
function sheet(name,headers,rows,widths,numericCols=[],height=23){
 const s=wb.worksheets.add(name);s.showGridLines=false;
 const all=s.getRangeByIndexes(0,0,rows.length+1,headers.length);all.values=[headers,...rows];
 all.format.font={name:'Microsoft YaHei',size:10,color:'#172435'};all.format.rowHeight=height;all.format.verticalAlignment='center';
 for(let c=0;c<headers.length;c++)s.getRangeByIndexes(0,c,rows.length+1,1).format.columnWidth=widths[c]??16;
 for(const c of numericCols){const r=s.getRangeByIndexes(1,c,rows.length,1);r.setNumberFormat('0.000000');r.format.horizontalAlignment='right';}
 const hdr=s.getRangeByIndexes(0,0,1,headers.length);hdr.format.fill='#243B55';hdr.format.font={bold:true,color:'#FFFFFF',name:'Microsoft YaHei',size:10};hdr.format.wrapText=true;hdr.format.rowHeight=42;hdr.format.horizontalAlignment='center';
 const t=s.tables.add(`A1:${col(headers.length)}${rows.length+1}`,true,'ResearchTable'+(++tableNumber));t.showFilterButton=true;
 s.freezePanes.freezeRows(1);s.freezePanes.freezeColumns(name==='口径与指标'?1:2);return s;
}
function col(n){let v='';for(;n>0;n=Math.floor((n-1)/26))v=String.fromCharCode(65+(n-1)%26)+v;return v;}
const tabs=[];
tabs.push(sheet('实验配置',['ID','日期(2026)','阶段','实验','配置','数据与路径','执行量','原结论','来源'],configs,[9,14,20,33,65,54,32,62,95],[],84));
tabs[0].getRange('D2:H'+(configs.length+1)).format.wrapText=true;
tabs[0].getRange('I2:I'+(configs.length+1)).format.wrapText=true;
tabs.push(sheet('三域核心',['ID','方法','O MAP','N MAP','Z MAP','最终全域MAP','首域遗忘MAP','两旧域遗忘MAP','新学习增益MAP','O Recall@5','N Recall@5','Z Recall@5'],triSummary,[9,28,...Array(10).fill(18)],Array.from({length:10},(_,i)=>i+2)));
tabs.push(sheet('AB核心',['ID','B末方法','A MAP','B MAP','A Recall@5','B Recall@5','A AP','B AP','A Brier(stage-cal)','B Brier(stage-cal)','A log-loss(stage-cal)','B log-loss(stage-cal)'],abSummary,[9,25,...Array(10).fill(19)],Array.from({length:10},(_,i)=>i+2)));
tabs.push(sheet('三域全部指标',['ID','类型','方法或比较','输出角色','端点','指标','均值','95%下界','95%上界','ABC','BCA','CAB','原字段路径','来源ID'],long,[9,14,34,30,16,29,16,16,16,16,16,16,76,13],[6,7,8,9,10,11]));
tabs.push(sheet('AB全部指标',['ID','类型','端点或比较','输出角色','域','指标','均值','95%下界','95%上界','解释或勘误','来源ID'],ab,[9,25,30,16,9,29,16,16,16,58,17],[6,7,8]));
tabs.push(sheet('早期原表',['ID','原表所在章节','原行名','原列名','数值','原复合值或区间','来源ID'],early,[9,51,37,36,18,68,13],[4]));
tabs.push(sheet('口径与指标',['项目','说明'],definitions,[33,135],[],42));tabs.at(-1).getRange('B2:B'+(definitions.length+1)).format.wrapText=true;
tabs.push(sheet('来源',['来源ID','项目相对路径','字节','SHA256'],sourceRows,[13,145,15,72],[],42));tabs.at(-1).getRange('C2:C'+(sourceRows.length+1)).setNumberFormat('#,##0');tabs.at(-1).getRange('B2:B'+(sourceRows.length+1)).format.wrapText=true;
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'三域核心!A1:L6',include:'values,formulas',tableMaxRows:6,tableMaxCols:12,maxChars:4000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000})).ndjson);
await fs.mkdir(path.join(out,'previews'),{recursive:true});
const ranges=['A1:D9','A1:H10','A1:H10','A1:I10','A1:I10','A1:E10','A1:B12','A1:C8'];
for(let i=0;i<tabs.length;i++){const blob=await wb.render({sheetName:tabs[i].name,range:ranges[i],scale:1,format:'png'});await fs.writeFile(path.join(out,'previews',`${i+1}.png`),new Uint8Array(await blob.arrayBuffer()));}
for(const [label,sheetName,range]of [['config','实验配置','E10:I15'],['definitions','口径与指标','A13:B19']]){const blob=await wb.render({sheetName,range,scale:1,format:'png'});await fs.writeFile(path.join(out,'previews',label+'.png'),new Uint8Array(await blob.arrayBuffer()));}
await(await SpreadsheetFile.exportXlsx(wb)).save(path.join(out,'mainline_experiments.xlsx'));
console.log('EXPORTED '+path.join(out,'mainline_experiments.xlsx'));
