from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import MappingProxyType
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_balanced_world_schedule_v9_4 as schedule_v94
import step28_v13_v1_13_joint_noise_signatures_v9_4 as signatures_v94
import step28_v13_v1_13_quality_probe_labels_v9_4 as labels_v94
import step28_v13_v1_13_quality_probe_policy_v9_4 as policy_v94


class FormalInputChainV94Contracts(unittest.TestCase):
    def test_real_upstream_capabilities_close_the_complete_input_chain(self) -> None:
        train_schedule = schedule_v94.build_split_schedule("train")
        development_schedule = schedule_v94.build_split_schedule(
            "development"
        )
        noise_signatures = signatures_v94.build_noise_signatures()
        time_key_hex = "01" * 32
        time_key_commitment = hashlib.sha256(
            bytes.fromhex(time_key_hex)
        ).hexdigest()
        bundle = policy_v94._assemble_formal_inputs_after_authorization(
            train_schedule=train_schedule,
            development_schedule=development_schedule,
            noise_signature_set=noise_signatures,
            time_key_hex=time_key_hex,
            expected_noise_signature_rows_sha256=(
                signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
            ),
            expected_noise_signature_set_commitment_sha256=(
                signatures_v94.EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
            ),
            expected_time_key_commitment_sha256=time_key_commitment,
        )
        self.assertEqual(
            bundle["train_prepared"].matrix.values.shape,
            (500 * 378, 14),
        )
        self.assertEqual(
            bundle["development_prepared"].matrix.values.shape,
            (500 * 378, 14),
        )
        receipt = bundle["schedule_pair_receipt"]
        self.assertNotEqual(
            receipt["train_latent_schedule_sha256"],
            receipt["development_latent_schedule_sha256"],
        )
        self.assertTrue(receipt["fixed_global_relabel_rejected"])
        with self.assertRaisesRegex(
            labels_v94.QualityProbeLabelsV94Error,
            "already been consumed",
        ):
            labels_v94._open_controller_truth_after_preparation(
                prepared=bundle["train_prepared"],
                schedule=train_schedule,
            )
        with self.assertRaisesRegex(
            policy_v94.QualityProbePolicyV94Error,
            "upstream capability drift|matrix/join drift",
        ):
            policy_v94._validate_formal_inputs(
                train_prepared=bundle["train_prepared"],
                development_prepared=bundle["development_prepared"],
                train_labels=bundle["train_labels"],
                development_labels=bundle["development_labels"],
                train_schedule=train_schedule,
                development_schedule=development_schedule,
                schedule_pair_receipt=receipt,
                noise_signature_set=noise_signatures,
                expected_noise_signature_rows_sha256=(
                    signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
                ),
                expected_noise_signature_set_commitment_sha256=(
                    signatures_v94.EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
                ),
                expected_time_key_commitment_sha256="f" * 64,
            )

        forged_receipt_payload = dict(receipt)
        forged_receipt_payload["fixed_global_relabel_rejected"] = False
        forged_receipt = MappingProxyType(forged_receipt_payload)
        with self.assertRaisesRegex(
            policy_v94.QualityProbePolicyV94Error,
            "upstream capability drift",
        ):
            policy_v94._validate_formal_inputs(
                train_prepared=bundle["train_prepared"],
                development_prepared=bundle["development_prepared"],
                train_labels=bundle["train_labels"],
                development_labels=bundle["development_labels"],
                train_schedule=train_schedule,
                development_schedule=development_schedule,
                schedule_pair_receipt=forged_receipt,
                noise_signature_set=noise_signatures,
                expected_noise_signature_rows_sha256=(
                    signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
                ),
                expected_noise_signature_set_commitment_sha256=(
                    signatures_v94.EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
                ),
                expected_time_key_commitment_sha256=time_key_commitment,
            )

        forged_rows = list(noise_signatures.rows)
        first = dict(forged_rows[0])
        first["title_present_mask"] = (
            "0" if first["title_present_mask"][0] == "1" else "1"
        ) + first["title_present_mask"][1:]
        forged_rows[0] = MappingProxyType(first)
        forged_signatures = signatures_v94.NoiseSignatureSet(
            rows=tuple(forged_rows),
            commitment=noise_signatures.commitment,
        )
        with self.assertRaisesRegex(
            signatures_v94.JointNoiseSignaturesV94Error,
            "commitment drift|row drift",
        ):
            policy_v94._assemble_formal_inputs_after_authorization(
                train_schedule=train_schedule,
                development_schedule=development_schedule,
                noise_signature_set=forged_signatures,
                time_key_hex=time_key_hex,
                expected_noise_signature_rows_sha256=(
                    signatures_v94.EXPECTED_SIGNATURE_ROWS_SHA256
                ),
                expected_noise_signature_set_commitment_sha256=(
                    signatures_v94.EXPECTED_SIGNATURE_SET_COMMITMENT_SHA256
                ),
                expected_time_key_commitment_sha256=time_key_commitment,
            )


if __name__ == "__main__":
    unittest.main()
