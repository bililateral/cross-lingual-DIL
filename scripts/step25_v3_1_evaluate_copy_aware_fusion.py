#!/usr/bin/env python3
"""Evaluate frozen Step25-v3 C0-C3 with the repaired v3.1 solver."""

import step25_v3_1_common as common
import step25_v3_evaluate_copy_aware_fusion as frozen_v3


if __name__ == "__main__":
    frozen_v3.common = common
    frozen_v3.main()
