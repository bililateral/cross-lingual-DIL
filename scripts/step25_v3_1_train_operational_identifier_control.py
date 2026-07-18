#!/usr/bin/env python3
"""Run the frozen operational control against repaired v3.1 clean scores."""

import step25_v3_1_common as common
import step25_v3_train_operational_identifier_control as frozen_v3


if __name__ == "__main__":
    frozen_v3.common = common
    frozen_v3.main()
