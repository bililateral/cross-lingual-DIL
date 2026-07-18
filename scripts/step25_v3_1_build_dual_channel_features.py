#!/usr/bin/env python3
"""Build Step25-v3.1 features through the frozen v3 feature implementation."""

import step25_v3_1_common as common
import step25_v3_build_dual_channel_features as frozen_v3


if __name__ == "__main__":
    frozen_v3.common = common
    frozen_v3.main()
