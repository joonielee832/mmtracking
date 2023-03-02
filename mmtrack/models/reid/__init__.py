# Copyright (c) OpenMMLab. All rights reserved.
from .base_reid import BaseReID
from .fc_module import FcModule
from .gap import GlobalAveragePooling
from .linear_reid_head import LinearReIDHead
from .probabilistic_reid_head import ProbabilisticReIDHead
from .extra_fc_linear_reid_head import ExtraLinearReIDHead

__all__ = ['BaseReID', 'GlobalAveragePooling', 'LinearReIDHead', 'FcModule', 'ProbabilisticReIDHead', 'ExtraLinearReIDHead']
