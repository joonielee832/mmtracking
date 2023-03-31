# Copyright (c) OpenMMLab. All rights reserved.
from .l2_loss import L2Loss
from .multipos_cross_entropy_loss import MultiPosCrossEntropyLoss
from .triplet_loss import TripletLoss
from .triplet_js_loss import TripletJSLoss
from .triplet_kl_loss import TripletKLDivLoss
from .uncertainty_loss import FeatureUncertaintyLoss
from .triplet_jr_loss import TripletJRLoss

__all__ = ['L2Loss', 'TripletLoss', 'MultiPosCrossEntropyLoss', 'TripletJSLoss', 'FeatureUncertaintyLoss', 'TripletKLDivLoss',
           'TripletJRLoss']
