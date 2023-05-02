# Copyright (c) OpenMMLab. All rights reserved.
from mmcls.models import ImageClassifier
from mmcv.runner import auto_fp16
import torch

from ..builder import REID


@REID.register_module()
class BaseReID(ImageClassifier):
    """Base class for re-identification."""

    def forward_train(self, img, gt_label, **kwargs):
        """"Training forward function."""
        if img.ndim == 5:
            # change the shape of image tensor from NxSxCxHxW to NSxCxHxW
            # where S is the number of samples by triplet sampling
            img = img.view(-1, *img.shape[2:])
            # change the shape of label tensor from NxS to NS
            gt_label = gt_label.view(-1)
        x = self.extract_feat(img)
        head_outputs = self.head.forward_train(x[0])

        losses = dict()
        reid_loss = self.head.loss(gt_label, *head_outputs)
        losses.update(reid_loss)
        return losses

    @auto_fp16(apply_to=('img', ), out_fp32=True)
    def simple_test(self, img, prob=False, **kwargs):
        """Test without augmentation."""
        # breakpoint()
        if img.nelement() > 0:
            x = self.extract_feat(img)
            if prob:
                head_outputs = self.head.forward_train(x[0])
                feats, feats_log_cov, feats_cov = head_outputs[0:3]
                return feats, feats_log_cov, feats_cov
            else:
                head_outputs = self.head.forward_train(x[0])
                feats = head_outputs[0]
                feats_cov = head_outputs[2]
                trace = feats_cov.sum()
                return feats, trace
        else:
            feat = img.new_zeros(0, self.head.out_channels)
            feat_cov = torch.ones_like(feat) * 0.01
            feats_logcov = torch.log(feat_cov)
            return img.new_zeros(0, self.head.out_channels), \
                feats_logcov, feat_cov
