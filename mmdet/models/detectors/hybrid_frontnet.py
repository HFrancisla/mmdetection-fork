"""E-V1 hybrid FrontNet with an optional YOLA-style consistency objective."""
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

from .hybrid_frontnet_utils import HybridFrontEnd
from .single_stage import SingleStageDetector


@MODELS.register_module()
class HybridFrontNet(SingleStageDetector):
    """A compatible detector front end with an optional self-supervised loss.

    The Gamma consistency loss is computed only in loss(). Prediction and
    tensor forward run the deterministic conventional detector interface.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 hybrid: dict = None):
        super().__init__(
            backbone=backbone, neck=neck, bbox_head=bbox_head,
            train_cfg=train_cfg, test_cfg=test_cfg,
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        kwargs = dict(hybrid or {})
        if isinstance(data_preprocessor, dict):
            kwargs.setdefault('input_mean', tuple(data_preprocessor['mean']))
            kwargs.setdefault('input_std', tuple(data_preprocessor['std']))
        self.front_end = HybridFrontEnd(**kwargs)

    def extract_feat(self, batch_inputs: Tensor):
        output = self.front_end(batch_inputs)
        feature = self.backbone(output)
        if self.with_neck:
            feature = self.neck(feature)
        return feature

    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList):
        output, aux = self.front_end(batch_inputs, return_aux=True)
        feature = self.backbone(output)
        if self.with_neck:
            feature = self.neck(feature)
        losses = self.bbox_head.loss(feature, batch_data_samples)
        if 'loss_consistency' in aux:
            losses['loss_consistency'] = aux['loss_consistency']
        return losses
