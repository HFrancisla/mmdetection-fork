# Copyright (c) OpenMMLab. All rights reserved.
"""Epoch loop that pairs labeled night batches with unlabeled day images."""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from mmengine.dataset import pseudo_collate
from mmengine.registry import LOOPS
from mmengine.runner.loops import EpochBasedTrainLoop

from mmdet.datasets.day_image_dataset import DayImageDataset


@LOOPS.register_module()
class FrontNetGANTrainLoop(EpochBasedTrainLoop):
    """Alternate PatchGAN updates and detector/FrontNet updates each iter."""

    def __init__(self,
                 runner,
                 dataloader,
                 max_epochs,
                 day_root=None,
                 day_batch_size=8,
                 day_num_workers=4,
                 day_max_side_choices=(320, 608),
                 day_horizontal_flip_prob=0.5,
                 val_begin=1,
                 val_interval=1,
                 dynamic_intervals=None):
        super().__init__(
            runner=runner,
            dataloader=dataloader,
            max_epochs=max_epochs,
            val_begin=val_begin,
            val_interval=val_interval,
            dynamic_intervals=dynamic_intervals)

        self.day_dataset = DayImageDataset(
            image_root=day_root,
            max_side_choices=day_max_side_choices,
            horizontal_flip_prob=day_horizontal_flip_prob)

        self.day_sampler = None
        if dist.is_available() and dist.is_initialized():
            self.day_sampler = DistributedSampler(
                self.day_dataset, shuffle=True, drop_last=False)
            self.day_sampler.set_epoch(0)

        self.day_dataloader = DataLoader(
            self.day_dataset,
            batch_size=day_batch_size,
            shuffle=self.day_sampler is None,
            sampler=self.day_sampler,
            num_workers=day_num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=pseudo_collate)
        self._day_epoch = 0
        self.day_iter = iter(self.day_dataloader)

    def _next_day_batch(self):
        try:
            return next(self.day_iter)
        except StopIteration:
            self._day_epoch += 1
            if self.day_sampler is not None:
                self.day_sampler.set_epoch(self._day_epoch)
            self.day_iter = iter(self.day_dataloader)
            return next(self.day_iter)

    def run_iter(self, idx, data_batch):
        day_batch = self._next_day_batch()
        model = self.runner.model
        unwrapped_model = model.module if hasattr(model, 'module') else model
        optim_g = self.runner.optim_wrapper['generator']
        optim_d = self.runner.optim_wrapper['discriminator']

        # Apply the exact detector preprocessor to both domains. The day
        # dataset returns BGR uint8 tensors, like LoadImageFromFile.
        night_data = unwrapped_model.data_preprocessor(
            data_batch, training=True)
        day_data = unwrapped_model.data_preprocessor(
            {'inputs': day_batch, 'data_samples': None}, training=True)
        batch_inputs = night_data['inputs']
        batch_data_samples = night_data['data_samples']
        day_inputs = day_data['inputs']

        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)

        unwrapped_model.discriminator.train()
        for parameter in unwrapped_model.discriminator.parameters():
            parameter.requires_grad_(True)
        with optim_d.optim_context(model):
            losses_d = model(
                batch_inputs,
                mode='gan_discriminator',
                day_inputs=day_inputs)
        parsed_loss_d, log_vars_d = unwrapped_model.parse_losses(losses_d)
        optim_d.update_params(parsed_loss_d)

        # Keep the discriminator's operations in the generator graph so the
        # adversarial gradient reaches FrontNet, while its weights stay fixed.
        for parameter in unwrapped_model.discriminator.parameters():
            parameter.requires_grad_(False)
        unwrapped_model.discriminator.eval()
        try:
            with optim_g.optim_context(model):
                losses_g = model(
                    batch_inputs,
                    batch_data_samples,
                    mode='gan_generator')
            parsed_loss_g, log_vars_g = unwrapped_model.parse_losses(losses_g)
            optim_g.update_params(parsed_loss_g)
        finally:
            for parameter in unwrapped_model.discriminator.parameters():
                parameter.requires_grad_(True)
            unwrapped_model.discriminator.train()

        outputs = dict(log_vars_g)
        outputs['loss_D'] = log_vars_d['loss']
        for name in ('d_real_prob', 'd_fake_prob'):
            outputs[name] = log_vars_d[name]

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1
