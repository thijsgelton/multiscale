from typing import Any, List

import torch
from pytorch_lightning import (
    LightningModule,
)
from nnunet_pathology.utils.metrics import Dice
from torchmetrics import MaxMetric
from wholeslidedata.source.files import WholeSlideImageFile


class YnetLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            net: torch.nn.Module,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net

        # loss function
        self.loss_fn_seg = torch.nn.CrossEntropyLoss()
        self.loss_fn_class = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # self.train_dice = Dice(num_classes=6)
        # self.val_dice = Dice(num_classes=6)

        self.class_name_to_onehot = {
            "Normal": torch.Tensor([1, 0, 0, 0, 0, 0]),
            "Hyperplasia_without_atypia": torch.Tensor([0, 1, 0, 0, 0, 0]),
            "Hyperplasia_with_atypia": torch.Tensor([0, 0, 1, 0, 0, 0]),
            "Carcinoma": torch.Tensor([0, 0, 0, 1, 0, 0])
        }
        # self.val_dice_best = MaxMetric()

    def forward(self, x: tuple):
        return self.net(*x)

    # def on_train_start(self):
    #     # by default lightning executes validation step sanity checks before training starts,
    #     # so we need to make sure val_dice_best doesn't store accuracy from these checks
    #     self.val_dice_best.reset()

    def step(self, batch: Any):
        x, y, info = batch
        y_class_true = self.create_class_label(y, info)
        y_seg_true = y[:, 0].type(torch.DoubleTensor).to(device=self.device).permute(0, 3, 1, 2)

        x = tuple([x[:, 0].permute(0, 3, 1, 2), x[:, 1].permute(0, 3, 1, 2)])

        seg_logits, class_logits = self.forward(x)
        loss = self.hybrid_loss(seg_logits, y_seg_true, class_logits, y_class_true)
        return loss, seg_logits, y_seg_true

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        # dice = self.train_dice(preds, targets)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    # def training_epoch_end(self, outputs: List[Any]):
    #     self.train_dice.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        # dice = self.val_dice(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("val/dice", dice, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    # def validation_epoch_end(self, outputs: List[Any]):
        # dice = self.val_dice.compute()  # get val accuracy from current epoch
        # self.val_dice_best.update(dice)
        # self.log("val/dice_best", self.val_dice_best.compute(), on_epoch=True, prog_bar=True)
        # self.val_dice.reset()

    def hybrid_loss(self, y_pred_seg, y_true_seg, y_pred_class, y_true_class):
        loss_seg = self.loss_fn_seg(y_pred_seg, y_true_seg)
        loss_class = self.loss_fn_class(y_pred_class, y_true_class)

        loss = loss_seg + loss_class
        return loss

    def create_class_label(self, y, info):
        associations = self.trainer.train_dataloader.dataset.datasets.iterator.dataset._associations
        y_class_true = torch.zeros(y.shape[0], y.shape[-1], device=self.device)
        for i in range(y.shape[0]):
            class_name = associations[info['sample_references'][0]['reference'].file_key][WholeSlideImageFile][
                0].original_path.split("/")[-2]
            y_class_true[i] = self.class_name_to_onehot[class_name]
        return y_class_true

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
