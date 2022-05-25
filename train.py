import warnings

from nnunet_pathology.datamodules.wsi_datamodule import WholeSlideDataModule
from pytorch_lightning import (
    Trainer,
)
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping
)
# or to ignore all warnings that could be false positives
from pytorch_lightning.utilities.warnings import PossibleUserWarning

from multiscale.models.YclassRes18Net import YclassRes18Net
from src.model import YnetLitModule

warnings.filterwarnings("ignore", category=PossibleUserWarning)

if __name__ == "__main__":
    datamodule = WholeSlideDataModule(user_train_config="notebooks/train_config.yml",
                                      user_val_config="notebooks/valid_config.yml",
                                      user_test_config="notebooks/valid_config.yml",
                                      num_classes=6,
                                      num_workers=2,
                                      return_info=True)
    trainer = Trainer(
        gpus=1,
        callbacks=[
            ModelCheckpoint(monitor="val/dice", mode="max", save_top_k=1, save_last=True, verbose=True,
                            dirpath="checkpoints/", filename="epoch_{epoch:03d}", auto_insert_metric_name=False),
            EarlyStopping(monitor="val/dice", mode="max", patience=10, min_delta=0)
        ],
        max_epochs=50,
        min_epochs=50,
        num_sanity_val_steps=0
    )

    # and a basic configurate might, such as, for instance
    cfg = {
        'num_classes': 6,
        'num_channels': 3,
        'activation_function': 'ReLU',
        'num_base_featuremaps': 64,
        'encoder_featuremap_delay': 2,
        'decoder_featuremaps_out': [512, 256, 256, 128, -1],
        'conv_norm_type': 'None',
        'depth_levels_down_main': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
        # 'depth_levels_down_side': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
        'depth_levels_down_tail': [[2, 3], [0, 3], [0, 1], [0, 1], [0, 1]],  # [Convs,Res] each
        'depth_levels_up': [1, 1, 1, 1, 1],  # Convs
        'depth_bottleneck': [0, 0, 0],  # [Conv,Res,Conv]
        'internal_prediction_activation': 'None',  # Softmax, Sigmoid or None. None for use with BCEWithLogitsLoss etc.
    }

    # we can then instantiate and apply an examplary multi-scale model using the syntax
    model = YclassRes18Net(cfg=cfg)

    lit_model_module = YnetLitModule(net=model)

    trainer.fit(model=lit_model_module, datamodule=datamodule)

