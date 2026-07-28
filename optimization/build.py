from torch.utils.data import DataLoader
from model import CUSTOMCNN
from data import MNISTCSVDATASET
import config
import torch


def build_model(trial):
    return CUSTOMCNN(
        conv1_channels=trial.suggest_categorical("conv1_channels", [8, 16, 32]),
        conv2_channels=trial.suggest_categorical("conv2_channels", [16, 32, 64]),
        fc_units=trial.suggest_categorical("fc_units", [64, 128, 256]),
        dropout_rate=trial.suggest_float("dropout_rate", 0.1, 0.5),
    )


def build_loaders(trial, train_images, train_labels, val_images, val_labels):
    g = torch.Generator()
    g.manual_seed(42)

    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    train_ds = MNISTCSVDATASET(
        train_images,
        train_labels,
        augment=config.DATA_AUGMENTATION,
        transform=config.DATA_NORMALIZATION,
    )
    val_ds = MNISTCSVDATASET(
        val_images, val_labels, transform=config.DATA_NORMALIZATION
    )
    return (
        DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            generator=g,
        ),
        DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,
            generator=g,
        ),
    )


if __name__ == "__main__":
    pass
