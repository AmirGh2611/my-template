import torch
from torch import nn
import optuna
from training.loop import train_one_epoch, evaluate
from optimization.build import build_model, build_loaders
import config


def objective(trial, train_images, train_labels, val_images, val_labels):
    model = build_model(trial).to(config.DEVICE)
    train_loader, val_loader = build_loaders(
        trial, train_images, train_labels, val_images, val_labels
    )

    lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.MAX_EPOCHS_PER_TRIAL):
        train_one_epoch(model, train_loader, optimizer, criterion, config.DEVICE)
        _, val_acc = evaluate(model, val_loader, criterion, config.DEVICE)

        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    del model, optimizer
    torch.cuda.empty_cache()
    return val_acc


if __name__ == "__main__":
    pass
