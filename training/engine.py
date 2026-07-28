from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint, EarlyStopping, global_step_from_engine


def build_engine(
    model,
    optimizer,
    criterion,
    device,
    val_loader,
    checkpoint_dir="outputs/checkpoints",
    patience=3,
):
    trainer = create_supervised_trainer(model, optimizer, criterion, device)
    val_metrics = {"accuracy": Accuracy(), "loss": Loss(criterion)}
    evaluator = create_supervised_evaluator(model, val_metrics, device)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        m = evaluator.state.metrics
        print(
            f"Epoch {engine.state.epoch} - Val Acc: {m['accuracy']:.4f} | Loss: {m['loss']:.4f}"
        )

    def score_function(engine):
        return engine.state.metrics["accuracy"]

    checkpoint = ModelCheckpoint(
        checkpoint_dir,
        n_saved=1,
        filename_prefix="best",
        score_function=score_function,
        score_name="accuracy",
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(Events.COMPLETED, checkpoint, {"model": model})

    early_stopping = EarlyStopping(
        patience=patience, score_function=score_function, trainer=trainer
    )
    evaluator.add_event_handler(Events.COMPLETED, early_stopping)

    return trainer, evaluator


if __name__ == "__main__":
    pass
