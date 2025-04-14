import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import Callback

from data import load_datasets, ChestXRayDataModule
from models import create_model
from total_config import get_default_config
import os
import optuna


class PyTorchLightningPruningCallback(Callback):
    def __init__(self, trial, monitor):
        super().__init__()
        self._trial = trial
        self.monitor = monitor
        self.is_pruned = False

    def on_validation_end(self, trainer, pl_module):
        if self._trial.should_prune():
            if not self.is_pruned:
                self.is_pruned = True
                message = "Trial was pruned at epoch {} based on {}={}.".format(
                    trainer.current_epoch, self.monitor, trainer.callback_metrics.get(self.monitor)
                )
                raise optuna.TrialPruned(message)

        epoch = trainer.current_epoch
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is not None:
            self._trial.report(current_score, step=epoch)


def train_model(config, optuna_trial=None):
    pl.seed_everything(config["seed"])

    if optuna_trial:
        config["training"]["learning_rate"] = optuna_trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        config["training"]["weight_decay"] = optuna_trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        # config["model"]["hidden_dim"] = optuna_trial.suggest_categorical("hidden_dim", [128, 256, 512], log=True)
        # config["model"]["classifier"]["dropout"] = optuna_trial.suggest_float("dropout", 0.1, 0.5, log=True)

    ds, labels = load_datasets(config)

    data_module = ChestXRayDataModule(
        config=config, dataset=ds, tokenizer=None, labels=labels
    )

    model = create_model(config, labels)
    if hasattr(model.model, 'tokenizer') and model.model.tokenizer is not None:
        data_module.tokenizer = model.model.tokenizer
    else:
        pass

    data_module.setup()

    callbacks = []
    monitor_metric = config["training"]["early_stopping"]["monitor"]

    early_stop_config = config["training"]["early_stopping"]
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        min_delta=early_stop_config["min_delta"],
        patience=early_stop_config["patience"],
        verbose=True,
        mode=early_stop_config["mode"],
    )
    callbacks.append(early_stop_callback)

    if optuna_trial:
        pruning_callback = PyTorchLightningPruningCallback(optuna_trial, monitor_metric)
        callbacks.append(pruning_callback)

    checkpoint_config = config["training"]["checkpoint"]
    checkpoint_dir = f"{checkpoint_config['dirpath']}/{config['mode']}"
    if optuna_trial:
        checkpoint_dir = f"{checkpoint_dir}/trial_{optuna_trial.number}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"chest-xray-{config['mode']}-" + "{epoch:02d}-{val_auroc_avg:.4f}",
        save_top_k=checkpoint_config["save_top_k"],
        verbose=False,
        monitor=monitor_metric,
        mode=checkpoint_config["mode"],
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(StochasticWeightAveraging(swa_lrs=1e-2))

    loggers = []
    logging_config = config["training"]["logging"]
    log_name_base = f"chest-xray-{config['mode']}"
    log_name = f"{log_name_base}_trial_{optuna_trial.number}" if optuna_trial else log_name_base

    if logging_config["use_tensorboard"]:
        tb_logger = TensorBoardLogger(
            "lightning_logs", name=log_name_base, version=f"trial_{optuna_trial.number}" if optuna_trial else None
        )
        loggers.append(tb_logger)

    if logging_config["use_wandb"]:
        wandb_logger = WandbLogger(
            name=log_name,
            project=logging_config["project_name"],
            log_model=False,
            config=config
        )
        if optuna_trial:
            wandb_logger.log_hyperparams(optuna_trial.params)

        loggers.append(wandb_logger)

    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        precision=config["training"]["precision"],
        devices=config["training"]["devices"],
        callbacks=callbacks,
        logger=loggers,
        strategy=config["training"]["strategy"],
        val_check_interval=config["training"]["val_check_interval"],
        enable_progress_bar=not optuna_trial,
        enable_model_summary=not optuna_trial,
    )

    print(f"--- Starting Training {'(Optuna Trial ' + str(optuna_trial.number) + ')' if optuna_trial else ''} for Mode: {config['mode']} ---")
    print(f"Hyperparameters: {config['training']['learning_rate']=}, {config['training']['weight_decay']=}, {config['model']['hidden_dim']=}, {config['model']['classifier']['dropout']=}")

    try:
        trainer.fit(model, data_module)
        print(f"--- Finished Training {'(Optuna Trial ' + str(optuna_trial.number) + ')' if optuna_trial else ''} for Mode: {config['mode']} ---")
        optimized_metric = trainer.callback_metrics.get(monitor_metric)
        return optimized_metric.item() if optimized_metric is not None else float('-inf')

    except optuna.TrialPruned as e:
        print(f"--- Pruned Optuna Trial {optuna_trial.number} for Mode: {config['mode']} ---")
        if logging_config["use_wandb"] and wandb_logger:
            wandb_logger.experiment.finish(exit_code=1, quiet=True)
        raise e
    except Exception as e:
        print(f"--- Error during Training {'(Optuna Trial ' + str(optuna_trial.number) + ')' if optuna_trial else ''} for Mode: {config['mode']} ---")
        print(e)
        if logging_config["use_wandb"] and wandb_logger:
            wandb_logger.experiment.finish(exit_code=1, quiet=True)
        return float('-inf')
    finally:
        # Ensure wandb run is finished properly, even if errors occurred
        if logging_config["use_wandb"] and wandb_logger and wandb_logger.experiment and wandb_logger.experiment.id:
            # Check if the run is still active before finishing
            # The check for backend might be outdated or cause issues
            try:
                # Simply finish the run if it exists
                wandb_logger.experiment.finish()
            except Exception as finish_error:
                # Log potential errors during finish, but don't crash the overall process
                print(f"Error finishing wandb run: {finish_error}")


def main():
    config = get_default_config()
    train_model(config)


if __name__ == "__main__":
    main()
