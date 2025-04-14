import copy
import optuna # Import optuna
from total_config import get_default_config
from train import train_model

# Define the objective function for Optuna
def objective(trial, mode, base_config):
    """Objective function for Optuna study."""
    # Create a deep copy for this trial to avoid conflicts
    config = copy.deepcopy(base_config)
    config["mode"] = mode

    # Adjust checkpoint and logging paths based on mode and trial
    # train_model now handles adding trial number to checkpoint path
    config["training"]["checkpoint"]["dirpath"] = f"./checkpoints_hpo" # Use a separate base dir for HPO checkpoints
    # WandB project name can be set per mode or kept general
    config["training"]["logging"]["project_name"] = f"chest-xray-hpo-{mode}"

    # Call train_model, passing the trial object
    # train_model will suggest hyperparameters based on the trial
    metric_value = train_model(config, optuna_trial=trial)

    # Return the metric value for Optuna to optimize
    # Handle cases where training might fail or return None/NaN
    return metric_value if metric_value is not None else float("-inf")


def run_training_for_all_modes():
    """
    Runs hyperparameter optimization using Optuna for vision, text, and multimodal modes.
    """
    modes = ["vision", "text", "multimodal"]
    base_config = get_default_config()
    n_trials = 10 # Number of HPO trials per mode - adjust as needed

    for mode in modes:
        print(f"==================================================")
        print(f" Starting Hyperparameter Optimization for Mode: {mode} ")
        print(f"==================================================")

        # Create an Optuna study for the current mode
        # Direction should match the metric being optimized (e.g., maximize for AUROC)
        study_name = f"chest-xray-{mode}-hpo"
        storage_name = f"sqlite:///{study_name}.db" # Store results in a local SQLite DB
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name, # Use storage to resume if needed
            load_if_exists=True, # Load previous results if the study exists
            direction="maximize", # Or "minimize" depending on the metric in train.py
            pruner=optuna.pruners.MedianPruner() # Example pruner
        )

        # Run the optimization
        # Use a lambda to pass the mode and base_config to the objective function
        study.optimize(lambda trial: objective(trial, mode, base_config), n_trials=n_trials)

        # Print best trial results
        print(f"\n--- Optuna Study Summary for Mode: {mode} ---")
        print(f"Number of finished trials: {len(study.trials)}")
        best_trial = study.best_trial
        print(f"Best trial value ({base_config['training']['early_stopping']['monitor']}): {best_trial.value}")
        print("Best parameters:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

        print(f"==================================================")
        print(f" Finished Hyperparameter Optimization for Mode: {mode} ")
        print(f"==================================================\n")

if __name__ == "__main__":
    # Make sure wandb is installed and logged in (`pip install wandb`, `wandb login`)
    # Make sure optuna is installed (`pip install optuna`)
    run_training_for_all_modes()