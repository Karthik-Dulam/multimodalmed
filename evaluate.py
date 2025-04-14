import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, roc_curve, auc
import wandb
from tqdm import tqdm

from data import load_datasets, ChestXRayDataModule
from models import create_model
from total_config import get_default_config

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained chest X-ray model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file")
    parser.add_argument("--batch_size", type=int, help="Batch size for evaluation (overrides config)")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="Directory to save evaluation results")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--wandb_project", type=str, default="chest-xray-evaluation", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity name")
    parser.add_argument("--use_wandb", action="store_true", help="Enable W&B logging")
    return parser.parse_args()

def load_model_from_checkpoint(checkpoint_path):
    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    config = checkpoint.get("hyper_parameters", {}).get("config", get_default_config())
    labels = checkpoint.get("hyper_parameters", {}).get("labels", [])
    
    lit_model = create_model(config, labels).to("cuda:0")
    
    # Load the model state dict
    lit_model.load_state_dict(checkpoint["state_dict"])
    
    return lit_model, config, labels

def predict_dataset(trainer, model, dataloader):
    """Run prediction on a dataloader and collect all outputs."""
    all_predictions = []
    all_probs = []
    all_labels = []
    
    for batch in tqdm(dataloader):
        # Move batch to the same device as the model
        device = next(model.parameters()).device
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        # Get predictions
        with torch.no_grad():
            images = batch["image"] if "image" in batch and model.model.mode != "text" else None
            reports = batch["report"] if "report" in batch and model.model.mode != "vision" else None
            labels = batch["labels"]
            
            logits = model(images, reports)
            probs = torch.sigmoid(logits)
            predictions = (probs > 0.5).float()
            
            all_predictions.append(predictions.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_predictions, all_probs, all_labels

def calculate_metrics(predictions, probs, labels, label_names):
    """Calculate per-label accuracy and AUROC scores."""
    results = []
    
    for i, label in enumerate(label_names):
        # Extract binary predictions and labels for this class
        label_preds = predictions[:, i].numpy()
        label_true = labels[:, i].numpy()
        label_probs = probs[:, i].numpy()
        
        # Calculate accuracy
        accuracy = accuracy_score(label_true, label_preds)
        
        # Calculate AUROC if possible (need both positive and negative samples)
        auroc = np.nan
        if np.sum(label_true) > 0 and np.sum(label_true) < len(label_true):
            try:
                auroc = roc_auc_score(label_true, label_probs)
            except:
                pass
                
        # Store results
        results.append({
            "label": label,
            "accuracy": accuracy,
            "auroc": auroc,
            "num_positive": np.sum(label_true),
            "num_total": len(label_true),
            "positive_rate": np.sum(label_true) / len(label_true),
        })
        
    # Calculate average metrics
    avg_accuracy = np.mean([r["accuracy"] for r in results])
    valid_aurocs = [r["auroc"] for r in results if not np.isnan(r["auroc"])]
    avg_auroc = np.mean(valid_aurocs) if valid_aurocs else np.nan
    
    # Add averages to results
    results.append({
        "label": "AVERAGE",
        "accuracy": avg_accuracy,
        "auroc": avg_auroc,
        "num_positive": np.nan,
        "num_total": np.nan,
        "positive_rate": np.nan,
    })
    
    return results

def plot_confusion_matrices(predictions, labels, label_names, output_dir, use_wandb=True):
    """Generate and save confusion matrices for each label."""
    print("Generating confusion matrices...")
    os.makedirs(os.path.join(output_dir, "confusion_matrices"), exist_ok=True)
    
    # Create a combined figure for all labels
    num_labels = len(label_names)
    n_cols = min(4, num_labels)
    n_rows = (num_labels + n_cols - 1) // n_cols
    
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    
    all_cms = {}
    
    for i, label in enumerate(label_names):
        # Get predictions and true labels for this class
        label_preds = predictions[:, i].numpy()
        label_true = labels[:, i].numpy()
        
        # Calculate confusion matrix
        cm = confusion_matrix(label_true, label_preds)
        all_cms[label] = cm
        
        # Plot confusion matrix
        plt.subplot(n_rows, n_cols, i + 1)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                   xticklabels=["Negative", "Positive"], 
                   yticklabels=["Negative", "Positive"])
        plt.title(f"Confusion Matrix: {label}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        
        # Also save individual confusion matrix
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                   xticklabels=["Negative", "Positive"], 
                   yticklabels=["Negative", "Positive"])
        plt.title(f"Confusion Matrix: {label}")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        cm_file = os.path.join(output_dir, "confusion_matrices", f"{label}_confusion.png")
        plt.savefig(cm_file, dpi=300)
        
        # Log individual confusion matrix to wandb
        if use_wandb:
            wandb.log({f"confusion_matrix/{label}": wandb.Image(cm_file)})
        
        plt.close()
    
    # Save the combined figure
    plt.tight_layout()
    all_cm_file = os.path.join(output_dir, "all_confusion_matrices.png")
    plt.savefig(all_cm_file, dpi=300)
    
    # Log combined confusion matrix to wandb
    if use_wandb:
        wandb.log({"confusion_matrix/all": wandb.Image(all_cm_file)})
        
        # Create a wandb table for all confusion matrices
        cm_table = wandb.Table(columns=["Label", "TN", "FP", "FN", "TP", "Confusion Matrix"])
        for label, cm in all_cms.items():
            tn, fp, fn, tp = cm.ravel()
            cm_file = os.path.join(output_dir, "confusion_matrices", f"{label}_confusion.png")
            cm_table.add_data(label, int(tn), int(fp), int(fn), int(tp), wandb.Image(cm_file))
        wandb.log({"confusion_matrices": cm_table})
    
    plt.close()
    
    print(f"Confusion matrices saved to {os.path.join(output_dir, 'confusion_matrices')}")
    
    return all_cms

def plot_roc_curves(probs, labels, label_names, output_dir, use_wandb=True):
    """Generate and save ROC curves for each label."""
    print("Generating ROC curves...")
    os.makedirs(os.path.join(output_dir, "roc_curves"), exist_ok=True)
    
    # Create a combined figure for all ROC curves
    plt.figure(figsize=(10, 8))
    
    valid_aurocs = []
    roc_data = []
    
    for i, label in enumerate(label_names):
        label_probs = probs[:, i].numpy()
        label_true = labels[:, i].numpy()
        
        # Skip if no positive or negative samples
        if np.sum(label_true) == 0 or np.sum(label_true) == len(label_true):
            continue
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(label_true, label_probs)
        roc_auc = auc(fpr, tpr)
        valid_aurocs.append((label, roc_auc))
        roc_data.append((label, fpr, tpr, roc_auc))
        
        # Plot on combined figure
        plt.plot(fpr, tpr, lw=2, label=f'{label} (AUC = {roc_auc:.2f})')
        
        # Also create and save individual ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, color='darkorange')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve: {label} (AUC = {roc_auc:.3f})')
        plt.grid(True)
        plt.tight_layout()
        roc_file = os.path.join(output_dir, "roc_curves", f"{label}_roc.png")
        plt.savefig(roc_file, dpi=300)
        
        # Log individual ROC curve to wandb
        if use_wandb:
            wandb.log({f"roc_curve/{label}": wandb.Image(roc_file)})
            
        plt.close()
    
    # Add reference diagonal to combined plot
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Labels')
    plt.legend(loc="lower right", fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    all_roc_file = os.path.join(output_dir, "all_roc_curves.png")
    plt.savefig(all_roc_file, dpi=300)
    
    # Log combined ROC curve to wandb
    if use_wandb:
        wandb.log({"roc_curve/all": wandb.Image(all_roc_file)})
        
        # Create a wandb table for all ROC data
        roc_table = wandb.Table(columns=["Label", "AUC", "ROC Curve"])
        for label, _, _, roc_auc in roc_data:
            roc_file = os.path.join(output_dir, "roc_curves", f"{label}_roc.png")
            roc_table.add_data(label, roc_auc, wandb.Image(roc_file))
        wandb.log({"roc_curves": roc_table})
    
    plt.close()
    
    # Sort and save AUC values to text file
    valid_aurocs.sort(key=lambda x: x[1], reverse=True)
    with open(os.path.join(output_dir, "auroc_ranking.txt"), "w") as f:
        f.write("Label AUC Rankings:\n")
        f.write("-" * 40 + "\n")
        for label, score in valid_aurocs:
            f.write(f"{label}: {score:.4f}\n")
        
        avg_auc = np.mean([score for _, score in valid_aurocs])
        f.write("-" * 40 + "\n")
        f.write(f"Average AUC: {avg_auc:.4f}\n")
    
    print(f"ROC curves saved to {os.path.join(output_dir, 'roc_curves')}")
    return valid_aurocs

def extract_checkpoint_info(checkpoint_path):
    """Extract information from checkpoint filename for naming output files."""
    checkpoint_name = os.path.basename(checkpoint_path).replace('.ckpt', '')
    
    # Try to extract epoch number and metrics from checkpoint name
    info = {}
    
    # Extract epoch number
    epoch_match = None
    if "epoch=" in checkpoint_name:
        # For format like: chest-xray-vision-epoch=00-val_auroc_avg=0.6755-val_accuracy_avg=0.8470
        epoch_match = checkpoint_name.split("epoch=")[1].split("-")[0]
    elif "epoch" in checkpoint_name:
        # For format like: chest-xray-vision-epoch00-val_auroc_avg=0.6755-val_accuracy_avg=0.8470
        epoch_parts = checkpoint_name.split("epoch")[1].split("-")[0]
        if epoch_parts.isdigit():
            epoch_match = epoch_parts
    
    if epoch_match:
        info["epoch"] = epoch_match
    
    # Extract metrics
    metrics = ["val_auroc_avg", "val_accuracy_avg"]
    for metric in metrics:
        if f"{metric}=" in checkpoint_name:
            metric_value = checkpoint_name.split(f"{metric}=")[1].split("-")[0]
            try:
                info[metric] = float(metric_value)
            except:
                pass
    
    return info

def main():
    args = parse_args()
    
    # Initialize wandb if enabled
    if args.use_wandb:
        print("Initializing Weights & Biases...")
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            job_type="evaluation",
        )
    
    # Load model from checkpoint
    model, config, labels = load_model_from_checkpoint(args.checkpoint)
    
    # Extract information from checkpoint filename
    checkpoint_info = extract_checkpoint_info(args.checkpoint)
    
    # Create a formatted identifier based on checkpoint info
    checkpoint_id = os.path.basename(args.checkpoint).replace('.ckpt', '')
    
    # If we have epoch information, make it part of the identifier
    if "epoch" in checkpoint_info:
        epoch_str = checkpoint_info["epoch"]
        checkpoint_id = f"{config['mode']}_epoch{epoch_str}_{args.split}"
    else:
        checkpoint_id = f"{config['mode']}_{args.split}"
    
    # Override config with command-line arguments if provided
    if args.batch_size:
        config["data"]["batch_size"] = args.batch_size
    
    # Update wandb config
    if args.use_wandb:
        # Set run name
        wandb.run.name = checkpoint_id
        
        # Log configuration
        wandb_config = {
            "model_type": config['mode'],
            "checkpoint": args.checkpoint,
            "split": args.split,
            "batch_size": config["data"]["batch_size"],
            "num_labels": len(labels),
        }
        
        # Add checkpoint info to config
        for key, value in checkpoint_info.items():
            wandb_config[f"checkpoint_{key}"] = value
            
        wandb.config.update(wandb_config)
    
    # Load datasets
    ds, _ = load_datasets(config)
    
    # Setup data module
    data_module = ChestXRayDataModule(
        config=config,
        dataset=ds, 
        tokenizer=model.model.tokenizer,
        labels=labels
    )
    
    # Setup logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = TensorBoardLogger(args.output_dir, name=f"evaluation-{checkpoint_id}")
    
    # Setup trainer
    trainer = pl.Trainer(
        devices=[0],
        logger=logger,
        precision=config["training"]["precision"],
    )
    
    # Prepare data
    data_module.setup()
    
    # Get the appropriate dataloader
    if args.split == "train":
        dataloader = data_module.train_dataloader()
        split_name = "Training"
    elif args.split == "val":
        dataloader = data_module.val_dataloader()
        split_name = "Validation"
    else:  # test
        dataloader = data_module.test_dataloader()
        split_name = "Test"
    
    print(f"Starting evaluation on {split_name} set...")
    
    # Predict and collect all outputs
    predictions, probs, true_labels = predict_dataset(trainer, model, dataloader)
    
    # Calculate metrics
    results = calculate_metrics(predictions, probs, true_labels, labels)
    
    # Create output directory for this checkpoint
    output_dir = os.path.join(args.output_dir, checkpoint_id)
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for better display and saving
    df_results = pd.DataFrame(results)
    
    # Print formatted results
    print(f"\n{split_name} Results:")
    print("-" * 80)
    print(df_results.to_string(index=False, float_format="%.4f"))
    
    # Save to CSV
    csv_path = os.path.join(output_dir, f"{checkpoint_id}_results.csv")
    df_results.to_csv(csv_path, index=False)
    
    # Log results to wandb
    if args.use_wandb:
        # Log average metrics
        avg_row = df_results.iloc[-1]
        wandb.log({
            "avg_accuracy": avg_row["accuracy"],
            "avg_auroc": avg_row["auroc"],
        })
        
        # Create a wandb table for per-label results
        results_table = wandb.Table(dataframe=df_results)
        wandb.log({"per_label_metrics": results_table})
        
        # Log the CSV file
        wandb.save(csv_path)
    
    # Also save to text file for readability
    txt_path = os.path.join(output_dir, f"{checkpoint_id}_results.txt")
    with open(txt_path, "w") as f:
        f.write(f"Model: {config['mode']}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        if "epoch" in checkpoint_info:
            f.write(f"Epoch: {checkpoint_info['epoch']}\n")
        if "val_auroc_avg" in checkpoint_info:
            f.write(f"Validation AUROC: {checkpoint_info['val_auroc_avg']:.4f}\n")
        if "val_accuracy_avg" in checkpoint_info:
            f.write(f"Validation Accuracy: {checkpoint_info['val_accuracy_avg']:.4f}\n")
        f.write(f"Split: {split_name}\n")
        f.write("-" * 80 + "\n\n")
        f.write(df_results.to_string(index=False, float_format="%.4f"))
    
    print(f"\nResults saved to:")
    print(f"  CSV: {csv_path}")
    print(f"  TXT: {txt_path}")
    
    # Generate and save confusion matrices
    confusion_dir = os.path.join(output_dir, "confusion_matrices")
    os.makedirs(confusion_dir, exist_ok=True)
    all_cms = plot_confusion_matrices(predictions, true_labels, labels, output_dir, args.use_wandb)
    
    # Generate and save ROC curves
    roc_dir = os.path.join(output_dir, "roc_curves")
    os.makedirs(roc_dir, exist_ok=True)
    valid_aurocs = plot_roc_curves(probs, true_labels, labels, output_dir, args.use_wandb)
    
    # Also run validation through the Lightning Trainer for more metrics
    print("\nRunning PyTorch Lightning validation...")
    trainer_results = trainer.validate(model, datamodule=data_module)
    
    # Save trainer results to the same directory
    trainer_results_path = os.path.join(output_dir, f"{checkpoint_id}_trainer_results.txt")
    with open(trainer_results_path, "w") as f:
        f.write(f"Model: {config['mode']}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        if "epoch" in checkpoint_info:
            f.write(f"Epoch: {checkpoint_info['epoch']}\n")
        f.write(f"Split: {split_name}\n")
        f.write("-" * 80 + "\n\n")
        for metric_name, value in trainer_results[0].items():
            f.write(f"{metric_name}: {value:.4f}\n")
    
    # Log trainer results to wandb
    if args.use_wandb:
        for metric_name, value in trainer_results[0].items():
            wandb.log({f"trainer/{metric_name}": value})
    
    print(f"Trainer results saved to: {trainer_results_path}")
    
    # Create a summary report
    summary_path = os.path.join(output_dir, f"{checkpoint_id}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Evaluation Summary\n")
        f.write(f"=================\n\n")
        f.write(f"Model: {config['mode']}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        if "epoch" in checkpoint_info:
            f.write(f"Epoch: {checkpoint_info['epoch']}\n")
        if "val_auroc_avg" in checkpoint_info:
            f.write(f"Checkpoint Validation AUROC: {checkpoint_info['val_auroc_avg']:.4f}\n")
        if "val_accuracy_avg" in checkpoint_info:
            f.write(f"Checkpoint Validation Accuracy: {checkpoint_info['val_accuracy_avg']:.4f}\n")
        f.write(f"Split: {split_name}\n")
        f.write(f"Batch Size: {config['data']['batch_size']}\n")
        f.write(f"Number of Labels: {len(labels)}\n\n")
        
        f.write(f"Overall Metrics:\n")
        f.write(f"---------------\n")
        f.write(f"Average Accuracy: {results[-1]['accuracy']:.4f}\n")
        f.write(f"Average AUROC: {results[-1]['auroc']:.4f}\n\n")
        
        # Add label distribution statistics
        common_threshold = 0.05  # 5%
        rare_threshold = 0.01    # 1%
        
        common_labels = [r for r in results[:-1] if r["positive_rate"] > common_threshold]
        rare_labels = [r for r in results[:-1] if r["positive_rate"] <= rare_threshold]
        
        f.write(f"Label Distribution:\n")
        f.write(f"----------------\n")
        f.write(f"Common labels (>{common_threshold*100}% positive rate): {len(common_labels)}\n")
        f.write(f"Rare labels (<={rare_threshold*100}% positive rate): {len(rare_labels)}\n\n")
        
        if valid_aurocs:
            f.write(f"Top 5 performing labels (by AUROC):\n")
            for i, (label, score) in enumerate(valid_aurocs[:5]):
                f.write(f"{i+1}. {label}: {score:.4f}\n")
            
            f.write(f"\nBottom 5 performing labels (by AUROC):\n")
            for i, (label, score) in enumerate(valid_aurocs[-5:]):
                f.write(f"{i+1}. {label}: {score:.4f}\n")
    
    # Log summary statistics to wandb
    if args.use_wandb:
        wandb.log({
            "label_stats/common_label_count": len(common_labels),
            "label_stats/rare_label_count": len(rare_labels),
            "label_stats/total_label_count": len(labels),
        })
        
        # Log top and bottom performing labels
        if valid_aurocs:
            for i, (label, score) in enumerate(valid_aurocs[:5]):
                wandb.log({f"top_labels/{label}": score})
                
            for i, (label, score) in enumerate(valid_aurocs[-5:]):
                wandb.log({f"bottom_labels/{label}": score})
                
        # Save the summary file to wandb
        wandb.save(summary_path)
    
    print(f"Evaluation summary saved to: {summary_path}")
    
    # Finish wandb run
    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
