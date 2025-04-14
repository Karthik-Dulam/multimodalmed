import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
import torchxrayvision as xrv
from transformers import AutoModel, AutoTokenizer
import os

class MultiModalModel(nn.Module):
    def __init__(self, config, vision_model, text_model, tokenizer, num_labels=None):
        super().__init__()
        self.config = config
        self.vision_model = vision_model
        self.text_model = text_model
        self.tokenizer = tokenizer
        self.hidden_dim = config["model"]["hidden_dim"]
        self.num_labels = num_labels
        self.mode = config["mode"]
        
        self.image_feature_dim = None
        self.text_feature_dim = None
        
        if vision_model is not None:
            dummy_img = torch.zeros(1, 1, 512, 512)
            self.image_feature_dim = self.vision_model.features(dummy_img).shape[1]
            self.image_projection = nn.Linear(self.image_feature_dim, self.hidden_dim)
        
        if text_model is not None:
            self.text_feature_dim = self.text_model.config.hidden_size
            self.text_projection = nn.Linear(self.text_feature_dim, self.hidden_dim)
            
        if num_labels is not None:
            hidden_layers = config["model"]["classifier"]["hidden_layers"]
            dropout_rate = config["model"]["classifier"]["dropout"]
            
            input_size = self.hidden_dim
            if self.mode == "multimodal":
                input_size = self.hidden_dim * 2
            
            layers = []
            for hidden_size in hidden_layers:
                layers.extend([
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate)
                ])
                input_size = hidden_size
            
            layers.append(nn.Linear(input_size, num_labels))
            self.classifier = nn.Sequential(*layers)

    def forward(self, images=None, reports=None):
        image_features = None
        text_features = None
        
        if (self.mode == "vision" or self.mode == "multimodal") and images is not None:
            image_features = self.vision_model.features(images)
            image_features = self.image_projection(image_features)
        elif self.mode == "vision" and images is None:
            raise ValueError("Images required for vision mode")
            
        if (self.mode == "text" or self.mode == "multimodal") and reports is not None:
            text_config = self.config["model"]["text"]
            text_inputs = self.tokenizer(
                reports, 
                return_tensors="pt", 
                padding="longest", 
                truncation=text_config["truncation"],
                max_length=text_config["max_length"]
            )
            device = next(self.parameters()).device
            text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
            text_outputs = self.text_model(**text_inputs).last_hidden_state
            text_features = text_outputs[:, 0, :]
            text_features = self.text_projection(text_features)
        elif self.mode == "text" and reports is None:
            raise ValueError("Reports required for text mode")
        
        if self.mode == "vision":
            features = image_features
        elif self.mode == "text":
            features = text_features
        else:
            features = torch.cat([image_features, text_features], dim=1)
        
        if self.num_labels is not None:
            return self.classifier(features)
        
        return features
    
    def save(self, path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "tokenizer": self.tokenizer,
            "hidden_dim": self.hidden_dim,
            "num_labels": self.num_labels,
            "mode": self.mode,
            "config": self.config,
        }
        torch.save(checkpoint, path)
    
class LitModel(pl.LightningModule):
    def __init__(self, config, model, labels):
        super().__init__()
        # Save hyperparameters: This makes them available to loggers like WandB
        # We pass the relevant parts of the config dictionary.
        hparams_to_save = {
            'mode': config['mode'],
            'learning_rate': config['training']['learning_rate'],
            'weight_decay': config['training']['weight_decay'],
            'batch_size': config['data']['batch_size'],
            'hidden_dim': config['model']['hidden_dim'],
            'vision_model': config['model']['vision']['model_name'] if config['mode'] != 'text' else 'N/A',
            'text_model': config['model']['text']['model_name'] if config['mode'] != 'vision' else 'N/A',
            'vision_frozen': config['model']['vision']['freeze_backbone'] if config['mode'] != 'text' else 'N/A',
            'text_frozen': config['model']['text']['freeze_backbone'] if config['mode'] != 'vision' else 'N/A',
            'classifier_dropout': config['model']['classifier']['dropout'],
            'classifier_hidden': '_'.join(map(str, config['model']['classifier']['hidden_layers']))
        }
        self.save_hyperparameters(hparams_to_save)

        self.config = config # Keep the full config if needed internally
        self.model = model
        self.labels = list(labels)
        self.num_labels = len(self.labels)
        # Access learning rate and weight decay from hparams after saving
        self.learning_rate = self.hparams.learning_rate
        self.weight_decay = self.hparams.weight_decay
        self.validation_step_outputs = []

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, images=None, reports=None):
        if self.model.mode == "vision":
            return self.model(images=images)
        elif self.model.mode == "text":
            return self.model(reports=reports)
        else:
            return self.model(images=images, reports=reports)

    def training_step(self, batch, batch_idx):
        self.model.train()
        images = batch["image"] if "image" in batch and self.model.mode != "text" else None
        reports = batch["report"] if "report" in batch and self.model.mode != "vision" else None
        labels = batch["labels"]
        
        logits = self(images, reports)
        loss = self.criterion(logits, labels)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        self.model.eval()
        images = batch["image"] if "image" in batch and self.model.mode != "text" else None
        reports = batch["report"] if "report" in batch and self.model.mode != "vision" else None
        labels = batch["labels"]
        
        logits = self(images, reports)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        
        predictions = (probs > 0.5).float()
        
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
        self.validation_step_outputs.append({"probs": probs, "labels": labels, "predictions": predictions})
        
        return {"loss": loss, "probs": probs, "labels": labels, "predictions": predictions}
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        all_probs = torch.cat([x["probs"] for x in outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in outputs], dim=0)
        all_predictions = torch.cat([x["predictions"] for x in outputs], dim=0)
        
        accuracies = []
        for i, label in enumerate(self.labels):
            correct = (all_predictions[:, i] == all_labels[:, i]).float().sum()
            total = all_labels.size(0)
            accuracy = correct / total
            accuracies.append(accuracy.item())
            self.log(f"val_accuracy_{label}", accuracy, on_epoch=True)
        
        if accuracies:
            mean_accuracy = sum(accuracies) / len(accuracies)
            self.log("val_accuracy_avg", mean_accuracy, on_epoch=True, prog_bar=True)
        
        auroc_scores = []
        for i, label in enumerate(self.labels):
            if torch.sum(all_labels[:, i]) > 0 and torch.sum(all_labels[:, i]) < len(all_labels):
                try:
                    auroc = roc_auc_score(all_labels[:, i].cpu().numpy(), all_probs[:, i].cpu().numpy())
                    auroc_scores.append(auroc)
                    self.log(f"val_auroc_{label}", auroc, on_epoch=True)
                except:
                    pass
        
        if auroc_scores:
            mean_auroc = sum(auroc_scores) / len(auroc_scores)
            self.log("val_auroc_avg", mean_auroc, on_epoch=True, prog_bar=True)
            
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        self.model.eval()
        images = batch["image"] if "image" in batch and self.model.mode != "text" else None
        reports = batch["report"] if "report" in batch and self.model.mode != "vision" else None
        labels = batch["labels"]
        
        logits = self(images, reports)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        
        predictions = (probs > 0.5).float()
        correct = (predictions == labels).float()
        accuracy = correct.sum() / (labels.size(0) * labels.size(1))
        
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_accuracy", accuracy, on_epoch=True)
        
        self.test_step_outputs = getattr(self, "test_step_outputs", [])
        self.test_step_outputs.append({"probs": probs, "labels": labels, "predictions": predictions})
        
        return {"loss": loss, "probs": probs, "labels": labels, "predictions": predictions}
    
    def on_test_epoch_end(self):
        outputs = getattr(self, "test_step_outputs", [])
        if not outputs:
            return
            
        all_probs = torch.cat([x["probs"] for x in outputs], dim=0)
        all_labels = torch.cat([x["labels"] for x in outputs], dim=0)
        all_predictions = torch.cat([x["predictions"] for x in outputs], dim=0)
        
        accuracies = []
        for i, label in enumerate(self.labels):
            correct = (all_predictions[:, i] == all_labels[:, i]).float().sum()
            total = all_labels.size(0)
            accuracy = correct / total
            accuracies.append(accuracy.item())
            self.log(f"test_accuracy_{label}", accuracy)
        
        if accuracies:
            mean_accuracy = sum(accuracies) / len(accuracies)
            self.log("test_accuracy_avg", mean_accuracy)
        
        auroc_scores = []
        for i, label in enumerate(self.labels):
            if torch.sum(all_labels[:, i]) > 0 and torch.sum(all_labels[:, i]) < len(all_labels):
                try:
                    auroc = roc_auc_score(all_labels[:, i].cpu().numpy(), all_probs[:, i].cpu().numpy())
                    auroc_scores.append(auroc)
                    self.log(f"test_auroc_{label}", auroc)
                except:
                    pass
        
        if auroc_scores:
            mean_auroc = sum(auroc_scores) / len(auroc_scores)
            self.log("test_auroc_avg", mean_auroc)
            
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
    
    def predict(self, images=None, reports=None):
        self.eval()
        with torch.no_grad():
            logits = self(images, reports)
            probs = torch.sigmoid(logits)
        return probs
        
    def save(self, path):
        self.model.save(path)

def create_model(config, labels):
    mode = config["mode"]
    
    vision_model = None
    if mode == "vision" or mode == "multimodal":
        vision_config = config["model"]["vision"]
        vision_model = xrv.models.ResNet(weights=vision_config["model_name"])
        
        if vision_config["freeze_backbone"]:
            for param in vision_model.parameters():
                param.requires_grad = False
    
    text_model = None
    tokenizer = None
    if mode == "text" or mode == "multimodal":
        text_config = config["model"]["text"]
        text_model = AutoModel.from_pretrained(
            text_config["model_name"], 
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            text_config["model_name"], 
            trust_remote_code=True
        )
        
        if text_config["freeze_backbone"]:
            for param in text_model.parameters():
                param.requires_grad = False
    
    multimodal_model = MultiModalModel(
        config=config,
        vision_model=vision_model,
        text_model=text_model,
        tokenizer=tokenizer,
        num_labels=len(labels)
    )
    
    lit_model = LitModel(
        config=config,
        model=multimodal_model,
        labels=labels
    )
    return lit_model
