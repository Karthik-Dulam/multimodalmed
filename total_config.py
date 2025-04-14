def get_default_config():
    config = {
        "seed": 42,
        "mode": "vision",  # "vision", "text", "multimodal"

        "data": {
            "dataset_name": "hongrui/mimic_chest_xray_v_1",
            "dataset_split": "train",
            "train_ratio": 0.8,
            "val_ratio": 0.1,
            "batch_size": 16,
            "num_workers": 10,
            "drop_last": True,
        },

        "model": {
            "hidden_dim": 256,
            "vision": {
                "model_name": "resnet50-res512-all",  # torchxrayvision model weights
                "freeze_backbone": False,
            },
            "text": {
                "model_name": "microsoft/BiomedVLP-CXR-BERT-specialized",
                "freeze_backbone": False,
                "max_length": 512,
                "truncation": False,
            },
            "classifier": {
                "dropout": 0.2,
                "hidden_layers": [256], 
            },
        },

        "training": {
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
            "max_epochs": 6,
            "precision": 16, 
            "devices": "auto",
            "strategy": "ddp_find_unused_parameters_true",
            "early_stopping": {
                "monitor": "val_auroc_avg",
                "min_delta": 0.00,
                "patience": 3,
                "mode": "max",
            },

            "checkpoint": {
                "dirpath": "./text_only_checkpoint",
                "monitor": "val_auroc_avg",
                "save_top_k": 3,
                "mode": "max",
            },

            "logging": {
                "use_tensorboard": True,
                "use_wandb": True,
                "project_name": "chest-xray",
            },
        },
    }

    return config
