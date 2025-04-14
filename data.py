import datasets as dts
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

def load_datasets(config):
    ds = dts.load_dataset(
        config["data"]["dataset_name"], 
        split=config["data"]["dataset_split"]
    )
    
    texts = ds.unique("text")
    assert any([x.startswith("chest x-ray;") for x in texts]) is True

    labels = set()
    for x in texts:
        tags = x.split(";")[1:]
        tags = [y.strip().replace("'", "") for y in tags]
        tags = [y for y in tags if y]
        labels.update(tags)
    
    return ds, labels

class ChestXRayDataset(Dataset):
    def __init__(self, dataset, tokenizer, labels, config):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.labels = list(labels)
        self.config = config
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = np.array(sample["image"])
        
        # Ensure we're using a single channel for the x-ray image
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert RGB to grayscale by taking first channel
            image = image[:, :, 0]
        
        image = (2 * (image / 255) - 1) * 1024
        
        # Explicitly set the dimensions to [1, H, W] for a single channel image
        image_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        
        report = sample["report"]
        text = sample["text"]
        
        tags = text.split(";")[1:]
        tags = [y.strip().replace("'", "") for y in tags]
        tags = [y for y in tags if y]
        
        label_tensor = torch.zeros(len(self.labels), dtype=torch.float32)
        for tag in tags:
            if tag in self.labels:
                label_idx = self.labels.index(tag)
                label_tensor[label_idx] = 1.0
        
        return {
            "image": image_tensor,
            "report": report,
            "labels": label_tensor
        }

class ChestXRayDataModule(pl.LightningDataModule):
    def __init__(self, config, dataset, tokenizer, labels):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.labels = labels
        self.batch_size = config["data"]["batch_size"]
        self.train_ratio = config["data"]["train_ratio"]
        self.val_ratio = config["data"]["val_ratio"]
        self.seed = config["seed"]
        self.num_workers = config["data"]["num_workers"]
        self.drop_last = config["data"]["drop_last"]
        
    def setup(self, stage=None):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        
        dataset_size = len(self.dataset)
        train_size = int(self.train_ratio * dataset_size)
        val_size = int(self.val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        
        self.train_dataset = ChestXRayDataset(train_dataset, self.tokenizer, self.labels, self.config)
        self.val_dataset = ChestXRayDataset(val_dataset, self.tokenizer, self.labels, self.config)
        self.test_dataset = ChestXRayDataset(test_dataset, self.tokenizer, self.labels, self.config)
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.batch_size, 
            num_workers=self.num_workers, 
            drop_last=self.drop_last
        )
