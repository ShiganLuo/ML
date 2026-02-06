import math
import numpy as np
import pandas as pd
import os
import csv
from pathlib import Path
from tqdm import tqdm
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import logging
from typing import Union, Tuple, List, Dict, Optional

# Logging Configuration
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def same_seed(seed: int): 
    """
    Fixes random seeds for reproducibility across numpy, torch, and cudnn.
    
    Args:
        seed (int): The seed value to use.
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.debug(f'Random seed set to: {seed}')

def train_valid_split(data_set: np.ndarray, valid_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splits a dataset into training and validation sets based on a ratio.
    
    Args:
        data_set (np.ndarray): The full dataset.
        valid_ratio (float): Proportion of the dataset to include in the validation split.
        seed (int): Seed for the random generator.
    """
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    # Use torch.utils.data.random_split for indices or directly shuffle via numpy
    indices = np.arange(len(data_set))
    np.random.seed(seed)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_set_size]
    valid_indices = indices[train_set_size:]
    
    return data_set[train_indices], data_set[valid_indices]

class COVID19Dataset(Dataset):
    """
    Custom Dataset for loading COVID-19 features and labels.
    
    Args:
        x (np.ndarray): Feature matrix.
        y (np.ndarray, optional): Target labels. Defaults to None (for testing).
    """
    def __init__(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        self.x = torch.FloatTensor(x)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.x)

class My_Model(nn.Module):
    """
    Simple Deep Neural Network for Regression.
    
    Args:
        input_dim (int): Number of input features.
    """
    def __init__(self, input_dim: int):
        super(My_Model, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.layers(x)
        return x.squeeze(1) 

def select_feat(train_data: np.ndarray, valid_data: np.ndarray, test_data: np.ndarray, select_all: bool = True) -> Tuple:
    """
    Selects specific features from the dataset for training.
    
    Args:
        train_data (np.ndarray): Training raw data including labels.
        valid_data (np.ndarray): Validation raw data including labels.
        test_data (np.ndarray): Testing raw data.
        select_all (bool): Whether to use all available features.
    """
    y_train, y_valid = train_data[:, -1], valid_data[:, -1]
    raw_x_train, raw_x_valid, raw_x_test = train_data[:, :-1], valid_data[:, :-1], test_data
    
    if select_all:
        feat_idx = list(range(raw_x_train.shape[1]))
    else:
        # Example: select first 5 features
        feat_idx = [0, 1, 2, 3, 4] 
        
    return raw_x_train[:, feat_idx], raw_x_valid[:, feat_idx], raw_x_test[:, feat_idx], y_train, y_valid

def trainer(train_loader: DataLoader, valid_loader: DataLoader, model: nn.Module, config: Dict, device: str):
    """
    Handles the training and validation loops for the model.
    """
    criterion = nn.MSELoss(reduction='mean') 
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9) 
    writer = SummaryWriter() # Log to runs/ folder
    
    save_path = Path(config['save_path'])
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_epochs, best_loss, step, early_stop_count = config['n_epochs'], math.inf, 0, 0
    
    for epoch in range(n_epochs):
        model.train() 
        loss_record = []
        train_pbar = tqdm(train_loader, position=0, leave=True, desc=f'Epoch [{epoch+1}/{n_epochs}]')
        
        for x, y in train_pbar:
            optimizer.zero_grad() 
            x, y = x.to(device), y.to(device) 
            pred = model(x) 
            loss = criterion(pred, y)
            loss.backward() 
            optimizer.step() 
            
            step += 1
            l_val = loss.detach().item()
            loss_record.append(l_val)
            train_pbar.set_postfix({'loss': f'{l_val:.4f}'})
        
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, epoch)

        # Validation
        model.eval() 
        val_loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)
            val_loss_record.append(loss.item())
            
        mean_valid_loss = sum(val_loss_record) / len(val_loss_record)
        writer.add_scalar('Loss/valid', mean_valid_loss, epoch)
        
        logger.debug(f'Epoch [{epoch+1}/{n_epochs}]: Train: {mean_train_loss:.4f}, Valid: {mean_valid_loss:.4f}')

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['save_path']) 
            logger.info(f'Best model saved (Loss: {best_loss:.4f})')
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            logger.warning('\nEarly stopping triggered.')
            break
    writer.close()

def predict(test_loader: DataLoader, model: nn.Module, device: str) -> np.ndarray:
    """
    Generates predictions using the trained model.
    """
    model.eval() 
    preds = []
    for x in tqdm(test_loader, desc='Predicting'):
        x = x.to(device)                        
        with torch.no_grad():
            pred = model(x)         
            preds.append(pred.detach().cpu())   
    return torch.cat(preds, dim=0).numpy()

def save_pred(preds: np.ndarray, file_path: str):
    """
    Saves predictions to a CSV file.
    """
    logger.info(f'Saving predictions to {file_path}')
    with open(file_path, 'w', newline='') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tested_positive'])
        for i, p in enumerate(preds):
            writer.writerow([i, p])

def main(train_file: str, test_file: str, out_predict: Union[str, Path], log: Optional[Union[str, Path]] = None):
    """
    Main orchestration function.
    """
    # 1. Directory Setup
    out_predict = Path(out_predict)
    model_dir = out_predict.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_ckpt_path = str(model_dir / "model.ckpt")

    if log:
        log_path = Path(log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = {
        'seed': 5201314,
        'select_all': True,
        'valid_ratio': 0.2,
        'n_epochs': 3000,
        'batch_size': 256,
        'learning_rate': 1e-6,
        'early_stop': 400,
        'save_path': model_ckpt_path  # 使用动态生成的路径
    }
    
    same_seed(config['seed'])

    # 2. Data Loading & Preprocessing
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    
    train_data, valid_data = train_valid_split(train_df.values, config['valid_ratio'], config['seed'])
    x_train, x_valid, x_test, y_train, y_valid = select_feat(train_data, valid_data, test_df.values, config['select_all'])
    
    # 3. Dataset & Dataloaders
    train_dataset = COVID19Dataset(x_train, y_train)
    valid_dataset = COVID19Dataset(x_valid, y_valid)
    test_dataset = COVID19Dataset(x_test)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    # 4. Training
    model = My_Model(input_dim=x_train.shape[1]).to(device)
    trainer(train_loader, valid_loader, model, config, device)

    # 5. Prediction (Using the same save_path)
    model.load_state_dict(torch.load(config['save_path'], weights_only=True))
    preds = predict(test_loader, model, device) 
    save_pred(preds, str(out_predict))

if __name__ == "__main__":
    # Example paths - update these as needed
    work_config = {
        "train_file": "/data/pub/zhousha/ML/leedl-tutorial-master/Homework/HW1_Regression/covid.train.csv",
        "test_file": "/data/pub/zhousha/ML/leedl-tutorial-master/Homework/HW1_Regression/covid.train.csv",
        "out_predict": "/data/pub/zhousha/ML/output/DNN/COVID_predict.csv",
        "log": "/data/pub/zhousha/ML/output/log/COVID.log"
    }
    main(**work_config)