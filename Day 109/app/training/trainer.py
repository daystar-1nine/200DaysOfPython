import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional
import numpy as np
from .callbacks import EarlyStopping

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).float()
        
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        running_loss += loss.item() * batch_x.size(0)
        binary_preds = (preds >= 0.5).long()
        correct += (binary_preds == batch_y.long()).sum().item()
        total += batch_y.size(0)
        
    return running_loss / max(total, 1), correct / max(total, 1)

def evaluate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    collect_attention: bool = False
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    all_attn_weights = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).float()
            
            if collect_attention and hasattr(model, "forward") and "return_attention" in model.forward.__code__.co_varnames:
                preds, attn = model(batch_x, return_attention=True)
                all_attn_weights.extend(attn.cpu().numpy().tolist())
            else:
                preds = model(batch_x)
                
            loss = criterion(preds, batch_y)
            running_loss += loss.item() * batch_x.size(0)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(batch_y.cpu().numpy().tolist())
            
    total = len(all_targets)
    epoch_loss = running_loss / max(total, 1)
    preds_arr = np.array(all_preds)
    targets_arr = np.array(all_targets)
    epoch_acc = np.mean((preds_arr >= 0.5).astype(int) == targets_arr) if total > 0 else 0.0
    
    attn_arr = np.array(all_attn_weights) if len(all_attn_weights) > 0 else None
    return epoch_loss, float(epoch_acc), preds_arr, targets_arr, attn_arr

def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 20,
    lr: float = 0.001,
    patience: int = 3,
    device: str = "cpu",
    verbose: bool = False
) -> Dict:
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, mode="min")
    
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "epochs_trained": 0,
        "training_time_sec": 0.0
    }
    
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["epochs_trained"] = epoch
        
        if verbose:
            print(f"Epoch {epoch:02d}/{epochs:02d} - Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
            
        if early_stopping(val_loss, model):
            early_stopping.restore(model)
            break
            
    history["training_time_sec"] = time.time() - start_time
    return history
