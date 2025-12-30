"""
Training script for Indian Vehicle Number Plate OCR
Uses Cross-Entropy Loss with Attention-based decoder

Usage:
    python train.py --data_dir ./data --csv_path ./data/train.csv
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR

from config import cfg
from model import Model
from dataset import create_dataloaders
from utils import AttnLabelConverter, Averager


class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing loss to prevent overconfident predictions.
    Helps avoid mode collapse by encouraging diversity.
    """
    def __init__(self, num_classes, smoothing=0.1, ignore_index=0):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confidence = 1.0 - smoothing
    
    def forward(self, pred, target):
        # pred: [N, C], target: [N]
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.num_classes - 2))  # -2 for ignore and true class
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            
            # Mask out ignore_index
            mask = (target != self.ignore_index).unsqueeze(1)
            true_dist = true_dist * mask.float()
        
        loss = (-true_dist * pred).sum(dim=-1)
        return loss[target != self.ignore_index].mean()


def init_weights(model):
    """
    Initialize weights for better convergence.
    Critical for attention models to avoid mode collapse.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            if 'bn' in name.lower() or 'batch' in name.lower():
                # BatchNorm: gamma=1, beta=0
                if 'weight' in name:
                    nn.init.ones_(param)
            elif len(param.shape) >= 2:
                # Linear and Conv layers
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
        elif 'bias' in name:
            nn.init.zeros_(param)
    
    # Special initialization for LSTM
    for name, param in model.named_parameters():
        if 'rnn' in name.lower() or 'lstm' in name.lower():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1
                n = param.size(0)
                param.data[n//4:n//2].fill_(1.0)
    
    print("Weights initialized!")


def train_one_epoch(model, train_loader, criterion, optimizer, scheduler, converter, cfg, epoch):
    """
    Train for one epoch.
    
    Returns:
        Average training loss for the epoch
    """
    model.train()
    loss_avg = Averager()
    
    start_time = time.time()
    
    for batch_idx, (images, text, length) in enumerate(train_loader):
        # Move to device
        images = images.to(cfg.device)
        text = text.to(cfg.device)
        
        # Forward pass with teacher forcing
        # text[:, :-1] is input (starts with [GO], excludes last char)
        # text[:, 1:] is target (excludes [GO], includes [s])
        preds = model(images, text[:, :-1], is_train=True)
        
        # Compute loss
        # preds shape: [batch, max_length+1, num_class]
        # target shape: [batch, max_length+1]
        target = text[:, 1:]  # Exclude [GO] token for target
        
        # Reshape for cross entropy
        # CrossEntropyLoss expects: (N, C) and (N,)
        batch_size = preds.size(0)
        preds_flat = preds.reshape(-1, preds.size(-1))  # [B*(max_len+1), num_class]
        target_flat = target.reshape(-1)  # [B*(max_len+1)]
        
        loss = criterion(preds_flat, target_flat)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        
        optimizer.step()
        
        # Step scheduler per batch for OneCycleLR
        scheduler.step()
        
        loss_avg.add(loss)
        
        # Print progress
        if (batch_idx + 1) % cfg.print_interval == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            current_lr = scheduler.get_last_lr()[0]
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Loss: {loss_avg.val():.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {elapsed:.1f}s")
    
    return loss_avg.val()


def validate(model, val_loader, criterion, converter, cfg, show_samples=False):
    """
    Validate the model.
    
    Returns:
        val_loss: Average validation loss
        accuracy: Sequence-level accuracy (exact match)
        char_accuracy: Character-level accuracy
    """
    model.eval()
    loss_avg = Averager()
    
    n_correct = 0
    n_total = 0
    n_char_correct = 0
    n_char_total = 0
    
    sample_preds = []  # Store sample predictions for debugging
    
    with torch.no_grad():
        for batch_idx, (images, text, length) in enumerate(val_loader):
            images = images.to(cfg.device)
            text = text.to(cfg.device)
            batch_size = images.size(0)
            
            # Forward pass (inference mode) - don't pass text for true inference
            preds = model(images, text[:, :-1], is_train=False)
            
            # Compute loss
            target = text[:, 1:]
            preds_flat = preds.reshape(-1, preds.size(-1))
            target_flat = target.reshape(-1)
            loss = criterion(preds_flat, target_flat)
            loss_avg.add(loss)
            
            # Get predictions (greedy decoding)
            _, preds_index = preds.max(2)  # [B, max_length+1]
            preds_index = preds_index.cpu()
            
            # Decode predictions and targets for this batch
            for i in range(batch_size):
                # Decode prediction - stop at [s] token (index 1)
                pred_chars = []
                for idx in preds_index[i]:
                    idx_val = idx.item()
                    if idx_val == 1:  # [s] end token
                        break
                    if idx_val >= 2:  # Skip [GO]=0, [s]=1, actual chars start at 2
                        pred_chars.append(converter.character[idx_val])
                pred_str = ''.join(pred_chars)
                
                # Decode ground truth
                gt_chars = []
                for idx in text[i, 1:]:  # Skip [GO] at position 0
                    idx_val = idx.item()
                    if idx_val == 1:  # [s] end token
                        break
                    if idx_val >= 2:
                        gt_chars.append(converter.character[idx_val])
                gt_str = ''.join(gt_chars)
                
                # Store samples for debugging
                if batch_idx == 0 and i < 5:
                    sample_preds.append((gt_str, pred_str))
                
                # Calculate accuracy
                if pred_str == gt_str:
                    n_correct += 1
                n_total += 1
                
                # Character accuracy (Levenshtein-style)
                min_len = min(len(pred_str), len(gt_str))
                for j in range(min_len):
                    if pred_str[j] == gt_str[j]:
                        n_char_correct += 1
                n_char_total += max(len(pred_str), len(gt_str))
    
    # Print sample predictions
    if show_samples and sample_preds:
        print("\n  Sample Predictions:")
        for gt, pred in sample_preds:
            match = "✓" if gt == pred else "✗"
            print(f"    GT: {gt:12s} | Pred: {pred:12s} {match}")
    
    accuracy = n_correct / max(n_total, 1) * 100
    char_accuracy = n_char_correct / max(n_char_total, 1) * 100
    
    return loss_avg.val(), accuracy, char_accuracy


def main(args):
    """Main training function."""
    
    # Update config with command line arguments
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.csv_path:
        cfg.csv_path = args.csv_path
    if args.epochs:
        cfg.epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.save_dir:
        cfg.save_dir = args.save_dir
    
    # Create save directory
    os.makedirs(cfg.save_dir, exist_ok=True)
    
    print("=" * 60)
    print("Indian Vehicle Number Plate OCR Training")
    print("=" * 60)
    print(f"Device: {cfg.device}")
    print(f"Data directory: {cfg.data_dir}")
    print(f"CSV path: {cfg.csv_path}")
    print(f"Image size: {cfg.imgH}x{cfg.imgW}")
    print(f"Max text length: {cfg.batch_max_length}")
    print(f"Character set: {cfg.character}")
    print(f"Num classes: {cfg.num_class}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Epochs: {cfg.epochs}")
    print(f"Learning rate: {cfg.learning_rate}")
    print("=" * 60)
    
    # Create label converter
    converter = AttnLabelConverter(cfg.character)
    
    # Create dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = create_dataloaders(cfg, converter)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    print("\nInitializing model...")
    model = Model(cfg)
    model = model.to(cfg.device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize weights properly
    init_weights(model)
    
    # Loss function - Label Smoothing to prevent mode collapse
    criterion = LabelSmoothingLoss(
        num_classes=cfg.num_class,
        smoothing=0.1,
        ignore_index=0
    )
    print(f"Using Label Smoothing Loss (smoothing=0.1)")
    
    # Optimizer with lower learning rate
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # OneCycleLR scheduler - better for convergence
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.learning_rate,
        epochs=cfg.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        anneal_strategy='cos'
    )
    
    # Training loop
    best_accuracy = 0.0
    best_epoch = 0
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch [{epoch}/{cfg.epochs}]")
        print("-" * 40)
        
        # Train (scheduler steps inside train_one_epoch)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, converter, cfg, epoch
        )
        
        # Validate (show samples every 5 epochs)
        show_samples = (epoch % 5 == 0) or (epoch == 1)
        val_loss, accuracy, char_accuracy = validate(
            model, val_loader, criterion, converter, cfg, show_samples=show_samples
        )
        
        # Print epoch summary
        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Sequence Accuracy: {accuracy:.2f}%")
        print(f"  Character Accuracy: {char_accuracy:.2f}%")
        
        # Save checkpoint
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            best_epoch = epoch
            
            # Save best model
            save_path = os.path.join(cfg.save_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
                'char_accuracy': char_accuracy,
                'config': {
                    'character': cfg.character,
                    'imgH': cfg.imgH,
                    'imgW': cfg.imgW,
                    'batch_max_length': cfg.batch_max_length,
                    'num_class': cfg.num_class,
                    'hidden_size': cfg.hidden_size,
                    'output_channel': cfg.output_channel,
                    'num_fiducial': cfg.num_fiducial,
                }
            }, save_path)
            print(f"  ★ New best model saved! (Accuracy: {accuracy:.2f}%)")
        
        # Save periodic checkpoint
        if epoch % cfg.save_interval == 0:
            save_path = os.path.join(cfg.save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': accuracy,
            }, save_path)
            print(f"  Checkpoint saved: {save_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best Accuracy: {best_accuracy:.2f}% at Epoch {best_epoch}")
    print(f"Model saved to: {os.path.join(cfg.save_dir, 'best_model.pth')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train OCR Model')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing images folder')
    parser.add_argument('--csv_path', type=str, default=None,
                        help='Path to CSV file with annotations')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--save_dir', type=str, default=None,
                        help='Directory to save checkpoints')
    
    args = parser.parse_args()
    main(args)
