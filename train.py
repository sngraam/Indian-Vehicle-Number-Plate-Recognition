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
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import cfg
from model import Model
from dataset import create_dataloaders
from utils import AttnLabelConverter, Averager


def train_one_epoch(model, train_loader, criterion, optimizer, converter, cfg, epoch):
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
        
        loss_avg.add(loss)
        
        # Print progress
        if (batch_idx + 1) % cfg.print_interval == 0 or batch_idx == 0:
            elapsed = time.time() - start_time
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] | "
                  f"Loss: {loss.item():.4f} | "
                  f"Avg Loss: {loss_avg.val():.4f} | "
                  f"Time: {elapsed:.1f}s")
    
    return loss_avg.val()


def validate(model, val_loader, criterion, converter, cfg):
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
    
    with torch.no_grad():
        for images, text, length in val_loader:
            images = images.to(cfg.device)
            text = text.to(cfg.device)
            batch_size = images.size(0)
            
            # Forward pass (inference mode)
            preds = model(images, text[:, :-1], is_train=False)
            
            # Compute loss
            target = text[:, 1:]
            preds_flat = preds.reshape(-1, preds.size(-1))
            target_flat = target.reshape(-1)
            loss = criterion(preds_flat, target_flat)
            loss_avg.add(loss)
            
            # Get predictions (greedy decoding)
            _, preds_index = preds.max(2)  # [B, max_length+1]
            
            # Decode predictions and targets
            preds_str = converter.decode(preds_index, length)
            
            # Decode targets
            target_text = []
            for i in range(batch_size):
                t = text[i, 1:]  # Exclude [GO]
                decoded = ""
                for idx in t:
                    if idx.item() == 1:  # [s] token (end)
                        break
                    if idx.item() >= 2:  # Skip [GO]=0, [s]=1
                        decoded += converter.character[idx.item()]
                target_text.append(decoded)
            
            # Calculate accuracy
            for pred, gt in zip(preds_str, target_text):
                # Clean prediction (stop at [s])
                if '[s]' in pred:
                    pred = pred[:pred.index('[s]')]
                pred = pred.replace('[GO]', '')
                
                if pred == gt:
                    n_correct += 1
                n_total += 1
                
                # Character accuracy
                for p_char, g_char in zip(pred, gt):
                    if p_char == g_char:
                        n_char_correct += 1
                    n_char_total += 1
                n_char_total += abs(len(pred) - len(gt))  # Penalize length mismatch
    
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
    
    # Loss function - Cross Entropy
    # Index 0 is [GO] token which we use for padding, so ignore it
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
    
    # Training loop
    best_accuracy = 0.0
    best_epoch = 0
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch [{epoch}/{cfg.epochs}] | LR: {scheduler.get_last_lr()[0]:.6f}")
        print("-" * 40)
        
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, converter, cfg, epoch
        )
        
        # Validate
        val_loss, accuracy, char_accuracy = validate(
            model, val_loader, criterion, converter, cfg
        )
        
        # Update scheduler
        scheduler.step()
        
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
