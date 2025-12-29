import json
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
from model import OCRModel
from train import IIIT5KDataset, LabelConverter, collate_fn  # Reuse components

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def train_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, converter, fold_idx, grad_clip):
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(dataloader):
        # Debug: Save first image of batch to check input
        # if i % 10 == 0: # Save occasionally
        #     save_dir = '/Users/nikhil/Desktop/Leegality_task/train_check'
        #     if not os.path.exists(save_dir):
        #         os.makedirs(save_dir)
            
        #     try:
        #         # Denormalize
        #         img = images[0].cpu().clone()
        #         mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        #         std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        #         img = img * std + mean
        #         img = torch.clamp(img, 0, 1)
                
        #         # Convert to numpy [H, W, C]
        #         img_np = (img.permute(1, 2, 0).numpy() * 255).astype('uint8')
                
        #         # Get label
        #         label_str = converter.decode(targets[0])
        #         # Sanitize label
        #         safe_label = "".join([c if c.isalnum() else "_" for c in label_str])
        #         Image.fromarray(img_np).save(os.path.join(save_dir, f"batch_{i}_{safe_label}.png"))
        #     except Exception as e:
        #         print(f"Error saving debug image: {e}")
        images = images.to(device)
        targets = targets.to(device)
        
        tgt_input = targets[:, :-1]
        tgt_output = targets[:, 1:]
        
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        tgt_padding_mask = (tgt_input == converter.PAD).to(device)
        
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type):
            output = model(images, tgt_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
            
            output = output.reshape(-1, converter.vocab_size)
            tgt_output_flat = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output_flat)
        loss.backward()
        
        # Gradient Clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
        optimizer.step()
        
        running_loss += loss.item()
        
        # Log to TensorBoard
        step = epoch * len(dataloader) + i
        if i % 10 == 0:
            writer.add_scalar(f'Fold_{fold_idx}/Training/Loss_Step', loss.item(), step)
            
    return running_loss / len(dataloader)

def validate(model, dataloader, criterion, device, converter):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)
            
            tgt_input = targets[:, :-1]
            tgt_output = targets[:, 1:]
            
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            tgt_padding_mask = (tgt_input == converter.PAD).to(device)
            
            output = model(images, tgt_input, tgt_mask=tgt_mask, tgt_padding_mask=tgt_padding_mask)
            
            output = output.reshape(-1, converter.vocab_size)
            tgt_output_flat = tgt_output.reshape(-1)
            
            loss = criterion(output, tgt_output_flat)
            running_loss += loss.item()
            
    return running_loss / len(dataloader)

def main():
    config = load_config('config.json')
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup Paths
    if not os.path.exists(os.path.dirname(config['training']['save_path'])):
        os.makedirs(os.path.dirname(config['training']['save_path']))
        
    writer = SummaryWriter(log_dir=config['training']['log_dir']+config['training']['exp_name'])
    
    # Data Setup
    train_df = pd.read_csv(config['data']['train_csv'])
    all_text = "".join(train_df['GroundTruth'].astype(str).tolist())
    converter = LabelConverter(all_text)
    print(f"Vocabulary size: {converter.vocab_size}")
    
    transform = transforms.Compose([
        transforms.RandomApply([
          transforms.ColorJitter(),
          transforms.RandomAffine(10),
          transforms.RandomPerspective(),
          transforms.RandomRotation(5),
          transforms.GaussianBlur(3),]),
        transforms.Resize((config['data']['image_height'],config['data']['image_width'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    full_dataset = IIIT5KDataset(
        csv_file=config['data']['train_csv'],
        root_dir=config['data']['train_dir'],
        size=(config['data']['image_height'],config['data']['image_width']),
        transform=transform,
        converter=converter
    )
    
    k_folds = config['training'].get('k_folds', 5)
    splits = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(splits.split(np.arange(len(full_dataset)))):
        print(f"---------------- Fold {fold+1}/{k_folds} ----------------")
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        
        train_loader = DataLoader(
            full_dataset, 
            batch_size=config['training']['batch_size'], 
            sampler=train_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            prefetch_factor=4
        )
        
        val_loader = DataLoader(
            full_dataset, 
            batch_size=config['training']['batch_size'], 
            sampler=val_sampler,
            collate_fn=collate_fn,
            num_workers=4,
            prefetch_factor=4
        )
        
        # Model Setup (Re-initialize for each fold)
        model = OCRModel(
            vocab_size=converter.vocab_size,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            resnet_layers=config['model']['resnet_layers'],
            max_len=config['data']['max_len'],
            dropout=config['model'].get('dropout', 0.1)
        ).to(device)
        
        criterion = nn.CrossEntropyLoss(
            ignore_index=converter.PAD, 
            label_smoothing=config['training'].get('label_smoothing', 0.0)
        )
        
        best_val_loss = float('inf')
        
        # --- Stage 1: Freeze ---
        print("Stage 1: Freezing ResNet backbone")
        for param in model.feature_extractor.parameters():
            param.requires_grad = False
            
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=config['training']['transformer_learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0)
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=config['training']['scheduler_patience'], 
            factor=config['training']['scheduler_factor']
        )
        
        freeze_epochs = config['training']['freeze_epochs']
        grad_clip = config['training'].get('grad_clip', 0.0)
        warmup_epochs = 5

        for epoch in range(freeze_epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, converter, fold, grad_clip)
            val_loss = validate(model, val_loader, criterion, device, converter)
            
            print(f"Fold {fold+1} Stage 1 - Epoch [{epoch+1}/{freeze_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
            writer.add_scalar(f'Fold_{fold}/Training/Loss', train_loss, epoch)
            writer.add_scalar(f'Fold_{fold}/Validation/Loss', val_loss, epoch)
            writer.add_scalar(f'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            if epoch < warmup_epochs:
                warmup_lr = config['training']['transformer_learning_rate'] * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                print(f"Warmup Epoch {epoch+1}: LR set to {warmup_lr:.6f}")
            else:
                scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = config['training']['save_path'].replace('.pth', f'_{config["training"]["exp_name"]}_fold{fold}_best.pth')
                torch.save(model.state_dict(), save_path)

        # --- Stage 2: Unfreeze ---
        print("Stage 2: Unfreezing ResNet backbone")
        # To unfreeze the last ~20 layers (Layer 3 and Layer 4)
        for name, param in model.feature_extractor.backbone.named_parameters():
            param.requires_grad = True
            
        optimizer = torch.optim.Adam([
            {'params': model.feature_extractor.parameters(), 'lr': config['training']['resnet_learning_rate']}, # Very small
            {'params': model.transformer.parameters(), 'lr': config['training']['transformer_learning_rate']} # Normal
        ], weight_decay=config['training'].get('weight_decay', 0))
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=config['training']['scheduler_patience'], 
            factor=config['training']['scheduler_factor']
        )
        
        unfreeze_epochs = config['training']['unfreeze_epochs']
        total_epochs = freeze_epochs + unfreeze_epochs
        
        for epoch in range(freeze_epochs, total_epochs):
            if epoch >= freeze_epochs+20:
                full_dataset.set_aug(True)
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, converter, fold, grad_clip)
            val_loss = validate(model, val_loader, criterion, device, converter)    
            
            print(f"Fold {fold+1} Stage 2 - Epoch [{epoch+1}/{total_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
            writer.add_scalar(f'Fold_{fold}/Training/Loss', train_loss, epoch)
            writer.add_scalar(f'Fold_{fold}/Validation/Loss', val_loss, epoch)
            writer.add_scalar(f'Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            scheduler.step(val_loss)
             
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = config['training']['save_path'].replace('.pth', f'_{config["training"]["exp_name"]}_fold{fold}_best.pth')
                torch.save(model.state_dict(), save_path)
                print(f"New Best Model Saved! Val Loss: {val_loss:.4f}")

        # Save Last Model for the fold
        last_save_path = config['training']['save_path'].replace('.pth', f'_{config["training"]["exp_name"]}_fold{fold}_last.pth')
        torch.save(model.state_dict(), last_save_path)
        print(f"Fold {fold+1} Completed. Last Model saved to {last_save_path}")
        break

    writer.close()
    print("Cross Validation Training Complete.")

if __name__ == "__main__":
    main()
