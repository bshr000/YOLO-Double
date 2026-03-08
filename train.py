import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from model_dual import DualModalDetector
from loss import DetectionLoss
from dataset import DualYOLODataset, collate_fn_dual
from utils import setup_seed, save_checkpoint, load_checkpoint, ModelEMA

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, config, ema=None, scaler=None):
    model.train()
    
    total_loss = 0
    total_box_loss = 0
    total_dfl_loss = 0
    total_cls_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (images_vis, images_ir, targets) in enumerate(pbar):
        images_vis = images_vis.to(device)
        images_ir = images_ir.to(device)
        
        # Forward
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            predictions = model(images_vis, images_ir)
            loss, loss_dict = criterion(predictions, targets)
        
        # Backward
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)  # Unscale gradients before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # Gradient clipping
            optimizer.step()
        
        # EMA Update
        if ema:
            ema.update(model)
        
        # 统计
        total_loss += loss_dict['total']
        total_box_loss += loss_dict['box']
        total_dfl_loss += loss_dict['dfl']
        total_cls_loss += loss_dict['cls']
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'box': f"{loss_dict['box']:.4f}",
            'dfl': f"{loss_dict['dfl']:.4f}",
            'cls': f"{loss_dict['cls']:.4f}"
        })
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    avg_dfl_loss = total_dfl_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    
    return {
        'loss': avg_loss,
        'box_loss': avg_box_loss,
        'dfl_loss': avg_dfl_loss,
        'cls_loss': avg_cls_loss
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    
    total_loss = 0
    total_box_loss = 0
    total_dfl_loss = 0
    total_cls_loss = 0
    
    with torch.no_grad():
        for images_vis, images_ir, targets in tqdm(dataloader, desc="Validating"):
            images_vis = images_vis.to(device)
            images_ir = images_ir.to(device)
            
            # Forward
            predictions = model(images_vis, images_ir)
            loss, loss_dict = criterion(predictions, targets)
            
            total_loss += loss_dict['total']
            total_box_loss += loss_dict['box']
            total_dfl_loss += loss_dict['dfl']
            total_cls_loss += loss_dict['cls']
    
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_box_loss = total_box_loss / num_batches
    avg_dfl_loss = total_dfl_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    
    return {
        'loss': avg_loss,
        'box_loss': avg_box_loss,
        'dfl_loss': avg_dfl_loss,
        'cls_loss': avg_cls_loss
    }


def main(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    setup_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)

    print("Loading datasets...")
    train_dataset = DualYOLODataset(
        image_dir=config['data']['train_images'],
        label_dir=config['data']['train_labels'],
        input_size=tuple(config['model']['input_size']),
        augment=True,
        augment_params=config['augmentation']
    )
    
    val_dataset = DualYOLODataset(
        image_dir=config['data']['val_images'],
        label_dir=config['data']['val_labels'],
        input_size=tuple(config['model']['input_size']),
        augment=False
    )
    
    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Val dataset: {len(val_dataset)} images")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn_dual,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=False,
        num_workers=config['train']['num_workers'],
        collate_fn=collate_fn_dual,
        pin_memory=True if device.type == 'cuda' else False
    )

    print("Building model (Dual Branch)...")
    # model = build_model(config)
    model = DualModalDetector(num_classes=config['data']['num_classes'])
    model = model.to(device)

    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n{'-'*40}")
        print(f"Model Summary:")
        print(f"  Total Parameters:     {total_params / 1e6:.2f} M")
        print(f"  Trainable Parameters: {trainable_params / 1e6:.2f} M")
        
        try:
            from thop import profile

            h, w = config['model']['input_size']
            dummy_vis = torch.zeros(1, 3, h, w).to(device)
            dummy_ir = torch.zeros(1, 3, h, w).to(device)
            flops, _ = profile(model, inputs=(dummy_vis, dummy_ir), verbose=False)
            
            print(f"  GFLOPS:               {flops / 1e9:.2f} G")
        except ImportError:
            print("  GFLOPS:               N/A (Please install 'thop' via 'pip install thop')")
        except Exception as e:
            print(f"  GFLOPS:               Error ({e})")
            
        print(f"{'-'*40}\n")
        
    except Exception as e:
        print(f"Error summarizing model: {e}")

    criterion = DetectionLoss(
        num_classes=config['data']['num_classes'],
        lambda_box=config['loss'].get('lambda_box', 7.5),
        lambda_cls=config['loss'].get('lambda_cls', 0.5),
        lambda_dfl=config['loss'].get('lambda_dfl', 1.5)
    )

    if config['optimizer']['type'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=config['train']['learning_rate'],
            momentum=config['train']['momentum'],
            weight_decay=config['train']['weight_decay']
        )
    elif config['optimizer']['type'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['train']['learning_rate'],
            weight_decay=config['train']['weight_decay']
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['optimizer']['type']}")

    if config['optimizer']['lr_schedule'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=config['train']['epochs']
        )
    elif config['optimizer']['lr_schedule'] == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(config['train']['epochs'] * 0.6), 
                       int(config['train']['epochs'] * 0.8)],
            gamma=0.1
        )
    else:
        scheduler = None
    
    # EMA
    ema = ModelEMA(model) if config['train'].get('ema', True) else None
    
    # AMP Scaler
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    
    start_epoch = 0
    best_loss = float('inf')
    
    if config['train']['resume']:
        checkpoint = load_checkpoint(config['train']['resume'])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('best_loss', float('inf'))
        if ema and 'ema_state_dict' in checkpoint:
            ema.ema.load_state_dict(checkpoint['ema_state_dict'])
            ema.updates = checkpoint.get('ema_updates', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    print("Start training...")
    for epoch in range(start_epoch, config['train']['epochs']):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, config, ema, scaler
        )
        
        print(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
              f"Box: {train_metrics['box_loss']:.4f}, "
              f"Dfl: {train_metrics['dfl_loss']:.4f}, "
              f"Cls: {train_metrics['cls_loss']:.4f}")
        
        # if len(val_dataset) > 0:
        if (epoch + 1) % 5 == 0:  
            eval_model = ema.ema if ema else model
            val_metrics = validate(eval_model, val_loader, criterion, device)
            print(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}, "
                  f"Box: {val_metrics['box_loss']:.4f}, "
                  f"Dfl: {val_metrics['dfl_loss']:.4f}, "
                  f"Cls: {val_metrics['cls_loss']:.4f}")
            
            current_loss = val_metrics['loss']
        else:
            current_loss = train_metrics['loss']
        
        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % config['train']['save_interval'] == 0:
            checkpoint_path = os.path.join(
                config['train']['checkpoint_dir'],
                f"checkpoint_epoch_{epoch}.pth"
            )
            save_checkpoint(
                checkpoint_path,
                model, optimizer, epoch, current_loss, ema
            )
            print(f"Saved checkpoint: {checkpoint_path}")
        
        if current_loss < best_loss:
            best_loss = current_loss
            best_path = os.path.join(
                config['train']['checkpoint_dir'],
                "best_model.pth"
            )
            save_checkpoint(
                best_path,
                model, optimizer, epoch, best_loss, ema
            )
            print(f"Saved best model with loss: {best_loss:.4f}")

    final_path = os.path.join(
        config['train']['checkpoint_dir'],
        "final_model.pth"
    )
    save_checkpoint(
        final_path,
        model, optimizer, config['train']['epochs'] - 1, current_loss, ema
    )
    print(f"Training finished! Final model saved to {final_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)

