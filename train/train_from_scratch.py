import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from typing import Optional
import argparse
import sys
import os
from functools import partial
from utils.util import set_all_seeds
from utils.metrics import Metrics

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def train_one_epoch(model, train_loader, optimizer, criterion, device, accumulation_steps, use_wandb):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    # Initialize tqdm progress bar for the epoch
    progress_bar = tqdm(train_loader, desc="Training", leave=False)

    accumulated_loss = 0.
    
    for i, batch in enumerate(progress_bar):
        inputs, labels = batch['pixel_values'], batch['labels']
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        loss = loss / accumulation_steps
        accumulated_loss += loss.item()
        running_loss += loss.item()
        loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            
            progress_bar.set_postfix(loss=accumulated_loss)
            
            optimizer.step()
            optimizer.zero_grad()
            
            if use_wandb:
                # Log loss and learning rate
                wandb.log({"batch_loss": accumulated_loss, "lr": optimizer.param_groups[0]['lr']})
            
            accumulated_loss = 0.
    
    return running_loss * accumulation_steps / len(train_loader)


# Validation function with tqdm
def validate_model(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    metric = Metrics(num_classes = 129, k = 3)

    # Initialize tqdm progress bar for validation
    progress_bar = tqdm(val_loader, desc="Validating", leave=False)

    with torch.no_grad():
        for batch in progress_bar:
            inputs, labels = batch['pixel_values'], batch['labels']
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            metric.update(outputs, labels)
            
    val_loss = running_loss / len(val_loader)
    perf = metric.compute()
    
    return val_loss, perf


# Full training loop with warmup scheduler and tqdm
def train_model(model: torch.nn.Module, 
                train_loader: torch.utils.data.DataLoader, 
                val_loader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                epochs: int, 
                device: torch.device, 
                accumulation_steps: int,  
                save_ckpt_every: int,
                save_ckpt_dir: str,
                scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
                use_wandb: bool = False):
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    model = model.to(device)

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Train for one epoch
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, accumulation_steps, use_wandb)
        
        print(f"Training loss: {train_loss:.4f}")

        # Validation
        if val_loader is not None:
            val_loss, perf = validate_model(model, val_loader, criterion, device)
            print(f"Validation loss: {val_loss:.4f}")
            print(f"Validation accuracy: {perf['mean_accuracy']:.4f}")
            print(f"Validation F1: {perf['mean_f1']:.4f}")
            print(f"Validation Precision: {perf['mean_precision']:.4f}")
            print(f"Validation Recall: {perf['mean_recall']:.4f}")
            print(f"Validation Top-K Accuracy: {perf['top_k_accuracy']:.4f}")
            
            if use_wandb:
                # Log epoch-level metrics
                wandb.log({
                           "epoch": epoch + 1,
                           "val_loss": val_loss, 
                           "train_loss": train_loss, 
                           "val_accuracy": perf['mean_accuracy'],
                           "val_f1": perf['mean_f1'],
                           "val_precision": perf['mean_precision'],
                           "val_recall": perf['mean_recall'],
                           "val_top_k_accuracy": perf['top_k_accuracy'],
                        }) 
        
        elif val_loader is None and use_wandb:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss}) 
    
        if (epoch + 1) % save_ckpt_every == 0:     
            torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                }, f'{save_ckpt_dir}/model_epoch_{epoch + 1}.pth')

        # schedule learning rate
        if scheduler:
            scheduler.step()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name of the model')
    parser.add_argument('--num_classes', type=int, required=True,
                        help='Number of classes')
    parser.add_argument('--model_path', type=str,
                        help='Path to the model file')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for data loading')
    parser.add_argument('--train_data_path', type=str,
                        help='Train data directory')
    parser.add_argument('--validation_data_path', type=str,
                        help='Validation data directory')
    parser.add_argument('--test_data_path', type=str,
                        help='Number of accumulation steps')
    parser.add_argument('--warmup_steps', type=float, default=0.0,
                        help='Percentage of total training steps for warmup')
    parser.add_argument('--save_ckpt_every', type=int, default=1,
                        help='Number of epochs between checkpoints')
    parser.add_argument('--save_ckpt_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Number of accumulation steps')
    parser.add_argument('--scheduler', type=str, default=None,
                        help='Type of learning rate scheduler, limited to StepLR, CosineAnnealingLR, ReduceLROnPlateau')
    parser.add_argument('--class_balance', type=str2bool, nargs='?', const=True, default=False,
                        help='Handle class imbalance')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use for training')
    parser.add_argument('--use_wandb', type=str2bool, nargs='?', const=True, default=False,
                        help='Use wandb')
    args = parser.parse_args()

    if args.model_name == 'VideoMAE-v2':    
        
        from models.VideoMAE_v2 import VideoMAEForClassification
        from transformers import VideoMAEImageProcessor
        
        model = VideoMAEForClassification(num_classes=args.num_classes,
                                          qkv_bias=True,
                                          norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                          init_values=0.,
                                          )
        
        if args.model_path:
            checkpoint = torch.load(args.model_path)
            
            del checkpoint['head.weight'], checkpoint['head.bias']
            model.load_state_dict(checkpoint, strict=False)

        # initialize custom transformations for VideoMAE-v2 model
        from utils.util import create_video_transforms
        
        image_processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
        image_mean, image_std, resize_to, crop_size = image_processor.image_mean, image_processor.image_std, image_processor.crop_size['height'], image_processor.crop_size.values()

        transformations = create_video_transforms(mean = image_mean, std = image_std, resize_to = resize_to, crop_size = crop_size)
        
    elif args.model_name == 'i3d':
        
        from models.i3d import InceptionI3d
        model = InceptionI3d(num_classes=args.num_classes, 
                            in_channels=3
                            )
        
        if args.model_path:
            checkpoint = torch.load(args.model_path)
            
            del checkpoint['logits.conv3d.weight'], checkpoint['logits.conv3d.bias']
            model.load_state_dict(checkpoint, strict=False)
        
        # initialize custom transformations for i3d model
        from utils.util import create_video_transforms_i3d
        transformations = create_video_transforms_i3d(resize_to = 224, crop_size = 224)
    
    elif args.model_name == 'x3d':
        
        from models.x3d import X3D
        model = X3D(gamma_w=1,
                    gamma_b=2.25,
                    gamma_d=2.2,
                    num_classes=args.num_classes
                    )
        if args.model_path:
            checkpoint = torch.load(args.model_path)
            
            model.load_state_dict(checkpoint, strict=False)
        
        # initialize custom transformations for i3d model
        from utils.util import create_video_transforms
        transformations = create_video_transforms(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resize_to = 224, crop_size = 224)
        # transformations = create_video_transforms(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225], resize_to = 224, crop_size = 224)  
    
    if args.use_wandb:
        import wandb
        from dotenv import load_dotenv
        
        # Load environment variables from the .env file
        load_dotenv()

        # Access the API key
        wandb_api_key = os.getenv("WANDB_API_KEY")

        wandb.login(key=wandb_api_key)

        wandb.init(project='classification_model', config={"monitor_gpus": True})
    
    if not os.path.exists(args.save_ckpt_dir):
        os.makedirs(args.save_ckpt_dir)

    # set seed
    set_all_seeds()
    
    # create dataset & dataloader
    from utils.prepare_dataset import CustomVideoDataset, FlexibleVideoDataset
    from utils.util import collate_fn
    
    train_dataset = FlexibleVideoDataset(root_dir = args.train_data_path, transform = transformations)
    val_dataset = FlexibleVideoDataset(root_dir = args.train_data_path, transform = transformations)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler = train_dataset.get_sampler())
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    # create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # create learning rate scheduler
    if args.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.75)
    elif args.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.001)
    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    else:
        scheduler = None
        
    # train!
    train_model(model=model,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                epochs=args.num_epochs,
                device=args.device,
                accumulation_steps=args.accumulation_steps,
                save_ckpt_every=args.save_ckpt_every,
                save_ckpt_dir=args.save_ckpt_dir,
                use_wandb=args.use_wandb
                )