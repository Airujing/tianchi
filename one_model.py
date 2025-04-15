# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset
from torchvision import transforms, models
from PIL import Image
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
import argparse
from typing import List, Tuple, Optional, Dict, Any, Union

# NEW: Import timm
import timm

# --- Configuration via argparse ---
def parse_args():
    parser = argparse.ArgumentParser(description='Single Model Fire Hazard Detection Training and Inference with timm models')

    # Paths
    parser.add_argument('--train_label_file', type=str, default='Data/智慧骑士_label/train.txt', help='Path to training label file')
    parser.add_argument('--train_img_dir', type=str, default='Data/智慧骑士_train/train', help='Path to training image directory')
    parser.add_argument('--test_label_file', type=str, default='Data/智慧骑士_label/A.txt', help='Path to test label file (for ordering)')
    parser.add_argument('--test_img_dir', type=str, default='Data/智慧骑士_A/A', help='Path to test image directory')
    parser.add_argument('--output_file', type=str, default='submit_single_model.txt', help='Path to output submission file')
    parser.add_argument('--model_save_path', type=str, default='best_single_model.pth', help='Path to save best model weights')

    # Model & Training Hyperparameters
    # UPDATED: Added timm model choices
    parser.add_argument('--model_name', type=str, default='convnext_tiny',
                        choices=['resnet50', 'efficientnet-b3',
                                 'convnext_tiny', 'convnext_small', 'convnext_base', # Added ConvNeXt options
                                 'vit_tiny_patch16_224', 'vit_small_patch16_224', 'vit_base_patch16_224' # Added ViT options
                                 ],
                        help='Choose the model architecture (torchvision or timm)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size (height and width). Ensure compatibility with model (e.g., ViT often uses 224 or 384)')
    parser.add_argument('--batch_size', type=int, default=32, help='Training and validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Test batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--lr_head', type=float, default=1e-4, help='Learning rate for the classifier head')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for the pre-trained backbone (differential LR)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 penalty)')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of linear warmup epochs')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'crossentropy_smooth'], help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor for CrossEntropyLoss')
    parser.add_argument('--mixup_cutmix_prob', type=float, default=0.8, help='Probability of applying Mixup or CutMix (split equally)')
    parser.add_argument('--mixup_alpha', type=float, default=0.4, help='Alpha parameter for Mixup')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Alpha parameter for CutMix')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of training data to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # TTA
    parser.add_argument('--tta', action='store_true', default=True, help='Enable Test Time Augmentation')
    parser.add_argument('--default_pred_idx', type=int, default=3, help='Default prediction index for failed images (0:High, 1:Mid, 2:Low, 3:None, 4:Non-hallway)')

    # Misc
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--wandb_project', type=str, default='fire_hazard_detection_single_timm', help='WandB project name') # Updated default
    parser.add_argument('--wandb_run_name', type=str, default='single_model_run', help='WandB run name')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable W&B logging')

    args = parser.parse_args()
    # Ensure image size matches ViT expectations if ViT is chosen
    if 'vit' in args.model_name and '224' not in args.model_name and args.img_size != 384:
        print(f"Warning: ViT model {args.model_name} selected, but image size is {args.img_size}. ViT often expects 224 or 384. Ensure compatibility or adjust img_size.")
    elif 'vit' in args.model_name and '224' in args.model_name and args.img_size != 224:
         print(f"Warning: ViT model {args.model_name} selected, but image size is {args.img_size}. Resetting image size to 224.")
         args.img_size = 224


    return args

args = parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if args.disable_wandb:
    os.environ["WANDB_MODE"] = "disabled"

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 定义风险类别映射
risk_categories: Dict[str, int] = {
    "高风险": 0,
    "中风险": 1,
    "低风险": 2,
    "无风险": 3,
    "非楼道": 4
}
reverse_risk_categories: Dict[int, str] = {v: k for k, v in risk_categories.items()}
num_classes: int = len(risk_categories)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"Timm version: {timm.__version__}") # Log timm version
print(f"Training single model: {args.model_name}")
print(f"Using image size: {args.img_size}x{args.img_size}")


# --- Custom Warmup Scheduler (Unchanged) ---
class WarmupScheduler:
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, base_lr: float, scheduler_after_warmup: Any) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.target_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.scheduler_after_warmup = scheduler_after_warmup
        self.epoch = 0
        start_lrs = [lr / max(1, warmup_epochs) for lr in self.target_lrs] # Avoid division by zero if warmup_epochs is 0
        for i, param_group in enumerate(self.optimizer.param_groups):
             param_group['lr'] = start_lrs[i]
        print(f"WarmupScheduler initialized. Start LRs: {[f'{lr:.6f}' for lr in start_lrs]}. Target LRs after warmup: {[f'{lr:.6f}' for lr in self.target_lrs]}")

    def step(self, val_score: Optional[float] = None) -> None:
        if self.epoch < self.warmup_epochs:
            lr_factors = [(self.epoch + 1) / self.warmup_epochs] * len(self.optimizer.param_groups)
            current_lrs = [target_lr * factor for target_lr, factor in zip(self.target_lrs, lr_factors)]
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = current_lrs[i]
        else:
            if self.epoch == self.warmup_epochs:
                 print(f"Warmup complete. Switching to main scheduler: {type(self.scheduler_after_warmup).__name__}")
            if isinstance(self.scheduler_after_warmup, optim.lr_scheduler.ReduceLROnPlateau):
                if val_score is None:
                    logging.warning("Validation score needed for ReduceLROnPlateau step after warmup, but none provided.")
                else:
                    self.scheduler_after_warmup.step(val_score)
            else:
                self.scheduler_after_warmup.step()
        self.epoch += 1

    def get_last_lr(self) -> List[float]:
        return [param_group['lr'] for param_group in self.optimizer.param_groups]


# --- Dataset Class (Unchanged) ---
class FireHazardDataset(Dataset):
    def __init__(self, txt_file: str, img_dir: str, transform: Optional[A.Compose] = None, is_test: bool = False) -> None:
        try:
            self.data = pd.read_csv(txt_file, sep='\t', header=None)
            if len(self.data.columns) == 2:
                self.data.columns = ['image_name', 'label']
            elif len(self.data.columns) == 1:
                 self.data.columns = ['image_name']
                 self.data['label'] = 'unknown'
            else:
                 raise ValueError(f"Unexpected number of columns in {txt_file}")
        except Exception as e:
            logging.error(f"Failed to read label file: {txt_file} - {e}")
            raise e
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        self.original_image_names = self.data['image_name'].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str], None]:
        if idx >= len(self.data):
             logging.error(f"Index {idx} out of bounds for dataset length {len(self.data)}")
             return None

        img_info = self.data.iloc[idx]
        img_name_short = img_info['image_name']
        img_name_full = os.path.join(self.img_dir, img_name_short)

        try:
            image = Image.open(img_name_full).convert('RGB')
            image_np = np.array(image)
            if self.transform:
                augmented = self.transform(image=image_np)
                image_tensor = augmented['image']
            else:
                image_tensor = transforms.ToTensor()(image_np) # Should not happen with current setup

            if not self.is_test:
                label_str = img_info['label']
                if label_str not in risk_categories:
                    logging.warning(f"Unknown label '{label_str}' for image {img_name_short}. Assigning '无风险' (3).")
                    label = risk_categories["无风险"]
                else:
                    label = risk_categories[label_str]
                return image_tensor, label
            else:
                return image_tensor, -1, img_name_short
        except FileNotFoundError:
             logging.error(f"Image file not found: {img_name_full}")
             return None if not self.is_test else (None, -1, img_name_short)
        except Exception as e:
            logging.error(f"Cannot load/process image {img_name_full}: {e}")
            return None if not self.is_test else (None, -1, img_name_short)

# --- Custom Collate Function (Unchanged) ---
def collate_fn(batch: List[Optional[Tuple]]) -> Optional[Tuple]:
    valid_items = [item for item in batch if item is not None and (len(item) < 3 or item[0] is not None)]
    failed_names = []
    is_test = False
    if valid_items and len(valid_items[0]) == 3: # Check structure of valid items
        is_test = True
        failed_names = [item[2] for item in batch if item is None or item[0] is None] # Collect names from failed items

    if not valid_items:
        # If test mode and there were failures, return structure with only failed names
        return (None, None, failed_names) if is_test and failed_names else None

    if is_test:
        images, labels, names = zip(*valid_items)
        images = torch.stack(images, 0) if images else None
        labels = torch.tensor(labels, dtype=torch.long) # Placeholders
        all_names = list(names) + failed_names # Combine valid and failed names
        return images, labels, all_names
    else:
        images, labels = zip(*valid_items)
        images = torch.stack(images, 0)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels


# --- Data Augmentations ---
# NOTE: Use timm's recommended normalization for timm models
# timm uses mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] by default for most models
timm_mean = (0.485, 0.456, 0.406)
timm_std = (0.229, 0.224, 0.225)

train_transform = A.Compose([
    A.Resize(args.img_size, args.img_size), # Resizes to specified size
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.5, border_mode=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
    A.CoarseDropout(max_holes=8, max_height=int(args.img_size*0.1), max_width=int(args.img_size*0.1), p=0.4),
    A.Normalize(mean=timm_mean, std=timm_std), # Use timm normalization stats
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(args.img_size, args.img_size),
    A.Normalize(mean=timm_mean, std=timm_std), # Use timm normalization stats
    ToTensorV2()
])

# --- Data Loading (Unchanged logic, uses new transforms) ---
full_train_dataset = FireHazardDataset(
    txt_file=args.train_label_file,
    img_dir=args.train_img_dir,
    transform=train_transform
)

val_dataset_instance = FireHazardDataset(
    txt_file=args.train_label_file,
    img_dir=args.train_img_dir,
    transform=val_transform
)

# Stratified Split (Unchanged logic)
if not full_train_dataset.data.empty:
    label_counts = full_train_dataset.data['label'].value_counts()
    print("训练集原始类别分布：")
    print(label_counts)

    all_indices = list(range(len(full_train_dataset)))
    all_labels_str = full_train_dataset.data['label']

    min_samples_per_class = label_counts.min() if not label_counts.empty else 0
    if min_samples_per_class < 2 and args.val_split > 0:
        print(f"Warning: Class '{label_counts.idxmin()}' has only {min_samples_per_class} sample(s). Falling back to non-stratified split.")
        stratify_labels = None
    else:
         stratify_labels = all_labels_str if not label_counts.empty else None

    if args.val_split > 0:
        if stratify_labels is None and len(all_indices)>0: # Handle case where stratification fails but data exists
             print("Using non-stratified split due to lack of samples for stratification or only one class.")
             train_indices, val_indices = train_test_split(
                all_indices, test_size=args.val_split, random_state=args.seed
             )
        elif len(all_indices) == 0:
             print("Warning: Empty dataset, cannot perform split.")
             train_indices, val_indices = [], []
        else:
             try:
                 train_indices, val_indices = train_test_split(
                    all_indices,
                    test_size=args.val_split,
                    stratify=stratify_labels,
                    random_state=args.seed
                 )
             except ValueError as e:
                 print(f"Error during stratified split: {e}. Using non-stratified split.")
                 train_indices, val_indices = train_test_split(
                    all_indices, test_size=args.val_split, random_state=args.seed
                 )

        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(val_dataset_instance, val_indices)
        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        if train_indices: # Only create sampler if training data exists
            train_labels_numeric = [risk_categories.get(all_labels_str.iloc[idx], risk_categories["无风险"]) for idx in train_indices] # Handle potential KeyError safely
            class_counts_train = np.bincount(train_labels_numeric, minlength=num_classes)
            class_counts_train = np.maximum(class_counts_train, 1)
            max_count = np.max(class_counts_train) if class_counts_train.size > 0 else 1
            class_weights_sampler = max_count / class_counts_train
            sample_weights = [class_weights_sampler[label] for label in train_labels_numeric]
            sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
        else:
            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn) # No sampler needed for empty dataset

        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    else: # No validation split
         print("Training on full dataset (no validation split).")
         train_dataset = full_train_dataset
         val_loader = None
         if len(full_train_dataset) > 0:
             train_labels_numeric = [risk_categories.get(label, risk_categories["无风险"]) for label in all_labels_str] # Handle potential KeyError safely
             class_counts_train = np.bincount(train_labels_numeric, minlength=num_classes)
             class_counts_train = np.maximum(class_counts_train, 1)
             max_count = np.max(class_counts_train) if class_counts_train.size > 0 else 1
             class_weights_sampler = max_count / class_counts_train
             sample_weights = [class_weights_sampler[label] for label in train_labels_numeric]
             sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
             train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
         else:
             train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_fn) # No sampler needed for empty dataset

else:
    print("Failed to load training data or dataset is empty. Exiting.")
    exit()


# --- CutMix and Mixup Functions (Unchanged) ---
def cutmix(data: torch.Tensor, targets: torch.Tensor, alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    indices = torch.randperm(data.size(0))
    shuffled_targets = targets[indices]
    lam = np.random.beta(alpha, alpha)
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data[indices, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))
    return data, targets, shuffled_targets, lam

def rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# --- Focal Loss (Unchanged) ---
class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[Union[List[float], np.ndarray]] = None) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float)
            self.alpha = alpha
        else:
            self.alpha = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.long()
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_term = (1 - pt) ** self.gamma
        loss = focal_term * ce_loss
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weights = self.alpha.gather(0, targets)
            loss = alpha_weights * loss
        return loss.mean()

# --- EfficientNet Import (Conditional) ---
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("efficientnet-pytorch not installed. Skipping related model option.")
    if args.model_name == 'efficientnet-b3':
         print("Error: efficientnet-b3 selected but library not found. Install efficientnet-pytorch.")
         exit()

# --- Model Definition ---
# UPDATED: Added timm model support and updated param group logic
def get_param_groups(model: nn.Module, model_name: str, lr_backbone: float, lr_head: float) -> List[Dict[str, Any]]:
    head_params_names = []
    # Identify head parameters based on model type
    if model_name == 'resnet50':
        head_layer_name = 'fc' # Default torchvision resnet
    elif model_name == 'efficientnet-b3':
         head_layer_name = '_fc' # efficientnet-pytorch convention
    elif model_name.startswith('convnext_'):
        head_layer_name = 'head.fc' # timm convnext convention
    elif model_name.startswith('vit_'):
        head_layer_name = 'head' # timm ViT convention (often just 'head')
    else:
        # Fallback: try common names or assume last layer is head
        # This might need adjustment for other custom models
        potential_head_names = ['fc', 'head.fc', 'classifier', 'head']
        found = False
        for name, mod in model.named_modules():
            if name in potential_head_names or isinstance(mod, nn.Linear) and name.endswith(tuple(potential_head_names)):
                head_layer_name = name
                print(f"Auto-detected head layer name: {head_layer_name}")
                found = True
                break
        if not found:
            # If auto-detection fails, assume the last parameter belongs to the head
            last_param_name = list(model.named_parameters())[-1][0]
            head_layer_name = last_param_name.split('.')[0] # Take the module name part
            print(f"Warning: Could not auto-detect head layer. Assuming '{head_layer_name}' based on last parameter '{last_param_name}'. Check if correct.")


    # Find all parameters belonging to the head module(s)
    for name, param in model.named_parameters():
        if name.startswith(head_layer_name):
            head_params_names.append(name)

    if not head_params_names:
         print(f"Warning: No parameters found starting with the assumed head name '{head_layer_name}'. Differential LR might not work correctly.")
         # Treat all parameters as head? Or all as backbone? Let's group all as 'head' in this case.
         # head_params_names = [name for name, _ in model.named_parameters()]

    backbone_params = [p for name, p in model.named_parameters() if name not in head_params_names and p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if name in head_params_names and p.requires_grad]

    # If head_params is empty but backbone isn't, maybe the assumption was wrong. Put all in head.
    if not head_params and backbone_params:
        print("Warning: Head parameter list is empty, assigning all parameters to the 'head' group.")
        head_params = backbone_params
        backbone_params = []


    param_groups = []
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': lr_backbone, 'group_name': 'backbone'})
    if head_params:
        param_groups.append({'params': head_params, 'lr': lr_head, 'group_name': 'head'})

    print(f"Param groups for {model_name}:")
    if backbone_params: print(f"  Backbone ({len(backbone_params)} tensors): LR={lr_backbone}")
    else: print("  Backbone: No parameters assigned (or none trainable).")
    if head_params: print(f"  Head ({len(head_params)} tensors): LR={lr_head}")
    else: print("  Head: No parameters assigned (or none trainable).")

    # Debugging: list head parameter names
    # print("  Head Parameter Names:", head_params_names)

    return param_groups

# Select and build the single model
if args.model_name == 'resnet50':
    try:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) # Use newer weights API
    except TypeError:
        print("Falling back to legacy `pretrained=True` for ResNet50.")
        model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes) # Replace head
    # Fine-tuning strategy (optional, example)
    # for name, param in model.named_parameters():
    #     if not (name.startswith('layer4') or name.startswith('fc')):
    #         param.requires_grad = False

elif args.model_name == 'efficientnet-b3':
    try:
        model = EfficientNet.from_pretrained('efficientnet-b3')
        model._fc = nn.Linear(model._fc.in_features, num_classes) # Replace head
    except NameError:
         print("Error: EfficientNet class not available. Cannot create efficientnet-b3.")
         exit()
    # Fine-tuning strategy (optional, example)
    # unfreeze_start_block_idx = 20
    # for name, param in model.named_parameters():
    #     param.requires_grad = False
    #     if name.startswith('_fc') or name.startswith('_conv_head') or name.startswith('_bn1'):
    #         param.requires_grad = True
    #     if name.startswith('_blocks'):
    #         try:
    #             block_idx = int(name.split('.')[1])
    #             if block_idx >= unfreeze_start_block_idx:
    #                 param.requires_grad = True
    #         except (IndexError, ValueError): pass

# NEW: Use timm for ConvNeXt and ViT
elif args.model_name.startswith('convnext_') or args.model_name.startswith('vit_'):
    try:
        # timm automatically replaces the head if num_classes is different from pretrained head
        model = timm.create_model(args.model_name, pretrained=True, num_classes=num_classes)
        print(f"Loaded timm model: {args.model_name} with {num_classes} output classes.")
        # Optional: Add dropout if desired (timm models might have it already)
        # Example: Replace ViT head with dropout
        # if args.model_name.startswith('vit_') and hasattr(model, 'head') and isinstance(model.head, nn.Linear):
        #      in_features = model.head.in_features
        #      model.head = nn.Sequential(
        #          nn.Dropout(0.5), # Add dropout
        #          nn.Linear(in_features, num_classes)
        #      )
    except Exception as e:
        print(f"Error creating timm model '{args.model_name}': {e}")
        print("Please ensure the model name is correct and timm is installed.")
        exit()
else:
    raise ValueError(f"Unsupported model name: {args.model_name}")

model = model.to(device)

# Get parameter groups for differential LR
param_groups = get_param_groups(model, args.model_name, args.lr_backbone, args.lr_head)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model ({args.model_name}) Trainable Params: {count_parameters(model)}")


# --- Loss Function (Unchanged) ---
if args.loss_type == 'focal':
    focal_alpha = [2.0, 1.5, 1.0, 1.0, 1.0] # Example weights
    print(f"Using Focal Loss with gamma={args.focal_gamma}, alpha={focal_alpha}")
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=focal_alpha).to(device)
elif args.loss_type == 'crossentropy_smooth':
    print(f"Using CrossEntropyLoss with label smoothing={args.label_smoothing}")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
else:
    raise ValueError(f"Unsupported loss type: {args.loss_type}")


# --- Optimizer (Logic unchanged, uses new param_groups) ---
if not param_groups:
     print("Error: No parameter groups defined for the optimizer. Check model structure and get_param_groups function.")
     # Fallback: Create a single group with default LR if needed, though differential LR is intended
     # param_groups = [{'params': model.parameters(), 'lr': args.lr_head}] # Use head LR for all if groups fail
     exit()

optimizer = optim.AdamW(
    param_groups, # Uses the potentially differential LR groups
    weight_decay=args.weight_decay
)

# --- Learning Rate Scheduler (Logic unchanged) ---
# Ensure T_max calculation is correct based on actual number of training epochs after warmup
# If warmup_epochs >= args.epochs, scheduler might behave unexpectedly. Add check?
t_max_cosine = max(1, args.epochs - args.warmup_epochs) # Ensure T_max is at least 1
scheduler_cos = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_cosine, eta_min=1e-7)
scheduler = WarmupScheduler(optimizer, warmup_epochs=args.warmup_epochs, base_lr=args.lr_head, scheduler_after_warmup=scheduler_cos)


# --- Initialize W&B (Unchanged) ---
wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name if args.wandb_run_name else f"{args.model_name}-run", # Default run name includes model
    config=vars(args)
)


# --- Training Function (Unchanged) ---
def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, mixup_cutmix_prob: float, mixup_alpha: float, cutmix_alpha: float) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        if batch is None: continue
        # Check if batch structure is correct (images, labels)
        if not isinstance(batch, (list, tuple)) or len(batch) != 2 or batch[0] is None or batch[1] is None:
             logging.warning(f"Skipping malformed batch in train: type {type(batch)}, len {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
             continue

        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        r = np.random.rand()
        apply_mixup_cutmix = mixup_cutmix_prob > 0 and mixup_alpha > 0 and cutmix_alpha > 0

        if apply_mixup_cutmix and r < mixup_cutmix_prob / 2: # CutMix
            images, targets_a, targets_b, lam = cutmix(images, labels, alpha=cutmix_alpha)
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif apply_mixup_cutmix and r < mixup_cutmix_prob: # Mixup
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else: # Standard
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.detach(), 1)
        total += labels.size(0)
        # Use original labels for accuracy calculation even with mixup/cutmix
        if apply_mixup_cutmix and (r < mixup_cutmix_prob):
             # Approx accuracy: check if prediction matches either original label
             correct += (lam * (predicted == targets_a).float() + (1 - lam) * (predicted == targets_b).float()).sum().item()
        else:
             correct += (predicted == labels).sum().item()


    if total == 0: return 0.0, 0.0
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# --- Validation Function (Unchanged) ---
def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds_val = []
    all_labels_val = []
    all_probs_val = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None: continue
            if not isinstance(batch, (list, tuple)) or len(batch) != 2 or batch[0] is None or batch[1] is None:
                 logging.warning(f"Skipping malformed batch in validate: type {type(batch)}, len {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
                 continue

            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(probabilities, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds_val.extend(predicted.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())
            all_probs_val.extend(probabilities.cpu().numpy())

    if total == 0: return 0.0, 0.0, np.array([]), np.array([]), np.array([])
    val_loss = running_loss / total
    val_acc = 100 * correct / total
    return val_loss, val_acc, np.array(all_labels_val), np.array(all_preds_val), np.array(all_probs_val)


# --- Weighted F1 Calculation (Unchanged) ---
competition_f1_weights = [2.0, 1.5, 1.0, 1.0, 1.0]

def compute_weighted_f1(all_labels: np.ndarray, all_preds: np.ndarray, competition_weights: List[float], num_classes: int) -> float:
    if len(all_labels) == 0 or len(all_preds) == 0:
        # print("Warning: No data found for F1 calculation in validation set.") # Reduced verbosity
        return 0.0

    present_labels = np.unique(all_labels)
    if len(present_labels) == 0: return 0.0

    precision, recall, f1_scores, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    # Only print metrics if support > 0? Optional.
    # print(f"  Metrics per class {list(risk_categories.keys())}:")
    # print(f"  Precision: {[f'{p:.4f}' for p in precision]}")
    # print(f"  Recall:    {[f'{r:.4f}' for r in recall]}")
    # print(f"  F1 Scores: {[f'{f:.4f}' for f in f1_scores]}")
    # print(f"  Support:   {support}")

    if len(competition_weights) != len(f1_scores):
         logging.error("Length of competition_weights must match the number of classes")
         return 0.0

    weighted_f1_numerator = sum(f * w for f, w, s in zip(f1_scores, competition_weights, support) if s > 0) # Ensure class was present
    weighted_f1_denominator = sum(w for w, s in zip(competition_weights, support) if s > 0)

    if weighted_f1_denominator == 0:
        # print("Warning: Weighted F1 denominator is zero. Returning 0.0.") # Reduced verbosity
         return 0.0

    competition_f1 = weighted_f1_numerator / weighted_f1_denominator
    return competition_f1


# --- Training Loop (Logic Unchanged) ---
best_val_score = 0.0
trigger_times = 0
best_epoch = 0

print("\nStarting training loop...")
for epoch in range(args.epochs):
    print("-" * 20)
    print(f"Epoch {epoch+1}/{args.epochs}")
    current_lrs = scheduler.get_last_lr()
    lr_log_str = " | ".join([f"{pg['group_name']}: {lr:.1e}" for pg, lr in zip(optimizer.param_groups, current_lrs)])
    print(f"  LRs: {lr_log_str}")


    # --- Train Model ---
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device, args.mixup_cutmix_prob, args.mixup_alpha, args.cutmix_alpha)
    print(f'[{args.model_name}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')

    # --- Validate Model ---
    val_score = 0.0 # Default score if no validation
    if val_loader:
        val_loss, val_acc, val_labels, val_preds, _ = validate(model, val_loader, criterion, device)
        if len(val_labels) > 0: # Check if validation produced results
             val_score = compute_weighted_f1(val_labels, val_preds, competition_f1_weights, num_classes)
             print(f'[{args.model_name}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Comp F1: {val_score:.4f}')

             # --- Logging & Checkpointing ---
             log_dict = {
                 "epoch": epoch + 1,
                 "train_loss": train_loss, "train_accuracy": train_acc,
                 "val_loss": val_loss, "val_accuracy": val_acc, "val_f1_score": val_score,
             }
             # Log LRs per group
             for i, lr in enumerate(current_lrs):
                  group_name = optimizer.param_groups[i].get('group_name', f'group_{i}')
                  log_dict[f'lr_{group_name}'] = lr
             wandb.log(log_dict)


             # Checkpoint and Early Stopping
             if val_score > best_val_score:
                 best_val_score = val_score
                 torch.save(model.state_dict(), args.model_save_path)
                 wandb.save(args.model_save_path) # Save best model to W&B artifact
                 trigger_times = 0
                 best_epoch = epoch + 1
                 print(f"  Best Score Updated: {best_val_score:.4f} at epoch {best_epoch}")
             else:
                 trigger_times += 1
                 print(f'  No Improvement: {trigger_times}/{args.patience}')

             # Early stopping condition
             if trigger_times >= args.patience:
                 print(f'Early stopping triggered after epoch {epoch+1}.')
                 break
        else:
             print(f'[{args.model_name}] Validation step produced no results (empty val_loader or all batches failed?). Skipping metrics and checkpointing.')
             # Log training metrics only if validation failed
             log_dict = {"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc}
             for i, lr in enumerate(current_lrs):
                 group_name = optimizer.param_groups[i].get('group_name', f'group_{i}')
                 log_dict[f'lr_{group_name}'] = lr
             wandb.log(log_dict)


    else: # No validation loader
        log_dict = {"epoch": epoch + 1, "train_loss": train_loss, "train_accuracy": train_acc}
        for i, lr in enumerate(current_lrs):
             group_name = optimizer.param_groups[i].get('group_name', f'group_{i}')
             log_dict[f'lr_{group_name}'] = lr
        wandb.log(log_dict)

        if epoch == args.epochs - 1:
             print("Saving model from last epoch as no validation split was used.")
             torch.save(model.state_dict(), args.model_save_path)
             wandb.save(args.model_save_path)
             best_epoch = args.epochs
             best_val_score = -1.0 # Indicate no validation score

    # Step scheduler (pass score only if ReduceLROnPlateau is used)
    # scheduler.step(val_score if isinstance(scheduler.scheduler_after_warmup, optim.lr_scheduler.ReduceLROnPlateau) else None) # More robust step call
    scheduler.step() # Simpler call, assumes CosineAnnealing or similar


wandb.log({
    "best_val_f1_score": best_val_score,
    "best_epoch": best_epoch
})

print("="*30)
print(f"Training finished.")
if val_loader:
    print(f"Best Model ({args.model_name}) score: {best_val_score:.4f} at epoch {best_epoch}")
else:
    print("Training finished (no validation performed). Using model from last epoch.")
print("="*30)

# --- Load Best Model ---
if os.path.exists(args.model_save_path):
    try:
        model.load_state_dict(torch.load(args.model_save_path, map_location=device))
        print("Best model loaded successfully from:", args.model_save_path)
    except Exception as e:
         print(f"Error loading best model weights from {args.model_save_path}: {e}. Using last state model from training.")
else:
    print(f"Warning: Saved model file not found at {args.model_save_path}. Using last state model from training.")


# --- Test Data Loading (Unchanged logic) ---
test_dataset = FireHazardDataset(
    txt_file=args.test_label_file,
    img_dir=args.test_img_dir,
    transform=val_transform, # Use validation transform for test
    is_test=True
)
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())


# --- Single Model Prediction (with TTA option - Unchanged Logic) ---
def predict_tta(model: nn.Module, test_loader: DataLoader, device: torch.device, use_tta: bool, num_classes: int, default_pred_idx: int) -> Dict[str, int]:
    model.eval()
    predictions_map: Dict[str, int] = {}

    tta_transforms: List[Any] = []
    if use_tta:
        print("Performing Test Time Augmentation...")
        # Define TTA transforms (example: original + horizontal flip)
        tta_transforms = [
            lambda x: x, # Original
            lambda x: torch.flip(x, dims=[-1]), # Horizontal flip
            # lambda x: torch.flip(x, dims=[-2]), # Optional: Vertical flip (check domain relevance)
        ]
        print(f"  Using {len(tta_transforms)} TTA transforms.")
    else:
        print("TTA disabled. Using single inference.")
        tta_transforms = [lambda x: x]

    with torch.no_grad():
        batch_count = 0
        for batch in test_loader:
            batch_count += 1
            if batch_count % 50 == 0:
                 print(f"  Processing test batch {batch_count}/{len(test_loader)}")

            images, _, names = None, None, None
            if batch is None:
                 logging.warning("Skipping empty batch in test_loader.")
                 continue
            # Expect (images_tensor | None, labels_tensor, names_list) from collate_fn
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                 images, _, names = batch
            else:
                 logging.error(f"Unexpected batch format in test_loader: type {type(batch)}, len {len(batch) if isinstance(batch, (list, tuple)) else 'N/A'}")
                 continue

            if not names: # Check if names list is empty or None
                 logging.warning("Skipping batch with no image names.")
                 continue

            predicted_indices_list = None # Initialize
            if images is not None: # Only predict if images were loaded
                images = images.to(device)
                batch_size = images.size(0)
                batch_tta_probs = torch.zeros(batch_size, num_classes).to(device)

                for transform in tta_transforms:
                    try:
                        tta_images = transform(images)
                        outputs = model(tta_images)
                        probs = torch.softmax(outputs, dim=1)
                        batch_tta_probs += probs
                    except Exception as e:
                         logging.error(f"Error during TTA prediction on batch: {e}")
                         # Handle error for this transform, maybe break or continue?
                         # For now, we just log and continue, results might be skewed.
                         batch_tta_probs += torch.zeros_like(batch_tta_probs) # Add zeros if TTA fails

                final_probs = batch_tta_probs / max(1, len(tta_transforms)) # Average probabilities
                _, predicted_indices = torch.max(final_probs, 1)
                predicted_indices_list = predicted_indices.cpu().numpy()

            # Map predictions to image names (handle failed images)
            valid_pred_idx = 0
            for name in names:
                 # If images were loaded AND we haven't exhausted the predictions for them
                 if predicted_indices_list is not None and valid_pred_idx < len(predicted_indices_list):
                     predictions_map[name] = predicted_indices_list[valid_pred_idx]
                     valid_pred_idx += 1
                 else:
                     # Assign default prediction if image loading failed or prediction failed
                     if name not in predictions_map: # Avoid overwriting if name appeared twice with one failure
                         logging.warning(f"Assigning default prediction ({default_pred_idx}: {reverse_risk_categories.get(default_pred_idx, 'Unknown')}) for unloaded/failed image: {name}")
                         predictions_map[name] = default_pred_idx

    return predictions_map

# --- Generate Predictions (Unchanged logic) ---
print(f"\nGenerating predictions on the test set {'with TTA' if args.tta else 'without TTA'}...")
predictions_map = predict_tta(model, test_loader, device, args.tta, num_classes, args.default_pred_idx)

# --- Create Submission File (Unchanged logic) ---
print(f"Writing predictions to {args.output_file}...")
ordered_predictions = []
missing_predictions = 0
# Use original order from the test dataset object
original_test_order = test_dataset.original_image_names

for name in original_test_order:
    pred_index = predictions_map.get(name)
    if pred_index is None:
        # This case should be less likely now with the collate_fn/predict_tta update, but keep as safeguard
        logging.error(f"Prediction missing for image in final ordering: {name}. Assigning default ({args.default_pred_idx}).")
        pred_index = args.default_pred_idx
        missing_predictions += 1
    ordered_predictions.append(pred_index)

if missing_predictions > 0:
     print(f"Warning: {missing_predictions} predictions were missing during final ordering and assigned default.")


submission_df = pd.DataFrame({
    'image_name': original_test_order,
    'prediction_index': ordered_predictions
})
# Map index back to label string, handle potential missing keys in map
submission_df['risk_label'] = submission_df['prediction_index'].map(lambda x: reverse_risk_categories.get(x, f"UnknownIndex_{x}"))


try:
    # Ensure the output format is exactly image_name<TAB>risk_label
    submission_df[['image_name', 'risk_label']].to_csv(
        args.output_file, sep='\t', index=False, header=False, encoding='utf-8'
    )
    print(f"Submission file '{args.output_file}' generated successfully with {len(ordered_predictions)} predictions.")
except Exception as e:
    print(f"Error writing submission file: {e}")


try:
    if not args.disable_wandb and wandb.run is not None: # Check if wandb is active
        wandb.log({"test_predictions": wandb.Table(dataframe=submission_df[['image_name', 'risk_label']])})
except Exception as e:
     print(f"Could not log submission table to WandB: {e}")


print("Script finished.")
if not args.disable_wandb and wandb.run is not None:
    wandb.finish()