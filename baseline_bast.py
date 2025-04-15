# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler, Subset # Added Subset
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
import argparse # Added for configuration
from typing import List, Tuple, Optional, Dict, Any, Union # Added for type hinting
# from scipy.optimize import minimize # Optional: Needed for advanced weight optimization

# --- Configuration via argparse ---
def parse_args():
    parser = argparse.ArgumentParser(description='Fire Hazard Detection Training and Inference')

    # Paths
    parser.add_argument('--train_label_file', type=str, default='Data/智慧骑士_label/train.txt', help='Path to training label file')
    parser.add_argument('--train_img_dir', type=str, default='Data/智慧骑士_train/train', help='Path to training image directory')
    parser.add_argument('--test_label_file', type=str, default='Data/智慧骑士_label/A.txt', help='Path to test label file (for ordering)')
    parser.add_argument('--test_img_dir', type=str, default='Data/智慧骑士_A/A', help='Path to test image directory')
    parser.add_argument('--output_file', type=str, default='submit_optimized.txt', help='Path to output submission file')
    parser.add_argument('--model1_save_path', type=str, default='best_model1_optimized.pth', help='Path to save best model 1 weights')
    parser.add_argument('--model2_save_path', type=str, default='best_model2_optimized.pth', help='Path to save best model 2 weights')

    # Model & Training Hyperparameters
    parser.add_argument('--img_size', type=int, default=224, help='Image size (height and width)')
    parser.add_argument('--batch_size', type=int, default=32, help='Training and validation batch size')
    parser.add_argument('--test_batch_size', type=int, default=64, help='Test batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Maximum number of training epochs')
    parser.add_argument('--lr_head', type=float, default=1e-4, help='Learning rate for the classifier head')
    parser.add_argument('--lr_backbone', type=float, default=1e-5, help='Learning rate for the pre-trained backbone (differential LR)')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 penalty)')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='Number of linear warmup epochs')
    parser.add_argument('--loss_type', type=str, default='focal', choices=['focal', 'crossentropy_smooth'], help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Gamma parameter for Focal Loss')
    # Consider making focal_alpha configurable via string input e.g., "2.0,1.5,1.0,1.0,1.0"
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor for CrossEntropyLoss')
    parser.add_argument('--mixup_cutmix_prob', type=float, default=0.8, help='Probability of applying Mixup or CutMix (split equally)')
    parser.add_argument('--mixup_alpha', type=float, default=0.4, help='Alpha parameter for Mixup')
    parser.add_argument('--cutmix_alpha', type=float, default=1.0, help='Alpha parameter for CutMix')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience (increased default)') # Increased default
    parser.add_argument('--val_split', type=float, default=0.2, help='Fraction of training data to use for validation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # TTA & Ensemble
    parser.add_argument('--tta', action='store_true', default=True, help='Enable Test Time Augmentation') # Changed default to True based on original code
    parser.add_argument('--optimize_ensemble_weights', action='store_true', help='Optimize ensemble weights on validation set')
    parser.add_argument('--default_pred_idx', type=int, default=3, help='Default prediction index for failed images (0:High, 1:Mid, 2:Low, 3:None, 4:Non-hallway)')


    # Misc
    parser.add_argument('--num_workers', type=int, default=0, help='Number of DataLoader workers (0 recommended for baseline comparison, increase for speed)') # Kept default 0 as requested
    parser.add_argument('--wandb_project', type=str, default='fire_hazard_detection_optimized', help='WandB project name')
    parser.add_argument('--wandb_run_name', type=str, default='optimized_run', help='WandB run name')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable W&B logging')

    args = parser.parse_args()
    return args

args = parse_args()

# Set random seed
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    # Consider adding these for reproducibility, but they might impact performance
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

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

# --- Custom Warmup Scheduler ---
class WarmupScheduler:
    def __init__(self, optimizer: optim.Optimizer, warmup_epochs: int, base_lr: float, scheduler_after_warmup: Any) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        # Base_lr here should be the target LR *after* warmup (e.g., args.lr_head)
        # We need to handle differential LR where base_lr might differ per group
        self.target_lrs = [pg['lr'] for pg in optimizer.param_groups] # Get target LRs from optimizer setup
        self.scheduler_after_warmup = scheduler_after_warmup
        self.epoch = 0
        # Initialize learning rates close to 0 for warmup start
        start_lrs = [lr / warmup_epochs if warmup_epochs > 0 else lr for lr in self.target_lrs]
        for i, param_group in enumerate(self.optimizer.param_groups):
             param_group['lr'] = start_lrs[i]
        print(f"WarmupScheduler initialized. Start LRs: {[f'{lr:.6f}' for lr in start_lrs]}. Target LRs after warmup: {[f'{lr:.6f}' for lr in self.target_lrs]}")

    def step(self, val_score: Optional[float] = None) -> None:
        if self.epoch < self.warmup_epochs:
            # Linearly increase learning rate for each param group
            lr_factors = [(self.epoch + 1) / self.warmup_epochs] * len(self.optimizer.param_groups)
            current_lrs = [target_lr * factor for target_lr, factor in zip(self.target_lrs, lr_factors)]
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = current_lrs[i]
            # Optional logging: print(f"Warmup Epoch {self.epoch+1}/{self.warmup_epochs}, LRs set to {[f'{lr:.6f}' for lr in current_lrs]}")
        else:
            if self.epoch == self.warmup_epochs:
                 print(f"Warmup complete. Switching to main scheduler: {type(self.scheduler_after_warmup).__name__}")
                 # Ensure the main scheduler starts from the correct LR if needed (esp. if its state wasn't updated during warmup)
                 # For CosineAnnealingLR, it typically calculates based on T_max and current epoch, so it should be fine.
                 # For ReduceLROnPlateau, the initial step needs the score.

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

# --- Dataset Class ---
class FireHazardDataset(Dataset):
    def __init__(self, txt_file: str, img_dir: str, transform: Optional[A.Compose] = None, is_test: bool = False) -> None:
        try:
            # Read labels, ensure 'image_name' column exists even if no header
            self.data = pd.read_csv(txt_file, sep='\t', header=None)
            if len(self.data.columns) == 2:
                self.data.columns = ['image_name', 'label']
            elif len(self.data.columns) == 1:
                 self.data.columns = ['image_name'] # Test file might only have names
                 self.data['label'] = 'unknown' # Placeholder
            else:
                 raise ValueError(f"Unexpected number of columns in {txt_file}")

        except Exception as e:
            logging.error(f"Failed to read label file: {txt_file} - {e}")
            raise e
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test
        # Store original image names list for test set ordering
        self.original_image_names = self.data['image_name'].tolist()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str], None]:
        if idx >= len(self.data):
             logging.error(f"Index {idx} out of bounds for dataset length {len(self.data)}")
             return None # Handled by collate_fn

        img_info = self.data.iloc[idx]
        img_name_short = img_info['image_name']
        img_name_full = os.path.join(self.img_dir, img_name_short)

        try:
            # Using PIL for consistency with original code, could switch to OpenCV if needed
            image = Image.open(img_name_full).convert('RGB')
            image_np = np.array(image) # Convert to numpy array for Albumentations

            if self.transform:
                augmented = self.transform(image=image_np)
                image_tensor = augmented['image']
            else:
                # Basic transform if none provided (shouldn't happen with current setup)
                image_tensor = transforms.ToTensor()(image_np)

            if not self.is_test:
                label_str = img_info['label']
                if label_str not in risk_categories:
                    logging.warning(f"Unknown label '{label_str}' for image {img_name_short}. Assigning '无风险' (3).")
                    # Assign a default label, e.g., '无风险'
                    label = risk_categories["无风险"] # Or use args.default_pred_idx
                    # Alternatively, return None to skip this sample:
                    # return None
                else:
                    label = risk_categories[label_str]
                return image_tensor, label
            else:
                # For test set, return image tensor, placeholder label (-1), and image name
                return image_tensor, -1, img_name_short

        except FileNotFoundError:
             logging.error(f"Image file not found: {img_name_full}")
             return None if not self.is_test else (None, -1, img_name_short) # Return structure for collate_fn
        except Exception as e:
            logging.error(f"Cannot load/process image {img_name_full}: {e}")
            return None if not self.is_test else (None, -1, img_name_short) # Return structure for collate_fn

# --- Custom Collate Function ---
def collate_fn(batch: List[Optional[Tuple]]) -> Optional[Tuple]:
    valid_items = [item for item in batch if item is not None and (len(item) < 3 or item[0] is not None)] # Check if image tensor exists for test items
    failed_names = []
    is_test = False
    if batch and len(batch[0]) == 3: # Check if it's test mode structure
        is_test = True
        failed_names = [item[2] for item in batch if item is None or item[0] is None]

    if not valid_items:
        return (None, None, failed_names) if is_test and failed_names else None

    if is_test: # Test mode: image, label_placeholder, name
        images, labels, names = zip(*valid_items)
        images = torch.stack(images, 0) if images else None # Handle case where all valid items had loading issues after None filter
        labels = torch.tensor(labels, dtype=torch.long)
        all_names = list(names) + failed_names
        return images, labels, all_names
    else: # Train/Val mode: image, label
        images, labels = zip(*valid_items)
        images = torch.stack(images, 0)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, labels


# --- Data Augmentations ---
# Consider tuning these parameters or adding/removing augmentations
train_transform = A.Compose([
    A.Resize(args.img_size, args.img_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3), # Added VerticalFlip, adjust p based on domain relevance
    A.Rotate(limit=30, p=0.5, border_mode=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    A.GaussNoise(var_limit=(10.0, 40.0), p=0.3),
    # CoarseDropout can be aggressive, review parameters or probability
    A.CoarseDropout(max_holes=8, max_height=int(args.img_size*0.1), max_width=int(args.img_size*0.1), p=0.4),
    # Consider adding other transforms like GridDistortion, ElasticTransform, RandomResizedCrop if beneficial
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(args.img_size, args.img_size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])

# --- Data Loading ---
full_train_dataset = FireHazardDataset(
    txt_file=args.train_label_file,
    img_dir=args.train_img_dir,
    transform=train_transform # Will be used by the train subset
)

# Separate dataset instance for validation with validation transform
val_dataset_instance = FireHazardDataset(
    txt_file=args.train_label_file,
    img_dir=args.train_img_dir,
    transform=val_transform # Assign validation transform here
)

# Stratified Split
if not full_train_dataset.data.empty:
    label_counts = full_train_dataset.data['label'].value_counts()
    print("训练集原始类别分布：")
    print(label_counts)

    all_indices = list(range(len(full_train_dataset)))
    all_labels_str = full_train_dataset.data['label'] # Use string labels for stratification

    min_samples_per_class = label_counts.min()
    if min_samples_per_class < 2 and args.val_split > 0:
        print(f"Warning: Class '{label_counts.idxmin()}' has only {min_samples_per_class} sample(s). Stratification might fail. Falling back to non-stratified split.")
        stratify_labels = None
    else:
         stratify_labels = all_labels_str

    if args.val_split > 0:
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

        # Create Subset objects using the correct dataset instances
        train_dataset = Subset(full_train_dataset, train_indices)
        # IMPORTANT: Use val_dataset_instance for the validation subset!
        val_dataset = Subset(val_dataset_instance, val_indices)

        print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

        # Weighted Random Sampler for Training Set
        # Get numeric labels corresponding to the train_indices
        train_labels_numeric = [risk_categories[all_labels_str.iloc[idx]] for idx in train_indices]

        class_counts_train = np.bincount(train_labels_numeric, minlength=num_classes)
        class_counts_train = np.maximum(class_counts_train, 1) # Avoid division by zero

        max_count = np.max(class_counts_train)
        class_weights_sampler = max_count / class_counts_train
        sample_weights = [class_weights_sampler[label] for label in train_labels_numeric]

        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

    else: # No validation split, train on full data
         print("Training on full dataset (no validation split).")
         train_dataset = full_train_dataset
         val_loader = None # No validation loader
         # Weighted Random Sampler for Full Training Set
         train_labels_numeric = [risk_categories[label] for label in all_labels_str]
         class_counts_train = np.bincount(train_labels_numeric, minlength=num_classes)
         class_counts_train = np.maximum(class_counts_train, 1)
         max_count = np.max(class_counts_train)
         class_weights_sampler = max_count / class_counts_train
         sample_weights = [class_weights_sampler[label] for label in train_labels_numeric]
         sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
         train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())

else:
    print("Failed to load training data. Exiting.")
    exit()


# --- CutMix and Mixup Functions ---
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


# --- Focal Loss ---
class FocalLoss(nn.Module):
    # Note: Implementing label smoothing directly in this Focal Loss is non-trivial.
    # Consider using standard CrossEntropyLoss with label_smoothing if needed.
    def __init__(self, gamma: float = 2.0, alpha: Optional[Union[List[float], np.ndarray]] = None) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float)
            # Ensure alpha is on the correct device later, during forward pass maybe?
            # Or assume it's moved when the module is moved.
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
             # Move alpha to the same device as inputs if necessary
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weights = self.alpha.gather(0, targets)
            loss = alpha_weights * loss

        return loss.mean()

# --- EfficientNet Import ---
try:
    from efficientnet_pytorch import EfficientNet
except ImportError:
    print("Please install efficientnet-pytorch: pip install efficientnet-pytorch")
    exit()

# --- Model Definition ---
# Consider trying other models: ResNeXt, ConvNeXt, Swin Transformer, etc.

# Function to separate backbone and head parameters for differential LR
def get_param_groups(model: nn.Module, model_name: str, lr_backbone: float, lr_head: float) -> List[Dict[str, Any]]:
    head_params_names = []
    if model_name == 'resnet50':
        # Identify head parameters (fc layer)
        for name, _ in model.named_parameters():
            if name.startswith('fc'):
                head_params_names.append(name)
    elif model_name == 'efficientnet-b3':
         # Identify head parameters (_fc, _conv_head, _bn1)
         for name, _ in model.named_parameters():
             if name.startswith(('_fc', '_conv_head', '_bn1')):
                 head_params_names.append(name)
    else:
        raise ValueError(f"Unknown model name for param grouping: {model_name}")

    backbone_params = [p for name, p in model.named_parameters() if name not in head_params_names and p.requires_grad]
    head_params = [p for name, p in model.named_parameters() if name in head_params_names and p.requires_grad]

    param_groups = [
        {'params': backbone_params, 'lr': lr_backbone, 'group_name': 'backbone'},
        {'params': head_params, 'lr': lr_head, 'group_name': 'head'},
    ]
    print(f"Param groups for {model_name}:")
    print(f"  Backbone ({len(backbone_params)} tensors): LR={lr_backbone}")
    print(f"  Head ({len(head_params)} tensors): LR={lr_head}")
    return param_groups

# Model 1: ResNet50
# Use weights argument for newer torchvision, fallback to pretrained for older versions
try:
    #model1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model1 = models.resnet50(pretrained=True)
except TypeError:
    print("Using legacy `pretrained=True` for ResNet50.")
    model1 = models.resnet50(pretrained=True)

model1.fc = nn.Sequential(
    nn.Dropout(0.5), # Consider tuning dropout rate
    nn.Linear(model1.fc.in_features, num_classes)
)
# Fine-tuning strategy: Freeze earlier layers, unfreeze later layers
# Experiment with which layers to freeze/unfreeze
for name, param in model1.named_parameters():
    if 'layer4' in name or 'fc' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False # Freeze layers 0, 1, 2, 3
model1 = model1.to(device)
param_groups1 = get_param_groups(model1, 'resnet50', args.lr_backbone, args.lr_head)


# Model 2: EfficientNet-B3
# Consider trying EfficientNet-B4, B5 etc. if resources allow
model2 = EfficientNet.from_pretrained('efficientnet-b3')
model2._fc = nn.Sequential(
    nn.Dropout(0.5), # Consider tuning dropout rate
    nn.Linear(model2._fc.in_features, num_classes)
)
# Fine-tuning strategy: Unfreeze later blocks + head
# Experiment with unfreeze_start_block_idx
unfreeze_start_block_idx = 20 # Example, adjust as needed
for name, param in model2.named_parameters():
     param.requires_grad = False # Freeze all first
     if name.startswith('_fc') or name.startswith('_conv_head') or name.startswith('_bn1'):
         param.requires_grad = True
     if name.startswith('_blocks'):
         try:
             block_idx = int(name.split('.')[1])
             if block_idx >= unfreeze_start_block_idx:
                  param.requires_grad = True
         except (IndexError, ValueError):
             pass # Ignore layers that don't match the pattern
model2 = model2.to(device)
param_groups2 = get_param_groups(model2, 'efficientnet-b3', args.lr_backbone, args.lr_head)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model 1 (ResNet50) Trainable Params: {count_parameters(model1)}")
print(f"Model 2 (EfficientNetB3) Trainable Params: {count_parameters(model2)}")


# --- Loss Function ---
if args.loss_type == 'focal':
    # Define alpha weights for Focal Loss (consider making these configurable)
    # These weights correspond to [High, Mid, Low, None, Non-hallway] risk
    focal_alpha = [2.0, 1.5, 1.0, 1.0, 1.0] # Example competition weights
    print(f"Using Focal Loss with gamma={args.focal_gamma}, alpha={focal_alpha}")
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=focal_alpha).to(device)
elif args.loss_type == 'crossentropy_smooth':
    print(f"Using CrossEntropyLoss with label smoothing={args.label_smoothing}")
    # Note: Standard CrossEntropyLoss doesn't inherently use alpha class weights like FocalLoss.
    # If class weighting is desired here, calculate weights (e.g., inverse frequency)
    # and pass them via the `weight` argument:
    # label_counts_all = full_train_dataset.data['label'].value_counts().sort_index()
    # total_samples = len(full_train_dataset)
    # class_weights = total_samples / (num_classes * label_counts_all)
    # weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
    # criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=weights_tensor).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).to(device)
else:
    raise ValueError(f"Unsupported loss type: {args.loss_type}")


# --- Optimizer ---
# Consider trying Adam instead of AdamW, or SGD with momentum
optimizer1 = optim.AdamW(
    param_groups1,
    # lr is set within param_groups
    weight_decay=args.weight_decay
)
optimizer2 = optim.AdamW(
    param_groups2,
    # lr is set within param_groups
    weight_decay=args.weight_decay
)


# --- Learning Rate Scheduler ---
# Using Warmup + Cosine Annealing
# Ensure T_max calculation is correct based on actual number of training epochs after warmup
scheduler_cos1 = optim.lr_scheduler.CosineAnnealingLR(optimizer1, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7) # Lower eta_min slightly
scheduler1 = WarmupScheduler(optimizer1, warmup_epochs=args.warmup_epochs, base_lr=args.lr_head, scheduler_after_warmup=scheduler_cos1) # base_lr here is mainly for logging init

scheduler_cos2 = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=args.epochs - args.warmup_epochs, eta_min=1e-7) # Lower eta_min slightly
scheduler2 = WarmupScheduler(optimizer2, warmup_epochs=args.warmup_epochs, base_lr=args.lr_head, scheduler_after_warmup=scheduler_cos2) # base_lr here is mainly for logging init


# --- Initialize W&B ---
# Consider using hyperparameter tuning tools like Optuna or Ray Tune for systematic search
wandb.init(
    project=args.wandb_project,
    name=args.wandb_run_name,
    config=vars(args) # Log all command line arguments
)


# --- Training Function ---
def train(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device, mixup_cutmix_prob: float, mixup_alpha: float, cutmix_alpha: float) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        if batch is None: continue
        if len(batch) != 2 or batch[0] is None or batch[1] is None:
            logging.warning(f"Skipping malformed batch in train: {type(batch)} len {len(batch)}")
            continue

        images, labels = batch
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        r = np.random.rand()
        if r < mixup_cutmix_prob / 2: # CutMix
            images, targets_a, targets_b, lam = cutmix(images, labels, alpha=cutmix_alpha)
            outputs = model(images)
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
        elif r < mixup_cutmix_prob: # Mixup
            images, targets_a, targets_b, lam = mixup_data(images, labels, alpha=mixup_alpha, device=device)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        else: # Standard
            outputs = model(images)
            loss = criterion(outputs, labels)

        loss.backward()
        # Optional: Gradient Clipping (uncomment if gradients explode)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.detach(), 1) # Use detach() for metrics calculation
        total += labels.size(0)
        # Accuracy measured against original labels even with mixup/cutmix
        correct += (predicted == labels).sum().item()

    if total == 0: return 0.0, 0.0
    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc

# --- Validation Function ---
# Modified to return predictions and labels (or probabilities) for ensemble weight optimization
def validate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds_val = []
    all_labels_val = []
    all_probs_val = [] # Store probabilities

    with torch.no_grad():
        for batch in val_loader:
            if batch is None: continue
            if len(batch) != 2 or batch[0] is None or batch[1] is None:
                 logging.warning(f"Skipping malformed batch in validate: {type(batch)} len {len(batch)}")
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
            all_probs_val.extend(probabilities.cpu().numpy()) # Save probabilities

    if total == 0: return 0.0, 0.0, np.array([]), np.array([]), np.array([]) # Return empty arrays if no validation data
    val_loss = running_loss / total
    val_acc = 100 * correct / total
    return val_loss, val_acc, np.array(all_labels_val), np.array(all_preds_val), np.array(all_probs_val) # Return labels, preds, probs


# --- Weighted F1 Calculation ---
# Competition weights (corresponds to High, Mid, Low, None, Non-hallway)
competition_f1_weights = [2.0, 1.5, 1.0, 1.0, 1.0] # Make this configurable?

def compute_weighted_f1(all_labels: np.ndarray, all_preds: np.ndarray, competition_weights: List[float], num_classes: int) -> float:
    if len(all_labels) == 0 or len(all_preds) == 0:
        print("Warning: No data found for F1 calculation in validation set.")
        return 0.0

    precision, recall, f1_scores, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(num_classes)), zero_division=0
    )
    print(f"  Metrics per class {list(risk_categories.keys())}:")
    print(f"  Precision: {[f'{p:.4f}' for p in precision]}")
    print(f"  Recall:    {[f'{r:.4f}' for r in recall]}")
    print(f"  F1 Scores: {[f'{f:.4f}' for f in f1_scores]}")
    print(f"  Support:   {support}")

    if len(competition_weights) != len(f1_scores):
         logging.error("Length of competition_weights must match the number of classes")
         # Fallback or raise error
         return 0.0 # Or raise ValueError

    weighted_f1_numerator = sum(f * w for f, w in zip(f1_scores, competition_weights))
    weighted_f1_denominator = sum(w for w, s in zip(competition_weights, support) if s > 0) # Only include weights for classes present in ground truth

    if weighted_f1_denominator == 0:
         # Handle case where no relevant classes were present or all F1 scores were zero for present classes
         print("Warning: Weighted F1 denominator is zero (no support or zero F1 for weighted classes). Returning 0.0.")
         return 0.0

    competition_f1 = weighted_f1_numerator / weighted_f1_denominator
    return competition_f1


# --- Optimize Ensemble Weights ---
# Placeholder function - needs saved validation probabilities from best epochs
def optimize_ensemble_weights(probs1: np.ndarray, probs2: np.ndarray, labels: np.ndarray, num_classes: int, competition_weights: List[float]) -> List[float]:
    """
    Optimizes ensemble weights based on validation set performance.
    Requires validation probabilities from the best epoch of each model.
    """
    if probs1 is None or probs2 is None or labels is None or len(labels) == 0:
         print("Warning: Insufficient data for ensemble weight optimization. Using default weights [0.5, 0.5].")
         return [0.5, 0.5]

    print("Attempting to optimize ensemble weights on validation set...")

    def objective(weights):
        w1, w2 = weights
        # Normalize weights to sum to 1
        norm_w1 = abs(w1) / (abs(w1) + abs(w2) + 1e-9)
        norm_w2 = abs(w2) / (abs(w1) + abs(w2) + 1e-9)

        ensembled_probs = norm_w1 * probs1 + norm_w2 * probs2
        final_preds = np.argmax(ensembled_probs, axis=1)

        # Calculate negative competition F1 score (since minimize seeks minimum)
        competition_f1 = compute_weighted_f1(labels, final_preds, competition_weights, num_classes)
        return -competition_f1 # Minimize negative F1

    # Simple grid search (more robust than optimization if objective is noisy)
    best_score = -1.0
    best_weights = [0.5, 0.5]
    for w1_try in np.linspace(0.0, 1.0, 11): # Test weights from 0.0 to 1.0 in 0.1 increments
        w2_try = 1.0 - w1_try
        score = -objective([w1_try, w2_try]) # Get F1 score
        if score > best_score:
            best_score = score
            best_weights = [w1_try, w2_try]

    # Optional: Use scipy.optimize (install scipy first)
    # try:
    #     initial_weights = [0.5, 0.5]
    #     bounds = [(0, 1), (0, 1)] # Weights between 0 and 1
    #     # L-BFGS-B might be sensitive, SLSQP could be an alternative
    #     result = minimize(objective, initial_weights, method='L-BFGS-B', bounds=bounds)
    #     if result.success:
    #         w1, w2 = result.x
    #         norm_w1 = abs(w1) / (abs(w1) + abs(w2) + 1e-9)
    #         norm_w2 = abs(w2) / (abs(w1) + abs(w2) + 1e-9)
    #         best_weights = [norm_w1, norm_w2]
    #         best_score = -result.fun
    #     else:
    #         print("Scipy optimization failed, using grid search result or default.")
    # except ImportError:
    #     print("Scipy not installed. Using grid search for weight optimization.")

    print(f"Optimized weights: {best_weights} with validation F1: {best_score:.4f}")
    return best_weights


# --- Training Loop ---
best_val_score1, best_val_score2 = 0.0, 0.0
trigger_times1, trigger_times2 = 0, 0
best_epoch1, best_epoch2 = 0, 0
best_val_labels1, best_val_preds1, best_val_probs1 = None, None, None
best_val_labels2, best_val_preds2, best_val_probs2 = None, None, None

print("\nStarting training loop...")
for epoch in range(args.epochs):
    print("-" * 20)
    print(f"Epoch {epoch+1}/{args.epochs}")
    current_lrs1 = scheduler1.get_last_lr()
    current_lrs2 = scheduler2.get_last_lr()
    print(f"  LRs M1: {[f'{lr:.1e}' for lr in current_lrs1]} | M2: {[f'{lr:.1e}' for lr in current_lrs2]}")

    # --- Train Models ---
    train_loss1, train_acc1 = train(model1, train_loader, criterion, optimizer1, device, args.mixup_cutmix_prob, args.mixup_alpha, args.cutmix_alpha)
    train_loss2, train_acc2 = train(model2, train_loader, criterion, optimizer2, device, args.mixup_cutmix_prob, args.mixup_alpha, args.cutmix_alpha)
    print(f'[Model 1 ResNet50] Train Loss: {train_loss1:.4f}, Train Acc: {train_acc1:.2f}%')
    print(f'[Model 2 EffNetB3] Train Loss: {train_loss2:.4f}, Train Acc: {train_acc2:.2f}%')

    # --- Validate Models ---
    if val_loader:
        val_loss1, val_acc1, val_labels1, val_preds1, val_probs1 = validate(model1, val_loader, criterion, device)
        val_score1 = compute_weighted_f1(val_labels1, val_preds1, competition_f1_weights, num_classes)
        print(f'[Model 1 ResNet50] Val Loss: {val_loss1:.4f}, Val Acc: {val_acc1:.2f}%, Comp F1: {val_score1:.4f}')

        val_loss2, val_acc2, val_labels2, val_preds2, val_probs2 = validate(model2, val_loader, criterion, device)
        val_score2 = compute_weighted_f1(val_labels2, val_preds2, competition_f1_weights, num_classes)
        print(f'[Model 2 EffNetB3] Val Loss: {val_loss2:.4f}, Val Acc: {val_acc2:.2f}%, Comp F1: {val_score2:.4f}')

        # --- Logging & Checkpointing ---
        wandb.log({
            "epoch": epoch + 1,
            "model1_train_loss": train_loss1, "model1_train_accuracy": train_acc1,
            "model1_val_loss": val_loss1, "model1_val_accuracy": val_acc1, "model1_val_f1_score": val_score1,
            "model1_lr_head": current_lrs1[1] if len(current_lrs1)>1 else current_lrs1[0], # Assuming head is second group
            "model1_lr_backbone": current_lrs1[0] if len(current_lrs1)>1 else current_lrs1[0],
            "model2_train_loss": train_loss2, "model2_train_accuracy": train_acc2,
            "model2_val_loss": val_loss2, "model2_val_accuracy": val_acc2, "model2_val_f1_score": val_score2,
            "model2_lr_head": current_lrs2[1] if len(current_lrs2)>1 else current_lrs2[0],
            "model2_lr_backbone": current_lrs2[0] if len(current_lrs2)>1 else current_lrs2[0],
        })

        # Checkpoint and Early Stopping for Model 1
        if val_score1 > best_val_score1:
            best_val_score1 = val_score1
            torch.save(model1.state_dict(), args.model1_save_path)
            wandb.save(args.model1_save_path) # Save best model to W&B
            trigger_times1 = 0
            best_epoch1 = epoch + 1
            # Save validation results for potential ensemble weight optimization
            best_val_labels1 = val_labels1
            best_val_preds1 = val_preds1
            best_val_probs1 = val_probs1
            print(f"  Model 1 Best Score Updated: {best_val_score1:.4f}")
        else:
            trigger_times1 += 1
            print(f'  Model 1 No Improvement: {trigger_times1}/{args.patience}')

        # Checkpoint and Early Stopping for Model 2
        if val_score2 > best_val_score2:
            best_val_score2 = val_score2
            torch.save(model2.state_dict(), args.model2_save_path)
            wandb.save(args.model2_save_path) # Save best model to W&B
            trigger_times2 = 0
            best_epoch2 = epoch + 1
             # Save validation results for potential ensemble weight optimization
            best_val_labels2 = val_labels2 # Assuming labels are the same for both models in one epoch
            best_val_preds2 = val_preds2
            best_val_probs2 = val_probs2
            print(f"  Model 2 Best Score Updated: {best_val_score2:.4f}")
        else:
            trigger_times2 += 1
            print(f'  Model 2 No Improvement: {trigger_times2}/{args.patience}')

        # Step schedulers (pass score for ReduceLROnPlateau if used after warmup)
        # For CosineAnnealingLR, score is not needed for step()
        scheduler1.step() # Pass val_score1 if scheduler_after_warmup is ReduceLROnPlateau
        scheduler2.step() # Pass val_score2 if scheduler_after_warmup is ReduceLROnPlateau

        # Early stopping condition (stop if *both* models failed to improve for patience epochs)
        if trigger_times1 >= args.patience and trigger_times2 >= args.patience:
            print(f'Early stopping triggered for both models after epoch {epoch+1}.')
            break

    else: # No validation loader, just train and save last model
        # Log training metrics only
        wandb.log({
            "epoch": epoch + 1,
            "model1_train_loss": train_loss1, "model1_train_accuracy": train_acc1,
            "model1_lr_head": current_lrs1[1] if len(current_lrs1)>1 else current_lrs1[0],
            "model1_lr_backbone": current_lrs1[0] if len(current_lrs1)>1 else current_lrs1[0],
            "model2_train_loss": train_loss2, "model2_train_accuracy": train_acc2,
            "model2_lr_head": current_lrs2[1] if len(current_lrs2)>1 else current_lrs2[0],
            "model2_lr_backbone": current_lrs2[0] if len(current_lrs2)>1 else current_lrs2[0],
        })
        # Step schedulers
        scheduler1.step()
        scheduler2.step()
        # Save last epoch models if no validation
        if epoch == args.epochs - 1:
             print("Saving models from last epoch as no validation split was used.")
             torch.save(model1.state_dict(), args.model1_save_path)
             wandb.save(args.model1_save_path)
             torch.save(model2.state_dict(), args.model2_save_path)
             wandb.save(args.model2_save_path)
             best_epoch1, best_epoch2 = args.epochs, args.epochs # Mark last epoch as best
             best_val_score1, best_val_score2 = -1.0, -1.0 # Indicate no validation score


wandb.log({
    "best_model1_val_f1_score": best_val_score1,
    "best_model2_val_f1_score": best_val_score2,
    "best_model1_epoch": best_epoch1,
    "best_model2_epoch": best_epoch2
})

print("="*30)
print(f"Training finished.")
if val_loader:
    print(f"Best Model 1 (ResNet50) score: {best_val_score1:.4f} at epoch {best_epoch1}")
    print(f"Best Model 2 (EfficientNetB3) score: {best_val_score2:.4f} at epoch {best_epoch2}")
else:
    print("Training finished (no validation performed). Using models from last epoch.")
print("="*30)

# --- Load Best Models ---
try:
    model1.load_state_dict(torch.load(args.model1_save_path, map_location=device))
    model2.load_state_dict(torch.load(args.model2_save_path, map_location=device))
    print("Best models loaded successfully.")
except FileNotFoundError:
    print("Warning: Could not find saved best model weights. Using last state models from training.")
    # Models will use weights from the last epoch if files not found


# --- Determine Ensemble Weights ---
if args.optimize_ensemble_weights and best_val_probs1 is not None and best_val_probs2 is not None and best_val_labels1 is not None:
     # Ensure labels from model 1 and 2 best epochs are consistent if epochs differ
     # Using labels from model 1's best epoch here, assuming they are the same validation set
     weights = optimize_ensemble_weights(best_val_probs1, best_val_probs2, best_val_labels1, num_classes, competition_f1_weights)
elif best_val_score1 > 0 or best_val_score2 > 0: # Use validation scores if optimization is off or failed
     total_score = best_val_score1 + best_val_score2
     weights = [best_val_score1 / total_score, best_val_score2 / total_score]
     print(f"Using ensemble weights based on validation F1 scores: {weights}")
else: # Fallback if no validation or zero scores
    weights = [0.5, 0.5]
    print("Using default ensemble weights [0.5, 0.5]")


# --- Test Data Loading ---
test_dataset = FireHazardDataset(
    txt_file=args.test_label_file,
    img_dir=args.test_img_dir,
    transform=val_transform, # Use validation transform for testing
    is_test=True
)
# Use configurable test batch size
test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=torch.cuda.is_available())


# --- Ensemble Prediction (with TTA option) ---
def predict_ensemble_tta(model1: nn.Module, model2: nn.Module, test_loader: DataLoader, device: torch.device, weights: List[float], use_tta: bool, num_classes: int, default_pred_idx: int) -> Dict[str, int]:
    model1.eval()
    model2.eval()
    predictions_map: Dict[str, int] = {} # Store predictions keyed by image name

    # Define TTA transforms
    tta_transforms: List[Any] = []
    if use_tta:
        print("Performing Test Time Augmentation...")
        tta_transforms = [
            lambda x: x, # Original
            lambda x: torch.flip(x, dims=[-1]), # Horizontal flip
            lambda x: torch.flip(x, dims=[-2]), # Vertical flip (Check if appropriate for the domain!)
            # Consider adding rotations, multi-scale etc.
            # lambda x: torch.rot90(x, k=1, dims=[-2,-1]), # 90 deg clockwise
            # lambda x: torch.rot90(x, k=3, dims=[-2,-1]), # 270 deg clockwise
        ]
        print(f"  Using {len(tta_transforms)} TTA transforms.")
    else:
        print("TTA disabled. Using single inference.")
        tta_transforms = [lambda x: x] # Just original image


    with torch.no_grad():
        batch_count = 0
        for batch in test_loader:
            batch_count += 1
            if batch_count % 50 == 0:
                 print(f"  Processing test batch {batch_count}/{len(test_loader)}")

            images, _, names = None, None, None # Initialize
            if batch is None:
                 logging.warning("Skipping empty batch in test_loader.")
                 continue
            if len(batch) == 3:
                 images, _, names = batch # images can be None if loading failed for all in batch
            else:
                 logging.error(f"Unexpected batch format in test_loader: length {len(batch)}")
                 continue

            if names is None or len(names) == 0:
                 logging.warning("Skipping batch with no image names.")
                 continue

            # Process valid images in the batch
            if images is not None:
                images = images.to(device)
                batch_size = images.size(0)
                # Accumulate probabilities over TTA transforms
                batch_tta_probs = torch.zeros(batch_size, num_classes).to(device)

                for transform in tta_transforms:
                    tta_images = transform(images)
                    outputs1 = model1(tta_images)
                    outputs2 = model2(tta_images)
                    # Ensemble probabilities
                    probs1 = torch.softmax(outputs1, dim=1)
                    probs2 = torch.softmax(outputs2, dim=1)
                    ensembled_probs = weights[0] * probs1 + weights[1] * probs2
                    batch_tta_probs += ensembled_probs

                final_probs = batch_tta_probs / len(tta_transforms)
                _, predicted_indices = torch.max(final_probs, 1)
                predicted_indices_list = predicted_indices.cpu().numpy()

            # Map predictions to image names
            valid_pred_idx = 0
            for name in names:
                 if images is not None and valid_pred_idx < len(predicted_indices_list):
                     # Prediction corresponds to a successfully loaded image
                     predictions_map[name] = predicted_indices_list[valid_pred_idx]
                     valid_pred_idx += 1
                 else:
                     # Image loading failed OR entire batch failed
                     logging.warning(f"Assigning default prediction ({default_pred_idx}: {reverse_risk_categories.get(default_pred_idx, 'Unknown')}) for unloaded/failed image: {name}")
                     predictions_map[name] = default_pred_idx # Use configured default


    return predictions_map

# --- Generate Predictions ---
print(f"\nGenerating predictions on the test set {'with TTA' if args.tta else 'without TTA'}...")
predictions_map = predict_ensemble_tta(model1, model2, test_loader, device, weights, args.tta, num_classes, args.default_pred_idx)

# --- Create Submission File ---
print(f"Writing predictions to {args.output_file}...")
# Ensure predictions are in the same order as the original test label file
ordered_predictions = []
missing_predictions = 0
original_test_order = test_dataset.original_image_names # Use stored order

for name in original_test_order:
    pred_index = predictions_map.get(name)
    if pred_index is None:
        # This should ideally not happen if collate_fn handles failed names correctly, but as a safeguard:
        logging.error(f"Prediction missing for image: {name}. Assigning default ({args.default_pred_idx}).")
        pred_index = args.default_pred_idx
        missing_predictions += 1
    ordered_predictions.append(pred_index)

if missing_predictions > 0:
     print(f"Warning: {missing_predictions} predictions were missing and assigned default.")


submission_df = pd.DataFrame({
    'image_name': original_test_order,
    'prediction_index': ordered_predictions
})
submission_df['risk_label'] = submission_df['prediction_index'].map(reverse_risk_categories)

# Ensure the output format is exactly image_name<TAB>risk_label
try:
    with open(args.output_file, 'w', encoding='utf-8') as f:
         for index, row in submission_df.iterrows():
             f.write(f"{row['image_name']}\t{row['risk_label']}\n")
    print(f"Submission file '{args.output_file}' generated successfully with {len(ordered_predictions)} predictions.")
except Exception as e:
    print(f"Error writing submission file: {e}")


# Log test predictions table to WandB
try:
    wandb.log({"test_predictions": wandb.Table(dataframe=submission_df[['image_name', 'risk_label']])})
except Exception as e:
     print(f"Could not log submission table to WandB: {e}") # Handle potential issues if WandB run finished etc.


print("Script finished.")
wandb.finish()