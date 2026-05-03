import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from dataset.Dataset import SaliencyDataset
from model.UNet import UNet

from trainer.Trainer import train, plot_losses
import sys

from utility.Utility import separator, validate_and_load_dataset

# BASE PATHS
BASE_PATH = Path("data")
IMAGE_DIR = BASE_PATH / "images"
MAP_DIR = BASE_PATH / "maps"

# TRAIN PATHS
IMAGE_TRAIN_PATH = IMAGE_DIR / "train"
MAP_TRAIN_PATH = MAP_DIR / "train"

# VALIDATION PATHS
IMAGE_VAL_PATH = IMAGE_DIR / "val"
MAP_VAL_PATH = MAP_DIR / "val"

# SAVE
SAVE_DIR = Path("saved_models")
SAVE_DIR.mkdir(exist_ok=True)
SAVE_PATH = SAVE_DIR / "unet_best.pth"

# PARAM
TARGET_SIZE = (224, 224)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1

# DEVICE SETUP
separator("DEVICE SETUP")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(f"Device: {device.type.upper()}")
    print(f"   GPU:  {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"Device: CPU (no GPU found)")

# ****************************************************************************************#

# IMAGE NAME TRACK
separator("DATASET CONTROLS")

train_images, train_maps, val_images, val_maps = validate_and_load_dataset(base_path=BASE_PATH,
                                                                           train_img_path=IMAGE_TRAIN_PATH,
                                                                           train_map_path=MAP_TRAIN_PATH,
                                                                           val_img_path=IMAGE_VAL_PATH,
                                                                           val_map_path=MAP_VAL_PATH)

# ****************************************************************************************#

# DATASET INIT
separator("DATASET INIT")

to_tensor = transforms.Compose([transforms.ToTensor()])
train_dataset = None
val_dataset = None

try:
    train_dataset = SaliencyDataset(
        image_files=train_images, map_files=train_maps,
        target_size=TARGET_SIZE, is_train=True,
        transform=to_tensor, map_transform=to_tensor
    )
    val_dataset = SaliencyDataset(
        image_files=val_images, map_files=val_maps,
        target_size=TARGET_SIZE, is_train=False,
        transform=to_tensor, map_transform=to_tensor
    )
    print(f"Datasets ready!")
    print(f"   {train_dataset}")
    print(f"   {val_dataset}")

except Exception as e:
    print(f"Dataset error: {e}")
    sys.exit(1)

# ****************************************************************************************#

# DATA LOADER INIT
separator("DATALOADER INIT")

pin = device.type == "cuda"
train_loader = None
val_loader = None

try:
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=pin)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=pin)

    print(f"DataLoaders ready!")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Val:   {len(val_loader)} batches")

except Exception as e:
    print(f"DataLoader error: {e}")
    sys.exit(1)

# ****************************************************************************************#

separator("NETWORK INITIALIZATION")

unet = UNet(in_channels=3, out_channels=1).to(device)

total_params = sum(p.numel() for p in unet.parameters())
trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

print(f"UNet initialized on {device.type.upper()}")
print(f"   Total parameters:     {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# ****************************************************************************************#

# TRAINING
separator("TRAINING")

criterion = nn.MSELoss()
print(f"Defined Loss Function: {type(criterion)}")

optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
print(f"Defined Optimizer: {type(optimizer)} with initial LR={LEARNING_RATE}")

train_history, val_history = train(
    model=unet,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    num_epochs=NUM_EPOCHS,
    save_path=SAVE_PATH
)

plot_losses(train_history, val_history)
