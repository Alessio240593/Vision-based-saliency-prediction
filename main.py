import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from dataset.Dataset import SaliencyDataset
from model.UNet import UNet
from model.UnetSkipInceptionSe import InceptionSeUNet

from trainer.Trainer import train, plot_losses, test
import sys

from utility.Utility import separator, validate_and_load_dataset

# --- PATH DEFINITIONS ---
BASE_PATH = Path("data")
IMAGE_BASE = BASE_PATH / "images"
MAP_BASE = BASE_PATH / "maps"

# TRAIN PATHS
IMAGE_TRAIN_PATH = IMAGE_BASE / "train"
MAP_TRAIN_PATH = MAP_BASE / "train"

# VALIDATION PATHS
IMAGE_VAL_PATH = IMAGE_BASE / "val"
MAP_VAL_PATH = MAP_BASE / "val"

# --- ERROR HANDLING ---
if not IMAGE_TRAIN_PATH.exists() or not MAP_TRAIN_PATH.exists():
    raise NotADirectoryError(
        f"Training Path Error: Folders not found!\n"
        f"Expected: {IMAGE_TRAIN_PATH} and {MAP_TRAIN_PATH}"
    )

if not IMAGE_VAL_PATH.exists() or not MAP_VAL_PATH.exists():
    raise NotADirectoryError(
        f"Validation Path Error: Folders not found!\n"
        f"Expected: {IMAGE_VAL_PATH} and {MAP_VAL_PATH}"
    )

# SAVE PATHS
SAVE_PATH = Path("saved_models")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

UNET_CHECKPOINT_PATH = SAVE_PATH / "unet_checkpoint.pth"
UNET_BEST_MODEL_PATH = SAVE_PATH / "unet_model.pth"

INCEPTION_SE_UNET_CHECKPOINT_PATH = SAVE_PATH / "inceptionSeUnet_checkpoint.pth"
INCEPTION_SE_UNET_BEST_MODEL_PATH = SAVE_PATH / "inceptionSeUnet_model.pth"

# --- FINAL CONFIGURATION SUMMARY ---
separator("CONFIGURATION READY")
print(f"Images Train Path: {IMAGE_TRAIN_PATH}")
print(f"Maps Train Path:   {MAP_TRAIN_PATH}")
print(f"Images Val Path:   {IMAGE_VAL_PATH}")
print(f"Maps Val Path:     {MAP_VAL_PATH}")
print(f"Model Save Path:   {SAVE_PATH}")

# PARAM
TARGET_SIZE = (224, 224)
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 1

# ****************************************************************************************#

# DEVICE SETUP
separator("DEVICE SETUP")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    print(f"Device: {device.type.upper()}")
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print(f"Device: CPU (no GPU found)")

# ****************************************************************************************#

# IMAGE NAME TRACK
separator("DATASET CONTROLS")

train_images, train_maps, val_images, val_maps, test_images, test_maps = validate_and_load_dataset(base_path=BASE_PATH,
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

    if test_images and test_maps:
        test_dataset = SaliencyDataset(
            image_files=test_images, map_files=test_maps,
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
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=2, pin_memory=pin)

    print(f"DataLoaders ready!")
    print(f"   Train: {len(train_loader)} batches")
    print(f"   Val:   {len(val_loader)} batches")

    if test_dataset:
        print(f"   Test:   {len(test_loader)} batches")

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

inceptionSeUNet = InceptionSeUNet(in_channels=3, out_channels=1).to(device)

total_params = sum(p.numel() for p in unet.parameters())
trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)

print(f"InceptionSeUNet initialized on {device.type.upper()}")
print(f"   Total parameters:     {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")

# ****************************************************************************************#

# TRAINING
separator("TRAINING")

criterion = nn.MSELoss()
print(f"Defined Loss Function: {type(criterion)}")

# Unet
unetOptimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
print(f"Defined Optimizer fro unet: {type(unetOptimizer)} with initial LR={LEARNING_RATE}")

# InceptionSeUNet
inceptionSeUNetOptimizer = optim.Adam(inceptionSeUNet.parameters(), lr=LEARNING_RATE)
print(f"Defined Optimizer for inceptionSeUNet: {type(inceptionSeUNetOptimizer)} with initial LR={LEARNING_RATE}")

unet_train_history, unet_val_history = train(
    model=unet,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=unetOptimizer,
    device=device,
    num_epochs=NUM_EPOCHS,
    save_path=UNET_BEST_MODEL_PATH,
    checkpoint_path=UNET_CHECKPOINT_PATH,
    useCheckpoint=True
)

inceptionSeUNet_train_history, inceptionSeUNet_val_history = train(
    model=inceptionSeUNet,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=inceptionSeUNetOptimizer,
    device=device,
    num_epochs=NUM_EPOCHS,
    save_path=INCEPTION_SE_UNET_BEST_MODEL_PATH,
    checkpoint_path=INCEPTION_SE_UNET_CHECKPOINT_PATH,
    useCheckpoint=True
)

# ****************************************************************************************#

separator("TESTING")

unet_test_loss, unet_test_iou, unet_test_dice = test(
    model=unet,
    test_loader=test_loader,
    criterion=criterion,
    device=device
)

inceptionSeUNet_test_loss, inceptionSeUNet_test_iou, inceptionSeUNet_test_dice = test(
    model=inceptionSeUNet,
    test_loader=test_loader,
    criterion=criterion,
    device=device
)

# ****************************************************************************************#

separator("TRAINING-VALIDATION LOSS")

plot_losses(
    unet_train_history,
    unet_val_history,
    inceptionSeUNet_train_history,
    inceptionSeUNet_val_history,
    name1="UNet",
    name2="Inception-SE UNet"
)

# ****************************************************************************************#

separator("TEST STATS")

results = pd.DataFrame({
    "Model": ["UNet", "Inception-SE UNet"],

    "Loss": [
        unet_test_loss,
        inceptionSeUNet_test_loss
    ],

    "IoU": [
        unet_test_iou,
        inceptionSeUNet_test_iou
    ],

    "Dice": [
        unet_test_dice,
        inceptionSeUNet_test_dice
    ],

    "% IoU Gain": [
        0.0,
        (inceptionSeUNet_test_iou - unet_test_iou) / unet_test_iou * 100
    ],

    "% Dice Gain": [
        0.0,
        (inceptionSeUNet_test_dice - unet_test_dice) / unet_test_dice * 100
    ],

    "% Loss Reduction": [
        0.0,
        (unet_test_loss - inceptionSeUNet_test_loss) / unet_test_loss * 100
    ]
})

print(results)

models = ["UNet", "Inception-SE UNet"]

loss = [unet_test_loss, inceptionSeUNet_test_loss]
iou = [unet_test_iou, inceptionSeUNet_test_iou]
dice = [unet_test_dice, inceptionSeUNet_test_dice]

x = range(len(models))

plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 3, 1)
plt.bar(x, loss)
plt.xticks(x, models, rotation=15)
plt.title("Test Loss")

# IoU
plt.subplot(1, 3, 2)
plt.bar(x, iou)
plt.xticks(x, models, rotation=15)
plt.title("IoU")

# Dice
plt.subplot(1, 3, 3)
plt.bar(x, dice)
plt.xticks(x, models, rotation=15)
plt.title("Dice")

plt.tight_layout()
plt.show()
