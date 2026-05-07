import torch
from tqdm import tqdm
import time
import copy
import matplotlib.pyplot as plt

def compute_metrics(outputs, targets, threshold=0.5):
    preds = (outputs > threshold).float()

    intersection = (preds * targets).sum(dim=(1,2,3))
    union = (preds + targets - preds * targets).sum(dim=(1,2,3))

    iou = intersection / (union + 1e-8)
    dice = (2 * intersection) / (preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + 1e-8)

    return iou.mean().item(), dice.mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    bar = tqdm(loader, desc="  Training", leave=False)

    for images, maps in bar:
        images, maps = images.to(device), maps.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, maps)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    iou_total = 0.0
    dice_total = 0.0

    with torch.no_grad():
        for images, maps in loader:
            images, maps = images.to(device), maps.to(device)

            outputs = model(images)
            loss = criterion(outputs, maps)

            running_loss += loss.item() * images.size(0)

            iou, dice = compute_metrics(outputs, maps)
            iou_total += iou * images.size(0)
            dice_total += dice * images.size(0)

    n = len(loader.dataset)

    return running_loss / n, iou_total / n, dice_total / n

def plot_comparison(images, targets, outputs):
    img = images[0].cpu().permute(1, 2, 0).numpy()

    gt = targets[0].cpu().squeeze().numpy()

    pred = outputs[0].cpu().squeeze().numpy()

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title("Immagine Input")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gt, cmap='jet')
    plt.title("Ground Truth (Target)")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(pred, cmap='jet')
    plt.title("Model prediction")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def test(model, test_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    iou_total = 0.0
    dice_total = 0.0

    last_images = None
    last_maps = None
    last_outputs = None

    bar = tqdm(test_loader, desc="Testing", leave=False)

    with torch.no_grad():
        for images, maps in bar:
            images, maps = images.to(device), maps.to(device)

            outputs = model(images)
            loss = criterion(outputs, maps)

            running_loss += loss.item() * images.size(0)

            iou, dice = compute_metrics(outputs, maps)

            iou_total += iou * images.size(0)
            dice_total += dice * images.size(0)

            bar.set_postfix(
                loss=f"{loss.item():.4f}",
                iou=f"{iou:.3f}",
                dice=f"{dice:.3f}"
            )

            last_images = images
            last_maps = maps
            last_outputs = outputs

    n = len(test_loader.dataset)
    test_loss = running_loss / n
    test_iou = iou_total / n
    test_dice = dice_total / n

    if last_images is not None:
        print("\nQualitative comparison:")
        plot_comparison(last_images, last_maps, last_outputs)

    return test_loss, test_iou, test_dice


def load_checkpoint(model, optimizer, checkpoint_path, device, useCheckpoint=True):

    if checkpoint_path.exists() and useCheckpoint:
        print(f"\nLoading checkpoint from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]

        train_loss_history = checkpoint["train_loss_history"]
        val_loss_history = checkpoint["val_loss_history"]

        patience_counter = checkpoint.get("patience_counter", 0)

        print(f"Resuming from epoch {start_epoch}")
        print(f"Best val loss: {best_val_loss:.4f}")

        return start_epoch, best_val_loss, train_loss_history, val_loss_history, patience_counter

    else:
        print("\nNo checkpoint found, starting from scratch.")
        return 0, float("inf"), [], [], 0

def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    save_path,
    checkpoint_path,
    patience=10,
    useCheckpoint=True
):

    start_epoch, best_val_loss, train_loss_history, val_loss_history, patience_counter = load_checkpoint(
        model, optimizer, checkpoint_path, device, useCheckpoint
    )

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, num_epochs):

        print(f"\nEpoch {epoch+1}/{num_epochs}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_iou, val_dice = validate(model, val_loader, criterion, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"IoU: {val_iou:.4f} | Dice: {val_dice:.4f}")

        # BEST MODEL
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, save_path)
            patience_counter = 0
        else:
            patience_counter += 1

        # CHECKPOINT
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "train_loss_history": train_loss_history,
            "val_loss_history": val_loss_history,
            "patience_counter": patience_counter
        }, checkpoint_path)

        # EARLY STOPPING
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model_wts)

    return train_loss_history, val_loss_history

def plot_losses(train1, val1, train2=None, val2=None, name1="Model 1", name2="Model 2"):
    epochs = range(1, len(train1) + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(epochs, train1, label=f"{name1} Train")
    plt.plot(epochs, val1, label=f"{name1} Val")

    if train2 is not None:
        plt.plot(epochs, train2, label=f"{name2} Train")
        plt.plot(epochs, val2, label=f"{name2} Val")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Comparison")

    plt.legend()
    plt.grid(True)
    plt.show()
