import torch
from tqdm import tqdm
import time
import copy
import matplotlib.pyplot as plt


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    bar = tqdm(loader, desc="  Training", leave=False)
    for images, maps in bar:
        images, maps = images.to(device), maps.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), maps)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    bar = tqdm(loader, desc="  Validation", leave=False)
    with torch.no_grad():
        for images, maps in bar:
            images, maps = images.to(device), maps.to(device)
            loss = criterion(model(images), maps)
            running_loss += loss.item() * images.size(0)
            bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader.dataset)


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, save_path):
    print("--- Starting Training ---")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    start_time = time.time()

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}\n" + '-' * 10)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            print(f"  Val loss improved ({best_val_loss:.4f} → {val_loss:.4f}), saving model...")
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), save_path)

    training_time = time.time() - start_time
    print(f"\nTraining complete in {training_time // 60:.0f}m {training_time % 60:.0f}s")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_wts)
    print("Loaded best model weights.")

    return train_loss_history, val_loss_history


def plot_losses(train_history, val_history):
    epochs = range(1, len(train_history) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_history, label='Train Loss')
    plt.plot(epochs, val_history, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
