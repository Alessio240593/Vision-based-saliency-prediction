import sys


def separator(title=""):
    if title:
        print(f"\n{'─' * 10} {title} {'─' * 10}")
    else:
        print(f"{'─' * 30}")


def validate_and_load_dataset(base_path, train_img_path, train_map_path, val_img_path, val_map_path):
    try:
        if not base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        # Gathering file paths
        train_images = sorted(list(train_img_path.glob("*.jpg")) + list(train_img_path.glob("*.png")))
        train_maps = sorted(list(train_map_path.glob("*.jpg")) + list(train_map_path.glob("*.png")))
        val_images = sorted(list(val_img_path.glob("*.jpg")) + list(val_img_path.glob("*.png")))
        val_maps = sorted(list(val_map_path.glob("*.jpg")) + list(val_map_path.glob("*.png")))

        # Integrity checks
        assert len(train_images) > 0, "Train folder is empty."
        assert len(val_images) > 0, "Validation folder is empty."
        assert len(train_images) == len(
            train_maps), f"Train mismatch: {len(train_images)} images vs {len(train_maps)} maps."
        assert len(val_images) == len(
            val_maps), f"Validation mismatch: {len(val_images)} images vs {len(val_maps)} maps."

        print("Dataset integrity check passed.")
        print(f"   Train: {len(train_images)} pairs")
        print(f"   Val:   {len(val_images)} pairs")

        return train_images, train_maps, val_images, val_maps

    except (FileNotFoundError, AssertionError) as e:
        print(f"Dataset Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


import torch


def load_model(model_class, checkpoint_path, device='cpu', **kwargs):
    model = model_class(**kwargs)

    try:
        state_dict = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(state_dict)

        model.to(device)
        model.eval()

        print(f"Model loaded successfully from {checkpoint_path}")
        return model

    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None