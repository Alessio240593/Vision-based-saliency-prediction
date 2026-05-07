import sys

from sklearn.model_selection import train_test_split


def separator(title=""):
    if title:
        print(f"\n{'─' * 10} {title} {'─' * 10}")
    else:
        print(f"{'─' * 30}")

def validate_and_load_dataset(base_path, train_img_path, train_map_path, val_img_path, val_map_path, create_test=True, test_size=0.5):
    try:
        if not base_path.exists():
            raise FileNotFoundError(f"Base path does not exist: {base_path}")

        # LOAD PATHS
        train_images = sorted(list(train_img_path.glob("*.jpg")) + list(train_img_path.glob("*.png")))
        train_maps = sorted(list(train_map_path.glob("*.jpg")) + list(train_map_path.glob("*.png")))

        val_images = sorted(list(val_img_path.glob("*.jpg")) + list(val_img_path.glob("*.png")))
        val_maps = sorted(list(val_map_path.glob("*.jpg")) + list(val_map_path.glob("*.png")))

        # CHECKS
        assert len(train_images) == len(train_maps), "Train mismatch images/maps"
        assert len(val_images) == len(val_maps), "Val mismatch images/maps"

        print("Dataset integrity check passed.")
        print(f"Train: {len(train_images)} pairs")
        print(f"Val:   {len(val_images)} pairs")

        if create_test:
            val_data = list(zip(val_images, val_maps))

            val_data, test_data = train_test_split(
                val_data,
                test_size=test_size,
                random_state=42
            )

            val_images, val_maps = zip(*val_data)
            test_images, test_maps = zip(*test_data)

            val_images, val_maps = list(val_images), list(val_maps)
            test_images, test_maps = list(test_images), list(test_maps)

            print(f"Test:  {len(test_images)} pairs")

            return train_images, train_maps, val_images, val_maps, test_images, test_maps

        return train_images, train_maps, val_images, val_maps

    except (FileNotFoundError, AssertionError) as e:
        print(f"Dataset Error: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)