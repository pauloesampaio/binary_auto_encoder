import yaml
import glob
from PIL import Image
import numpy as np


def yaml_loader(yaml_path):
    """Loads yaml from a path

    Args:
        yaml_path (str): Path to yaml file

    Returns:
        dict: Dict with yaml content
    """
    with open(yaml_path, "r") as f:
        loaded_yaml = yaml.safe_load(f)
    return loaded_yaml


def _load_images(image_paths):
    images = []
    for fpath in image_paths:
        images.append(np.array(Image.open(fpath)))
    return np.array(images) / 255.0


def load_dataset(modified_folder, original_folder):
    modified_paths = sorted(glob.glob(modified_folder))
    original_paths = sorted(glob.glob(original_folder))
    for file_combo in zip(modified_paths, original_paths):
        if file_combo[0].split("/")[-1] != file_combo[1].split("/")[-1]:
            print("Different file form modified and original")
            break
    X = _load_images(modified_paths)
    y = _load_images(original_paths)
    return X, y


def generate_tiles(image, tile_side=24, step_size=12):
    tiles = []
    image_side = image.shape[0]
    slices = range(0, image_side - tile_side + 1, step_size)
    for i in slices:
        for j in slices:
            tiles.append(image[i : i + tile_side, j : j + tile_side])
    return tiles


def restore_tiles(tiles):
    tile_side = int(tiles[0].shape[0])
    image_side = int(np.sqrt(len(tiles)) * tile_side)
    slices = list(range(0, image_side, tile_side))
    image_output = np.empty((image_side, image_side), dtype=float)
    tile_counter = 0
    for i in slices:
        for j in slices:
            image_output[i : i + tile_side, j : j + tile_side] = tiles[tile_counter]
            tile_counter = tile_counter + 1
    return image_output
