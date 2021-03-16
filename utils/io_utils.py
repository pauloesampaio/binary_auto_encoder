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
    """Loads all images from list of paths

    Args:
        image_paths (list): List with pathnames

    Returns:
        np.array: Array with all images
    """
    images = []
    for fpath in image_paths:
        images.append(np.array(Image.open(fpath)))
    return np.array(images) / 255.0


def load_dataset(modified_folder, original_folder):
    """Receiving two glob instructions, get all pathnames and
    use the _load_images helper to load all images from them.

    Args:
        modified_folder (str): Path to modified image folder
        original_folder (str): Path to original image folder

    Returns:
        np.array, np.array: Arrays with modified images and original images
    """
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
    """Receives an image and generate square tiles of a given side size and
    with a given displacement from each other.

    Args:
        image (np.array): Image array
        tile_side (int, optional): Size of the tile sides. Defaults to 24.
        step_size (int, optional): [description]. Displacement for each tile. Defaults to 12.

    Returns:
        [type]: [description]
    """
    tiles = []
    image_side = image.shape[0]
    slices = range(0, image_side - tile_side + 1, step_size)
    for i in slices:
        for j in slices:
            tiles.append(image[i : i + tile_side, j : j + tile_side])
    return tiles


def restore_tiles(tiles):
    """From a list of tiles, reconstruct the images.
    Works only for images tiled with side equals to the displacement.

    Args:
        tiles (list): List of tiles

    Returns:
        np.array: Reconstructed image
    """
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


def hamming_distance(original_image, restored_image):
    """Given two binary images, calculates the hamming distance.

    Args:
        original_image (np.array): Image array (binary)
        restored_image (np.array): Image array (binary)

    Returns:
        [float]: Hamming distance
    """
    return np.count_nonzero(original_image != restored_image) / len(
        original_image.reshape(
            -1,
        )
    )


def binarize_tiles(tiles, threshold=0.5):
    """Given a list of tiles of floats, apply thresholding to pass them to binaries

    Args:
        tiles (list): list with arrays of 0-1 floats
        threshold (float, optional): Threshold. Defaults to 0.5.

    Returns:
        np.array: Binary array
    """
    for tile in tiles:
        if tile.mean() > threshold:
            tile[:, :] = 1
        else:
            tile[:, :] = 0
    return tiles
