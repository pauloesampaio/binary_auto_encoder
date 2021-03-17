import os
import numpy as np
from utils.io_utils import (
    yaml_loader,
    load_dataset,
    generate_tiles,
    binarize_tiles,
    restore_tiles,
    hamming_distance,
)
from utils.model_utils import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split

config = yaml_loader("./config/config.yml")

config["save_model_path"]

print("Loading datasets")
X, y = load_dataset(config["modified_image_folder"], config["original_image_folder"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12345
)


print("Tiling images")
X_tiles = []
for each_image in X_train:
    current_tiles = generate_tiles(each_image, config["tile_side"], config["step_size"])
    X_tiles = X_tiles + current_tiles
X_tiles = np.array(X_tiles)

y_tiles = []
for each_image in y_train:
    current_tiles = generate_tiles(each_image, config["tile_side"], config["step_size"])
    y_tiles = y_tiles + current_tiles
y_tiles = np.array(y_tiles)

# Reshaping to keras format
X_tiles = X_tiles.reshape(X_tiles.shape + (1,))
y_tiles = y_tiles.reshape(y_tiles.shape + (1,))

image_generator = ImageDataGenerator(validation_split=0.2)

train_generator = image_generator.flow(X_tiles, y_tiles, seed=12345, subset="training")
validation_generator = image_generator.flow(
    X_tiles, y_tiles, seed=12345, subset="validation"
)

print("Building model")
e, d, a = build_model(
    config["tile_side"],
    config["latent_dimension"],
    config["n_filters"],
    config["kernel_size"],
    config["strides"],
)

a.compile(optimizer=Adam(config["initial_learning_rate"]), loss=config["loss"])

early_stopping = EarlyStopping(
    patience=config["early_stopping_patience"], verbose=2, restore_best_weights=True
)

# If CSV and model directories don't exit, create them
csv_dir = os.path.dirname(config["model_log_path"])
if not os.path.exists(csv_dir):
    os.mkdir(csv_dir)

model_dir = os.path.dirname(config["save_model_path"])
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

# CSV logger to save loss evolution by epoch
csv_logger = CSVLogger(config["model_log_path"])

print("Training model")
a.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=config["max_epochs"],
    batch_size=config["batch_size"],
    shuffle=True,
    callbacks=[early_stopping, csv_logger],
)

print(f'Saving model to {config["save_model_path"]}')
a.save(config["save_model_path"])

print("Testing model")
print("Binarizing modified images from test set")
binarized_modified_images = []
for image in X_test.copy():
    tiles = generate_tiles(
        image, config["binary_code_size"], config["binary_code_size"]
    )
    binarized_tiles = binarize_tiles(tiles, config["binary_level_threshold"])
    binarized_image = restore_tiles(binarized_tiles)
    binarized_modified_images.append(binarized_image)

print("Calculating hamming distance for test set")
benchmark_hamming_distances = []
for test_image, original_image in zip(binarized_modified_images, y_test):
    benchmark_hamming_distances.append(hamming_distance(test_image, original_image))

print("Generating model predictions for test set")
predicted_images = []
for image in X_test.copy():
    tiles = np.array(generate_tiles(image, config["tile_side"], config["tile_side"]))
    predicted_tiles = a.predict(tiles.reshape(tiles.shape + (1,)))
    predicted_image = restore_tiles(predicted_tiles[:, :, :, 0])
    predicted_images.append(predicted_image)

print("Binarizing predicted images")
predicted_binary_images = []
for image in predicted_images.copy():
    tiles = generate_tiles(
        image, config["binary_code_size"], config["binary_code_size"]
    )
    binarized_tiles = binarize_tiles(tiles, config["binary_level_threshold"])
    binarized_image = restore_tiles(binarized_tiles)
    predicted_binary_images.append(binarized_image)

print("Calculating hamming distance for predicted images")
model_hamming_distances = []
for predicted_image, original_image in zip(predicted_binary_images, y_test):
    model_hamming_distances.append(hamming_distance(predicted_image, original_image))

print("Final metrics:")
print(f"Original hamming distance: {np.mean(benchmark_hamming_distances)}")
print(f"Model hamming distance: {np.mean(model_hamming_distances)}")