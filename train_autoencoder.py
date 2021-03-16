import os
import numpy as np
from utils.io_utils import yaml_loader, load_dataset, generate_tiles
from utils.model_utils import build_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

config = yaml_loader("./config/config.yml")

config["save_model_path"]

print("Loading datasets")
X, y = load_dataset(config["modified_image_folder"], config["original_image_folder"])

print("Tiling images")
X_tiles = []
for each_image in X:
    current_tiles = generate_tiles(each_image, config["tile_side"], config["step_size"])
    X_tiles = X_tiles + current_tiles
X_tiles = np.array(X_tiles)

y_tiles = []
for each_image in y:
    current_tiles = generate_tiles(each_image, config["tile_side"], config["step_size"])
    y_tiles = y_tiles + current_tiles
y_tiles = np.array(y_tiles)

X_tiles = X_tiles.reshape(X_tiles.shape + (1,))
y_tiles = y_tiles.reshape(y_tiles.shape + (1,))

augmentator = ImageDataGenerator(
    horizontal_flip=True, vertical_flip=True, validation_split=0.2
)
train_generator = augmentator.flow(X_tiles, y_tiles, seed=12345, subset="training")
test_generator = augmentator.flow(X_tiles, y_tiles, seed=12345, subset="validation")

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
lr_reducer = ReduceLROnPlateau(
    patience=config["lr_reducer_patience"],
    factor=0.1,
    min_lr=config["minimum_learning_rate"],
    verbose=2,
)

csv_dir = os.path.dirname(config["model_log_path"])
if not os.path.exists(csv_dir):
    os.mkdir(csv_dir)

model_dir = os.path.dirname(config["save_model_path"])
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

csv_logger = CSVLogger(config["model_log_path"])

print("Training model")
a.fit(
    train_generator,
    validation_data=test_generator,
    epochs=config["max_epochs"],
    batch_size=config["batch_size"],
    shuffle=True,
    callbacks=[early_stopping, lr_reducer, csv_logger],
)

print(f'Saving model to {config["model_path"]}')
a.save(config["save_model_path"])
