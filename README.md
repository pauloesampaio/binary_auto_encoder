# Autoencoder for binary codes challenge

## How to run it

Install requirements on the `requirements.txt` file (`pip install -r requirements.txt`).

Update `config\config.yml` file following this instructions:

- Update images folder location (with the extension)
- Define your tiling strategy:

    Your original Image will be split into smaller tiles, defined as:
  
  - `tile_side`: Side of your tile that will be used as input for the network
  - `step_size`: distance for the next tile

- Define your network construction parameters:

  - Define your reduced space `latent_dimension`
  - Number of layers on the encoder / decoder should will be defined from the length of the parameters
  - Parameters are:
    - `n_filters`: List with number of units in each layers
    - `kernel_size`: List with kernel size of each layer
    - `strides`: list with strides of each layer
  - Set your loss function ("mean_squared_error" or other you want to experiment)
  - Set your initial learning rate, number of epochs and batch size
  - You can also define the patience of you learning rate reducer (reduces LR when validation loss reaches a plateau) and early stopping (stop training when validation loss stops reducing to avoid overfitting)

Then you should be able to just run `python train_autoencoder.py`. Model and training log will be saved on the locations defined on the configuration file. If there is a GPU available (recommended), tensorflow will be able to get it.

If you want to run it on google colab to use their GPU, is easy:

- Open google colab and change runtime type to GPU
- Clone the repository: `!git clone https://github.com/pauloesampaio/binary_auto_encoder.git`)
- Enter on the directory: `cd binary_auto_encoder`
- Git pull in case there any modifications: `!git pull`
- Run it: `!python train_autoencoder.py`

## Training schedule

Model will train with LR reducer and Early stopping:

- If training validation doesn't reduce for 5 epochs, LR is divided by 10
- Overall, if no improvement after 10 epochs, training stops and best model so far is saved
- Training log is also saved for checking

## Testing

In the beginning of the training script, 20% of the data is stored for final validation.
In this step:

- For each validation modified images, each 6x6 block is binarized and compared to the original images using hamming distance. The mean of these hamming distances is our benchmark.
- After the model is trained, it predicts on these same validation images and once again we binarize every 6x6 block. We then compare again to the original images using hamming distance. The mean of these hamming distances is our model result.
- We compare our model result with the original benchmark.
  - With the parameters currently on the config file, we reduce the average hamming distance from 0.0385 to 0.0084, a reduction of 78.3%.

