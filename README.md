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
