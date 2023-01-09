# import the necessary packages
import torch
import os

# define the path to the images and masks dataset
IMAGE_DATASET_PATH = "/workspace/FinalProject/dataset/train/"
MASK_DATASET_PATH = "/workspace/FinalProject/dataset/"


# determine the device to be used for training and evaluation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# determine if we will be pinning memory during data loading
PIN_MEMORY = True if DEVICE == "cuda" else False

# initialize learning rate, number of epochs to train for, and the
# batch size
INIT_LR = 0.0001
NUM_EPOCHS = 30
BATCH_SIZE = 8

# define the input image dimensions
INPUT_IMAGE_WIDTH = 384
INPUT_IMAGE_HEIGHT = 384

# define threshold to filter weak predictions
THRESHOLD = 0.5

# define the path to the base output directory
BASE_OUTPUT = "output"

# define the path to the output serialized model, model training
# plot, and testing image paths
MODEL_PATH_A = os.path.join(BASE_OUTPUT, "unet_tgs_salt_A_30_with_dice.pth")
MODEL_PATH_B = os.path.join(BASE_OUTPUT, "unet_tgs_salt_B_oh_30_dc.pth")
MODEL_PATH_C = os.path.join(BASE_OUTPUT, "unet_tgs_salt_C_oh_30_dc.pth")
MODEL_PATH_D = os.path.join(BASE_OUTPUT, "unet_tgs_salt_D_oh_30_dc.pth")
MODEL_PATH_E = os.path.join(BASE_OUTPUT, "unet_tgs_salt_E_oh_30_dc.pth")
PLOT_PATH_A = os.path.sep.join([BASE_OUTPUT, "plot_TopKA_30_dc.png"])
PLOT_PATH_B = os.path.sep.join([BASE_OUTPUT, "plot_TopKB_oh_30_dc.png"])
PLOT_PATH_C = os.path.sep.join([BASE_OUTPUT, "plot_TopKC_oh_30_dc.png"])
PLOT_PATH_D = os.path.sep.join([BASE_OUTPUT, "plot_TopKD_oh_30_dc.png"])
PLOT_PATH_E = os.path.sep.join([BASE_OUTPUT, "plot_TopKE_oh_30_dc.png"])
TEST_PATHS = os.path.sep.join([BASE_OUTPUT, "test_paths.txt"])