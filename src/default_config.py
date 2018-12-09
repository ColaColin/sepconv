#
# KTH Royal Institute of Technology
#

#
# This file is for reference only and should *not* me modified.
# To change the configuration, create a new config.py file
# and import this one in it as 'from src.default_config import *'
#

# The size of the input images to be fed to the network during training.
CROP_SIZE: int = 128

# The size of the patches to be extracted from the datasets
PATCH_SIZE = (150, 150)

# Whether or not we should store the patches produced by the data manager
CACHE_PATCHES: bool = False

# Number of epochs used for training
EPOCHS: int = 10

# Kernel size of the custom Separable Convolution layer. Cannot be changed without changing the cuda kernel code
OUTPUT_1D_KERNEL_SIZE: int = 51

# The batch size used for mini batch gradient descent
BATCH_SIZE: int = 100

# Upper limit on the number of samples used for training
MAX_TRAINING_SAMPLES: int = 500_000

# Upper limit on the number of samples used for validation
MAX_VALIDATION_SAMPLES: int = 100

# Number of workers of the torch.utils.data.DataLoader AND of the data manager
# Set this to 0 to work on the main thread
NUM_WORKERS: int = 0

# Random seed fed to torch
SEED: int = None

# "DAVIS" or "custom". Custom searches all direct subdirectories of the given directory for images, names as <framenumber>.<jpg|jpeg|png>, each subdirectory should start counting from 0, writes the patches.json for them and might cache them in a cache directory for faster access
# custom does not yet support validation or visual test sets, that should change.... #TODO
DATASET = "DAVIS"

# length of image sequences captures, specifies number of images to be generated in the middle of the first and the last image
# the dataset needs to be generated with a length bigger or equal to the one used when training the network
# must be a power of 2 plus 1, at least 3. Reasonable for small patches is probably at most 9...
# != 3 is only supported for custom dataset and FORCE_HORIZONTAL = False
SEQUENCE_LENGTH = 3

# sequence length as supported by the dataset
MAX_SEQUENCE_LENGTH = 3

# number of frames between frames packed together as tuples
CUSTOM_STRIDE = 2

# number of frames skipped after a set of frames used for a tuple
# only supported for custom dataset
CUSTOM_FRAMESKIP = 90

FLOW_THRESHOLD: float = 25

# rotate the patches such that the main movement, according to the optical flow, is horizontal
FORCE_HORIZONTAL = False

# Path to the dataset directory
DATASET_DIR = './dataset'

# Force torch to run on CPU even if CUDA is available
ALWAYS_CPU: bool = False

# Path to the outout directory where the model checkpoins should be stored
OUTPUT_DIR: str = './out'

# Whether or not the model parameters should be written to disk at each epoch
SAVE_CHECKPOINS: bool = False

# Force model to use the slow Separable Convolution implementation even if CUDA is available
ALWAYS_SLOW_SEP_CONV: bool = False

# Whether or not we should run the validation set on the model at each epoch
VALIDATION_ENABLED: bool = False

# Whether or not we should run the visual test set on the model at each epoch
VISUAL_TEST_ENABLED: bool = False

# Whether or not the data should be augmented with random transformations
AUGMENT_DATA: bool = True

# Probability of performing the random temporal order swap of the two input frames
RANDOM_TEMPORAL_ORDER_SWAP_PROB: float = 0.5

START_FROM_EPOCH = 1

# Start from pre-trained model (path)
START_FROM_EXISTING_MODEL = None

# One of {"l1", "vgg", "ssim"}
LOSS: str = "l1"

VGG_FACTOR: float = 1.0

GENERATE_PARALLAX_VIEW: bool = False
PARALLAX_VALIDATION: bool = False

PARALLAX_VIEW_T: int = 193
PARALLAX_VIEW_CAM_INTERVAL: int = 16
PARALLAX_OUTPUT_DIR = "./video/icme/parallax_d3_c2/"
PARALLAX_DATASET_DIR = "./video/icme/Development_dataset_3"