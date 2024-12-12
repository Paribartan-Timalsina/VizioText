from pathlib import Path
from datetime import datetime

# dataset and features
DATASET_NAME = "flickr30k"
FEATURES_NAME = "flickr-31k-features-all"
TAKE_FEATURES_FROM_INPUT = (
    True  # load features from already saved file as infering this takes a long time
)

# paths
WORKING_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input/")
DATASET_INPUT_DIR = INPUT_DIR.joinpath(DATASET_NAME)
TEMP_DIR = Path("/tmp")
PRETRAINED_MODEL_LOCATION = Path('/kaggle/input/1024_batch_size_35_epoch/keras/default/1/models.keras')

# preprocessing
FILTER_NON_ALPHA_NUMERIC_STRINGS = True
NON_ALPHA_NUMERIC_THRESHOLD = (
    3  # the words containing less than this words is discarded
)
# filtering the columns having too few or too many words
# as having too few and too many can improperly skew whole training process
# having too many causes the whole network to be trained mostly on padding rather than the actual data
FILTER_ROWS_HAVING_TOO_FEW_OR_TOO_MANY_WORDS = True
N_WORDS_UPPER_LIMIT = 30
N_WORDS_LOWER_LIMIT = 6

# model definition
TRAIN_CNN = False
NUM_OUTPUT_CAPTIONS = 1
IMAGE_INPUT_SHAPE = (224, 224, 3)  # (height, width, channel)
INPUT_FEATURE_SHAPE = (4096, ) # the output shape of the base model

# model training
LOAD_PRETRAINED = False  # make it possible to continue training from a saved model
EVALUATE_AFTER_TRAIN = False
CONTINUE_FROM_WANDB_RUN = False  # if needed to load pretrained model from wandb
USE_MULTIPLE_GPUS = True

# Hyperparameters
N_EPOCHS = 35
BATCH_SIZE = 1024  # since there are many example of same image, this is kept large
# so that model can see many different images before update
TRAIN_TEST_VAL_SPLIT = (70, 20, 10)

# logging
DEBUG = False
LOGGER_NAME = "Image-Caption-Generator-Logger"
TENSORBOARD_LOG_DIR = WORKING_DIR.joinpath("logs", "scalers", datetime.now().strftime("%Y%m%d-%H%M%S"))
WANDB_PROJECT_NAME = "Image Caption Generator-Multi-GPU Pre-Infer Images"
