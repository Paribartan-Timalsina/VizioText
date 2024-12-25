import gc
import os
import pickle
from contextlib import nullcontext

import keras
import pandas as pd
import tensorflow as tf
import wandb
from _logging import get_logger
from data_generator import data_generator
from IPython import display
from keras import backend as k
from keras.callbacks import Callback
from keras.models import load_model
from main_model import get_image_captioning_model
from parameters import *
from pre_infer_images import extract_image_features_and_save, get_base_model
from preprocessing import filter_based_on_special_characters, filter_based_on_word_count
from vocab_handler import VocabHandler
from wandb.integration.keras import (
    WandbCallback,
    WandbMetricsLogger,
    WandbModelCheckpoint,
)

from sklearn.model_selection import train_test_split
from eval import (
    test_on_image_having_feature,
    test_on_new_image,
    get_blue_score,
    get_all_blue_scores,
    get_predictions_parallelized
)
from tqdm import tqdm

logger = get_logger()

# dataset preparation
images_dir = DATASET_INPUT_DIR.joinpath("Images")
captions_file = DATASET_INPUT_DIR.joinpath("captions.txt")

if TAKE_FEATURES_FROM_INPUT:
    features_dir = INPUT_DIR.joinpath(FEATURES_NAME)
else:
    features_dir = WORKING_DIR.joinpath("features")

    features_dir.mkdir(exist_ok=True)


# reading the data
captions_data = pd.read_csv(captions_file)
captions_data.astype(str)
captions_data.dropna(inplace=True)
logger.info(f"Data Sample: \n{captions_data.head()}")


# filtering based on the word count
if FILTER_ROWS_HAVING_TOO_FEW_OR_TOO_MANY_WORDS:
    logger.info("Filtering based on Word Count.")
    captions_data = filter_based_on_word_count(
        captions_data, N_WORDS_UPPER_LIMIT, N_WORDS_LOWER_LIMIT
    )


# filtering based on special characters
if FILTER_NON_ALPHA_NUMERIC_STRINGS:
    logger.info("Filtering based on special characters.")
    unfiltered_vocabulary = list(
        set(" ".join(captions_data["caption"].to_list()).lower().split())
    )
    captions_data, vocabulary_from_filtered_captions_data, removed_items = (
        filter_based_on_special_characters(captions_data, NON_ALPHA_NUMERIC_THRESHOLD)
    )

    for x in vocabulary_from_filtered_captions_data:
        if x not in unfiltered_vocabulary:
            logger.error(f"Word: '{x}' not in vocabulary")
            raise Exception("Found a word that is not in the vocabulary")

    vocabulary = vocabulary_from_filtered_captions_data
    logger.info(f"Removed items: {removed_items}")
    logger.info(f"Total unique words: {len(vocabulary)}")


# creating the vocab handler
default_vocab_handler = VocabHandler(vocabulary)

logger.info(f"Vocab Size: {default_vocab_handler.vocab_size}")
logger.info(f"Id of young is {default_vocab_handler.id_of('young')}")
logger.info(
    f"The word corresponding to id 12414 {default_vocab_handler.word_of(12414)}"
)
logger.info(
    f"Id of {default_vocab_handler.stop_word} is {default_vocab_handler.id_of(default_vocab_handler.stop_word)}"
)

# saving the vocab handler for later use
logger.info("")
VOCAB_HANDLER_SAVE_PATH = WORKING_DIR.joinpath("vocab-handler")
VOCAB_HANDLER_SAVE_PATH.mkdir(exist_ok=True)
default_vocab_handler.save(VOCAB_HANDLER_SAVE_PATH)

default_vocab_handler.load(VOCAB_HANDLER_SAVE_PATH)

logger.info(f"Vocab Size: {default_vocab_handler.vocab_size}")
logger.info(f"Id of young is {default_vocab_handler.id_of('young')}")
logger.info(
    f"The word corresponding to id 12414 {default_vocab_handler.word_of(12414)}"
)

logger.info(
    f"Id of {default_vocab_handler.stop_word} is {default_vocab_handler.id_of(default_vocab_handler.stop_word)}"
)

# loading the vocab handler from saved file (trial)
default_vocab_handler.load(VOCAB_HANDLER_SAVE_PATH)

logger.info(f"Vocab Size: {default_vocab_handler.vocab_size}")
logger.info(f"Id of young is {default_vocab_handler.id_of('young')}")
logger.info(
    f"The word corresponding to id 12414 {default_vocab_handler.word_of(12414)}"
)

logger.info(
    f"Id of {default_vocab_handler.stop_word} is {default_vocab_handler.id_of(default_vocab_handler.stop_word)}"
)

# getting the maximum length of the words in a caption
# this is important for padding the input as to provide equal length text input
maximum_length = max(
    captions_data["caption"].apply(lambda caption: len(caption.split()))
)
logger.info(f"The maximum number of words is {maximum_length}")

absolute_max_length = maximum_length + 2  # including start and stop words


# Pre-infer Images

strategy = None
if USE_MULTIPLE_GPUS:
    logger.info("Using multiple GPUs with mirrored strategy.")
    strategy = tf.distribute.MirroredStrategy()
    logger.info("Number of devices: {}".format(strategy.num_replicas_in_sync))

with strategy.scope() if USE_MULTIPLE_GPUS else nullcontext():
    base_model = get_base_model()

    feature_shape = base_model.output.shape[1:]  # discarding None from shape
    logger.info(f"Feature shape: {feature_shape}")
    display(base_model.summary())

images = captions_data["image"].unique().tolist()

if TAKE_FEATURES_FROM_INPUT:
    with open(features_dir.joinpath("features.pkl"), "rb") as f:
        features = pickle.load(f)
else:
    features = extract_image_features_and_save(
        images,
        images_dir,
        features_dir,
        base_model,
        save_to_file=False,
        strategy=strategy,
    )


# Model Definition
model = get_image_captioning_model(
    default_vocab_handler.vocab_size, absolute_max_length, feature_shape
)

# Open a strategy scope if needed.
with strategy.scope() if USE_MULTIPLE_GPUS else nullcontext():

    if LOAD_PRETRAINED:
        logger.info("Loading the pretrained model.")
        model: keras.Model = load_model(PRETRAINED_MODEL_LOCATION)
    else:
        model = get_image_captioning_model()

        # compiling
        model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )


# Model training
callbacks = []
# tensorboard logging
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=TENSORBOARD_LOG_DIR)
callbacks.append(tensorboard_callback)

# wandb logging
wandb.login(key=os.environ["WANDB_API_KEY"])

# Initialize a new W&B run
run = wandb.init(config={"bs": 12}, project=WANDB_PROJECT_NAME)

metric_logger_callback = WandbMetricsLogger(log_freq="batch")
model_save_callback = WandbModelCheckpoint(filepath="model.keras", save_freq="epoch")
callbacks.append(model_save_callback)

wandb.save(str(VOCAB_HANDLER_SAVE_PATH) + "/*", base_path=str(WORKING_DIR))

# reuse old model and the same tokenizer (vocab_handler)
if CONTINUE_FROM_WANDB_RUN:
    entity = os.environ["WANDB_ENTITY"]
    project = WANDB_PROJECT_NAME
    alias = "latest"  # semantic nickname or identifier for the model version
    model_artifact_name = "run_cq1wzywj_model"

    # Access and download model. Returns path to downloaded artifact

    downloaded_model_path = run.use_model(
        name=f"{entity}/{project}/{model_artifact_name}:{alias}"
    )
    run_path = f"/{entity}/{project}/{model_artifact_name[4:-6]}"
    id_to_word_dict_save_path = wandb.restore(
        "vocab-handler/id-to-word-dict.pkl", run_path=run_path
    )
    word_to_id_dict_save_path = wandb.restore(
        "vocab-handler/word-to-id-dict.pkl", run_path=run_path
    )

    loading_path = Path(id_to_word_dict_save_path.name).parent
    logger.info(f"Loading from the path {loading_path}.")

    default_vocab_handler.load(loading_path)

# Callback to clear memory and restart keras backend at the end of each epoch
# https://stackoverflow.com/questions/53683164/keras-occupies-an-indefinitely-increasing-amount-of-memory-for-each-epoch


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()


clear_memory_callback = ClearMemory()
callbacks.append(clear_memory_callback)

early_stopping_callback = keras.callbacks.EarlyStopping(monitor="loss", patience=3)
callbacks.append(early_stopping_callback)

# Training
all_img_ids = captions_data["image"].unique().tolist()

logger.info(f"Total samples: {len(all_img_ids)}")


def calculate_total_samples(data):
    # Calculate the length of captions
    length_of_caption_df = captions_data.copy()
    length_of_caption_df["length_of_caption"] = captions_data["caption"].apply(
        lambda x: len(str(x).split())
    )

    # Filter the dataframe to only include images in the given data
    filtered_df = length_of_caption_df[length_of_caption_df["image"].isin(data)]

    # Sum the lengths directly
    total_samples = filtered_df["length_of_caption"].sum()

    return total_samples


# setup generators
train, test_and_validation = train_test_split(
    all_img_ids,
    test_size=(
        (TRAIN_TEST_VAL_SPLIT[1] + TRAIN_TEST_VAL_SPLIT[2]) / sum(TRAIN_TEST_VAL_SPLIT)
    ),
)
test, validation = train_test_split(
    test_and_validation,
    test_size=TRAIN_TEST_VAL_SPLIT[2]
    / (TRAIN_TEST_VAL_SPLIT[1] + TRAIN_TEST_VAL_SPLIT[2]),
)

total_training_samples = calculate_total_samples(train)
total_validation_samples = calculate_total_samples(validation)

steps = total_training_samples // BATCH_SIZE
val_steps = total_validation_samples // BATCH_SIZE

logger.info(f"Total batches in an epoch: {steps}")
logger.info(f"Total batches in validation set: {val_steps}")

generator = data_generator(
    train, default_vocab_handler, absolute_max_length, BATCH_SIZE
)
val_generator = data_generator(
    validation, default_vocab_handler, absolute_max_length, BATCH_SIZE
)
test_generator = data_generator(
    test, default_vocab_handler, absolute_max_length, BATCH_SIZE
)

training_args = {
    "epochs": N_EPOCHS,
    "steps_per_epoch": steps,
    "verbose": 1,
    "validation_data": val_generator,
    "callbacks": callbacks,
    "validation_freq": 1,
    "validation_steps": val_steps,
}

with strategy.scope() if USE_MULTIPLE_GPUS else nullcontext():
    if CONTINUE_FROM_WANDB_RUN:
        model = tf.keras.models.load_model(downloaded_model_path)
        logger.info(f"Continuing from the previous training sample from wandb")
    history = model.fit(generator, **training_args)
    if EVALUATE_AFTER_TRAIN:
        model.evaluate(test_generator)


batch_size = 32


batch_size = 32
with strategy.scope() if USE_MULTIPLE_GPUS else nullcontext():
    predicted = get_predictions_parallelized(test, batch_size)

actual = []
for i, image_id in tqdm(enumerate(test)):
    # since all the inference is run up to maxlength, the generated text
    # might contain many <stop> i.e. stop words
    # so discarding after encountering the first stop word
    first_index_of_end_sequence = predicted[i].find(default_vocab_handler.stop_word)
    predicted[i] = predicted[i][0: first_index_of_end_sequence].split()
    
    _actual = list(map(lambda x: x.split(), captions_data.loc[captions_data['image'] == image_id]['caption'].tolist()))
    actual.append(_actual)

n_grams = (2,4)
scores = get_all_blue_scores(actual, predicted,n_grams)

for n_gram, score in zip(n_grams, scores):
    logger.info(f"Bleu@{n_gram}:\t{score}")
