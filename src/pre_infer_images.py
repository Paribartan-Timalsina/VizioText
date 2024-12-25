from keras import Model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing.image import load_img, img_to_array

from parameters import IMAGE_INPUT_SHAPE, TRAIN_CNN, BATCH_SIZE, INPUT_DIR
from contextlib import nullcontext
from tensorflow.distribute import MirroredStrategy
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pickle


def get_base_model():
    """
    Returns
    -----
    VGG19 model with the last layer removed
    """
    base_model = VGG19(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=IMAGE_INPUT_SHAPE,
        pooling="max",
    )
    base_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    base_model.training = TRAIN_CNN

    return base_model


def extract_image_features_and_save(
    images,
    images_dir: Path,
    features_dir: Path,
    base_model: Model,
    batch_size=BATCH_SIZE,
    save_to_file=False,
    strategy: MirroredStrategy | None = None,
) -> dict | None:
    """
    Returns features if kept in RAM None otherwise
    """
    features = {}

    with nullcontext() if strategy is None else strategy.scope():
        for i in tqdm(range(0, len(images), batch_size)):
            img_arr = map(
                lambda x: img_to_array(
                    load_img(images_dir.joinpath(x), target_size=IMAGE_INPUT_SHAPE)
                ),
                images[i : i + batch_size],
            )
            predictions = base_model.predict(
                preprocess_input(np.array(list(img_arr))), verbose=0
            )

            for image_name, prediction in zip(images[i : i + batch_size], predictions):
                features[image_name] = prediction

    if save_to_file:
        output_file = features_dir.joinpath("features.pkl")
        pickle.dump(features, open(output_file, "wb"))

    return features
