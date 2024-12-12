from contextlib import nullcontext
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
from keras import Model
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.utils import img_to_array, load_img
from nltk.translate.bleu_score import corpus_bleu
from tensorflow.distribute import MirroredStrategy
from tqdm import tqdm
from vocab_handler import VocabHandler


def test_on_image_having_feature(
    image_id: str,
    model: Model,
    features: dict[str, np.ndarray],
    feature_shape: int,
    absolute_max_length: int,
    vocab_handler: VocabHandler,
):
    """
    _summary_

    Parameters
    ----------
    image_id : str
        _description_
    model : Model
        _description_
    features : dict[str, np.ndarray]
        _description_
    feature_shape : int
        _description_
    absolute_max_length : int
        _description_
    vocab_handler : VocabHandler
        _description_

    Returns
    -------
    _type_
        _description_
    """
    _feature = features[image_id]

    text_input = "<start>"
    whole_text_output = ""
    for i in range(absolute_max_length):
        sequence_input = vocab_handler.text_to_sequence(
            text_input, absolute_max_length, True
        )

        model_input = [
            _feature.reshape((1, feature_shape)),
            sequence_input.reshape((1, absolute_max_length)),
        ]
        predictions = model.predict(model_input, verbose=0)
        sequence_output = np.argmax(predictions[0])

        text_output = vocab_handler.word_of(sequence_output)

        if text_output == vocab_handler.stop_word:
            break
        whole_text_output += " " + text_output
        text_input += " " + text_output

    return whole_text_output


def test_on_new_image(
    image: Path | np.ndarray,
    base_model: Model,
    model: Model,
    input_image_shape: tuple[int, int, int],
    feature_shape: int,
    absolute_max_length: int,
    vocab_handler: VocabHandler,
):
    """
    _summary_

    Parameters
    ----------
    image : Path | np.ndarray
        _description_
    base_model : Model
        _description_
    model : Model
        _description_
    input_image_shape : tuple[int, int, int]
        _description_
    feature_shape : int
        _description_
    absolute_max_length : int
        _description_
    vocab_handler : VocabHandler
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if isinstance(image, Path):
        image = load_img(image, target_size=input_image_shape)
        image = img_to_array(image)

    reshaped_img = image.reshape(1, *input_image_shape)
    image_input = preprocess_input(reshaped_img)

    _feature = base_model.predict(image_input)[0]

    text_input = "<start> "
    whole_text_output = ""
    for i in range(absolute_max_length):
        sequence_input = vocab_handler.text_to_sequence(
            text_input, absolute_max_length, True
        )

        model_input = [
            _feature.reshape((1, feature_shape)),
            sequence_input.reshape((1, absolute_max_length)),
        ]

        sequence_output = np.argmax(model.predict(model_input, verbose=0)[0])

        text_output = vocab_handler.word_of(sequence_output)

        if text_output == vocab_handler.stop_word:
            break
        whole_text_output += " " + text_output
        text_input += " " + text_output

    return whole_text_output


def get_blue_score(
    test: Generator,
    captions_data: pd.DataFrame,
    model: Model,
    features: dict[str, np.ndarray],
    feature_shape: int,
    absolute_max_length: int,
    vocab_handler: VocabHandler,
):
    actual = []
    predicted = []
    for image_id in tqdm(test[:3]):
        whole_text_output = test_on_image_having_feature(
            image_id, model, features, feature_shape, absolute_max_length, vocab_handler
        )

        _actual = list(
            map(
                lambda x: x.split(),
                captions_data.loc[captions_data["image"] == image_id][
                    "caption"
                ].tolist(),
            )
        )
        _predicted = whole_text_output.split()

        actual.append(_actual)
        predicted.append(_predicted)

    score = corpus_bleu(actual, predicted)


def get_predictions_parallelized(
    image_ids: Generator | list[str],
    model: Model,
    features: dict[str, np.ndarray],
    feature_shape: int,
    absolute_max_length: int,
    vocab_handler: VocabHandler,
    batch_size: int = 32,
):

    predicted: list[str] = []

    for i in tqdm(range(0, len(image_ids), batch_size)):
        images_batch = image_ids[i : i + batch_size]
        num_images_in_batch = len(images_batch)

        text_inputs = ["<start>" for _ in range(num_images_in_batch)]
        text_outputs = ["" for _ in range(num_images_in_batch)]

        sequence_inputs = np.zeros(shape=(num_images_in_batch, absolute_max_length))
        feature_inputs = np.zeros(shape=(num_images_in_batch, feature_shape))

        for j in range(absolute_max_length):
            for k, image in enumerate(images_batch):
                sequence_inputs[k] = vocab_handler.text_to_sequence(
                    text_inputs[k], absolute_max_length, True
                )
                feature_inputs[k] = features[image]

            model_inputs = [feature_inputs, sequence_inputs]

            sequence_outputs = list(
                map(lambda x: np.argmax(x), model.predict(model_inputs, verbose=0))
            )

            text_output = list(
                map(lambda x: vocab_handler.word_of(x), sequence_outputs)
            )

            for l in range(num_images_in_batch):
                text_outputs[l] += " " + text_output[l]
                text_inputs[l] += " " + text_output[l]

        predicted.extend(text_outputs)

    return predicted


def get_blue_score(
    test: Generator,
    captions_data: pd.DataFrame,
    model: Model,
    features: dict[str, np.ndarray],
    feature_shape: int,
    absolute_max_length: int,
    vocab_handler: VocabHandler,
    strategy: MirroredStrategy | None = None,
):
    batch_size = 32
    with nullcontext() if strategy is None else strategy.scope():
        predicted = get_predictions_parallelized(
            test,
            model,
            features,
            feature_shape,
            absolute_max_length,
            vocab_handler,
            batch_size,
        )

    actual = []
    for i, image_id in tqdm(enumerate(test)):
        # since all the inference is run up to maxlength, the generated text
        # might contain many <stop> i.e. stop words
        # so discarding after encountering the first stop word
        first_index_of_end_sequence = predicted[i].find(vocab_handler.stop_word)
        predicted[i] = predicted[i][0:first_index_of_end_sequence].split()

        _actual = list(
            map(
                lambda x: x.split(),
                captions_data.loc[captions_data["image"] == image_id][
                    "caption"
                ].tolist(),
            )
        )
        actual.append(_actual)

    return corpus_bleu(actual, predicted)


def get_all_blue_scores(
    references: list[list[list[str]]],
    hypotheses: list[list[str]],
    n_grams: tuple[int, int] = (2, 5),
):
    scores = []
    for i in range(*n_grams):
        weights = np.ones((i,))
        weights /= i
        scores.append(corpus_bleu(references, hypotheses, weights=weights))

    return scores
