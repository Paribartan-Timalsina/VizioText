from vocab_handler import VocabHandler
import random
import numpy as np
import pandas as pd
from keras.utils import to_categorical


def data_generator(
    training_ids: list[str],
    vocab_handler: VocabHandler,
    max_length: int,
    batch_size: int,
    features: dict[str, np.ndarray],
    captions_data: pd.DataFrame,
):
    """Generate infinite stream of data for training and validation

    Args
    -----
        training_ids (list[str]): The list of image ids
        vocab_handler (VocabHandler): The Tokenizer for handling vocabulary
        max_length (int): Maximum length for applying padding
        batch_size (int): Batch size
        features (dict[str, np.ndarray]): The feature output of the feature extraction model
        captions_data (pd.DataFrame): DataFrame of captions data containing image ids and captions

    Yields
    ------
        tuple(tuple(Batch of feature input, Batch of sequence input), Batch of one hot encoded sequence outputs): \
        The batch of input and outputs, outputs are just a single word so, the output batch consist of one-hot-encoded word ids
    """

    vocab_size = vocab_handler.vocab_size

    while True:
        img_features_input, text_inputs, text_outputs = list(), list(), list()

        random.shuffle(training_ids)
        sample = 0
        for img_id in training_ids:
            feature = features[img_id]

            # get the captions corresponding to the image_id
            captions = captions_data.loc[captions_data["image"] == img_id]

            for caption in captions["caption"].tolist():
                words = caption.split()
                words.insert(0, vocab_handler.start_word)
                words.append(vocab_handler.stop_word)
                n_words = len(words)

                for i in range(1, n_words):
                    img_features_input.append(feature[0])
                    text_inputs.append(
                        vocab_handler.text_to_sequence(
                            " ".join(words[:i]), max_length, True
                        )
                    )
                    text_outputs.append(
                        to_categorical(
                            [vocab_handler.id_of(words[i])], num_classes=vocab_size
                        )[0]
                    )

                    sample += 1

                    if sample == batch_size:
                        sample = 0

                        img_features_input, text_inputs, text_outputs = (
                            np.array(img_features_input),
                            np.array(text_inputs),
                            np.array(text_outputs),
                        )

                        yield (img_features_input, text_inputs), text_outputs

                        img_features_input, text_inputs, text_outputs = (
                            list(),
                            list(),
                            list(),
                        )

        # Yield any remaining samples that didn't make a full batch
        if img_features_input:
            img_batch = np.array(img_features_input)
            text_batch = np.array(text_inputs)
            output_batch = np.array(text_outputs)

            yield (img_batch, text_batch), output_batch
