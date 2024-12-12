from keras.layers import Input, Dropout, Dense, Embedding, add, LSTM
from keras import Model

from parameters import INPUT_FEATURE_SHAPE


def get_image_captioning_model(
    vocab_size: int,
    max_text_length: int,
    input_feature_shape: tuple[int, int] = INPUT_FEATURE_SHAPE,
) -> Model:
    """Returns the main model for image captioning

    Args
    -----
        vocab_size (int): The size of the vocabulary of the data upon which model is to be trained and tested
        max_text_length (int): The maximum number of token(encoded words) accepted by model as an input
        input_feature_shape (tuple[int, int], optional): The shape of output given by the base model. Defaults to INPUT_FEATURE_SHAPE.

    Returns
    -------
        keras.Model: The image captioning model which accepts (feature, encoded text) as input and gives encoded word (one-hot) as output
    """
    feature_input_layer = Input(shape=(input_feature_shape[1],))

    # some trainable layers before merging
    dropout_1 = Dropout(0.4)(feature_input_layer)
    image_feature_output = Dense(256, activation="relu")(dropout_1)

    # text feature extraction
    text_input_layer = Input(shape=(max_text_length,))
    embed = Embedding(vocab_size, 256, mask_zero=True)(text_input_layer)
    dropout_2 = Dropout(0.4)(embed)
    text_feature_output = LSTM(256)(dropout_2)

    # decoding
    combine = add([image_feature_output, text_feature_output])
    dense_decoder = Dense(256, activation="relu")(combine)
    outputs = Dense(vocab_size, activation="softmax")(dense_decoder)

    model = Model(inputs=[feature_input_layer, text_input_layer], outputs=outputs)

    return model
