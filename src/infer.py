from pathlib import Path
import numpy as np
from parameters import IMAGE_INPUT_SHAPE, INPUT_FEATURE_SHAPE, PRETRAINED_MODEL_LOCATION
from keras import Model
from keras.utils import load_img, img_to_array
from keras.applications.vgg19 import preprocess_input, VGG19
from keras.models import load_model
from vocab_handler import VocabHandler

from PIL import Image
import matplotlib.pyplot as plt

model_for_inference: Model | None = None
base_model: Model | None = None
inference_vocab_handler: VocabHandler | None = None
default_model_path: Path = PRETRAINED_MODEL_LOCATION
default_vocab_handler_path: Path = default_model_path.parent

test_images_path: Path = Path(__file__).parents[1].joinpath("test_images")

def get_base_model():
    global base_model
    base_model = VGG19(
        include_top=True,
        weights="imagenet",
        input_tensor=None,
        input_shape=IMAGE_INPUT_SHAPE,
        pooling="max",
    )
    base_model = Model(inputs=base_model.inputs, outputs=base_model.layers[-2].output)
    base_model.training = False

    return base_model


def load_model_for_inference(model_path: Path = default_model_path) -> Model:
    global model_for_inference
    model_for_inference = load_model(model_path)

    return model_for_inference

def test_on_new_image(image: Path | np.ndarray, max_text_length: int, tokenizer: VocabHandler, base_model: Model):
    if isinstance(image, Path):
        image = load_img(image, target_size=IMAGE_INPUT_SHAPE)
        image = img_to_array(image)

    reshaped_img = image.reshape(1, *IMAGE_INPUT_SHAPE)
    image_input = preprocess_input(reshaped_img)

    if model_for_inference is None:
        raise Exception("Inference model not initialized, call load_model_for_inference first.")
    _feature = base_model.predict(image_input)[0]
    
    text_input = "<start> "
    whole_text_output = ""
    for i in range(max_text_length):
        sequence_input = tokenizer.text_to_sequence(text_input, max_text_length, True)

        model_input = [_feature.reshape((1, INPUT_FEATURE_SHAPE[0])), sequence_input.reshape((1,max_text_length))]
        sequence_output = np.argmax(model_for_inference.predict(model_input, verbose=0)[0])
        
        text_output = tokenizer.word_of(sequence_output)
        
        if text_output == tokenizer.stop_word:
            break
        whole_text_output += " " + text_output
        text_input += " " + text_output
    
    return whole_text_output

def initialize():
    global inference_vocab_handler
    base_model: Model = get_base_model()

    model = load_model_for_inference()
    inference_vocab_handler = VocabHandler(["a", "an", "the"])
    inference_vocab_handler.load(default_vocab_handler_path)

def run_on_custom_image(image):
    return test_on_new_image(image, 31, inference_vocab_handler, base_model)

if __name__=="__main__":
    base_model: Model = get_base_model()

    model = load_model_for_inference()
    inference_vocab_handler = VocabHandler(["a", "an", "the"])
    inference_vocab_handler.load(default_vocab_handler_path)

    image_name = "little_girl_wooden_house.png"
    full_image_path = test_images_path.joinpath(image_name)

    generated_caption = test_on_new_image(full_image_path, 31, inference_vocab_handler, base_model)

    image = Image.open(full_image_path)
    # plt.imshow(image)
    print(generated_caption)
