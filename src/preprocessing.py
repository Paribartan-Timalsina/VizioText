import os

import pandas as pd
from _logging import get_logger
from parameters import (DATASET_INPUT_DIR, FILTER_NON_ALPHA_NUMERIC_STRINGS,
                        N_WORDS_LOWER_LIMIT, N_WORDS_UPPER_LIMIT, TEMP_DIR,
                        WORKING_DIR)


def check_directories():
    """
    Check if WORKING_DIR, DATASET_INPUT_DIR, TEMP_DIR exist.

    Returns
    -------
    True if all directories are valid, False otherwise.
    """

    logger = get_logger()
    all_good = True

    if not os.path.exists(WORKING_DIR):
        logger.error(f"The directory {WORKING_DIR} doesn't exist.")
        all_good = False

    if not os.path.exists(DATASET_INPUT_DIR):
        logger.error(f"The directory {DATASET_INPUT_DIR} doesn't exist.")
        all_good = False

    if not os.path.exists(TEMP_DIR):
        logger.error(f"The directory {TEMP_DIR} doesn't exist.")
        all_good = False

    if all_good:
        logger.info(f"All the directories are valid.")

    return all_good

def filter_based_on_word_count(captions_data: pd.DataFrame, n_words_upper_limit: int = N_WORDS_UPPER_LIMIT, n_words_lower_limit: int = N_WORDS_LOWER_LIMIT):
    """
    Parameters
    ----------
    captions_data: The captions data containing image_id and caption
    n_words_upper_limit: The upper limit of words above which the whole caption is discarded
    n_words_lower_limit: The lower limit of words below which the whole caption is discarded

    Returns
    -------
    The dataframe with dropped values outside of the given limits
    """
    return captions_data.drop(captions_data.loc[captions_data["caption"]
                    .apply(lambda x: len(str(x).split()) > n_words_upper_limit or len(str(x).split()) < n_words_lower_limit)]
                    .index)

def filter_based_on_special_characters(captions_data: pd.DataFrame, non_alpha_thershold: int = 3):
    """
    Remove short special characters 

    Paramaters
    ----------
    captions_data: The captions data containing image_id and caption
    non_alpha_threshold: The minimum length of word above which the word is kept regardless of occurance of special characters
    
    Returns
    -------
    The filtered captions data, new vocabulary corresponding to that data and the removed words
    """
    unfiltered_vocabulary = list(set(" ".join(captions_data["caption"].to_list()).lower().split()))

    removed_items = []

    vocabulary = list(filter(lambda x: len(x) >= non_alpha_thershold or x.isalpha() , unfiltered_vocabulary))
    removed_items += list(filter(lambda x: len(x) < non_alpha_thershold and not x.isalpha() , unfiltered_vocabulary))

    filtered_captions_data = captions_data.copy()
    filtered_captions_data["caption"] = filtered_captions_data["caption"].apply(
        lambda x: " ".join(list(filter(
            lambda y: y not in removed_items, x.lower().split()))
                        )
    )

    return filtered_captions_data, vocabulary, removed_items
