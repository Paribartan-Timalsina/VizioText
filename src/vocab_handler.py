import numpy as np
import pickle
from pathlib import Path
from _logging import get_logger

class VocabHandler():
    """
    Handles vocabulary. Indices start from 1 since 0 is reserved for padding.
    """
    word_to_id_dict: dict[str, int] = {}
    id_to_word_dict: dict[int, str] = {}
    vocab_size = 0
        
    def __init__(self, vocabulary: list[str], start_word: str = "<start>", stop_word: str = "<stop>", count_padding_as_separate_word: bool = True, padding: str = "<padding>"):
        """
        ID 0 is used for padding
        
        Parameters
        ----------
        vocabulary: The list of words
        start_word: special word for denoting the start of generation
        stop_word: special word for denoting the stop of generation
        count_padding_as_separate_word: if false, there won't be a entry for ID 0
        padding: the special word place where id=0
        """
        
        self.start_word: str = start_word
        self.stop_word: str = stop_word
        
        if count_padding_as_separate_word:
            self.padding = padding
            self.word_to_id_dict[self.padding] = 0
            self.id_to_word_dict[0] = self.padding
        
        # adding start word in the vocabulary
        self.word_to_id_dict[self.start_word] = 1
        self.id_to_word_dict[1] = self.start_word
        
        last_index = 0
        for idx, word in enumerate(vocabulary):
            self.word_to_id_dict[word] = idx + 2
            self.id_to_word_dict[idx + 2] = word 
            last_index = idx + 2
            
        # adding start word in the vocabulary
        self.word_to_id_dict[self.stop_word] = last_index + 1
        self.id_to_word_dict[last_index + 1] = self.stop_word
        
        assert len(self.word_to_id_dict) == len(self.id_to_word_dict)
        
        self.vocab_size = len(self.word_to_id_dict)

        self.logger = get_logger()

    def id_of(self, word: str) -> int | None:
        """
        Get the id of the given word.
        """
        return self.word_to_id_dict[word]

    def word_of(self, idx: int) -> str | None:
        """
        Get the word corresponding to the given id.
        """
        return self.id_to_word_dict[idx]
    
    def text_to_sequence(self, text: str, max_length: int = 0, padding: bool = False, pad_with: int = 0) -> np.ndarray:
        """
        Converts the given sentence into sequence of tokens(ids).

        Parameters
        ----------
        text: The input sentence
        max_length: The maximum length of the output sequence with padding applied
        padding: Whether to apply padding ( the padding is right padding )
        pad_with: The padding index, default is 0

        Returns
        -------
        The sequence (right padded if padding = True, otherwise sequence corredpoding to the given text.) 
        """
        words = text.split()
        
        if not padding or max_length < 1:
            if max_length < 1:
                self.logger.error(f"The provided maximum length {max_length} is invalid.")
            return np.array(list(map(lambda x: self.id_of(x), words)))
        
        len_words = len(words)
        
        padded_sequence = np.full(max_length, pad_with)
        padded_sequence[:len_words] = np.array(list(map(lambda x: self.id_of(x), words)))
        
        
        return padded_sequence
    
    def sequence_to_text(self, sequence: np.ndarray, padded: bool = False, padded_with: int = 0):
        """
        Converts the given sequence(collection of tokens(ids)) into corresponding words

        Parameters
        ----------
        sequence: The input sequence (1d numpy array of integers)
        padded: Whether the input sequence is padded
        padded_with: What the input sequence is padded with (if any)
        """
        if not padded:
            return " ".join(map(lambda x: self.word_of(x), sequence))
        
        return " ".join(filter(lambda y: y!="", map(lambda x: self.word_of(x) if x!= padded_with else "", sequence)))
    
    def save(self, file_location: Path):
        """
        Save the current vocab handler for later use
        """
        pickle.dump(self.word_to_id_dict, open(file_location.joinpath("word-to-id-dict.pkl"), 'wb'))
        pickle.dump(self.id_to_word_dict, open(file_location.joinpath("id-to-word-dict.pkl"), 'wb'))

    def load(self, file_location: Path):
        """
        Load the previously saved vocab handler
        """
        self.word_to_id_dict = pickle.load(open(file_location.joinpath("word-to-id-dict.pkl"), 'rb'))
        self.id_to_word_dict= pickle.load(open(file_location.joinpath("id-to-word-dict.pkl"), 'rb'))
    