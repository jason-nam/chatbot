"""
Input pipelines
"""

import abc

import tensorflow as tf
from convokit import Corpus, download

class InputPipeline(abc.ABC):

    """
    Abstract input pipeline class
    """

    def __init__(self, source, batch_size):
        self.source = source
        self.batch_size = batch_size

    @staticmethod
    def default_params():
        return {
            "shuffle": True, 
            "num_epochs": None,
        }

    def load_data(self):
        pass

    def preprocess_data(self):
        pass

    def batch_data(self):
        pass

    def get_next_batch(self):
        pass

    @property
    def feature_keys(self):
        return set()
    
    @property
    def label_keys(self):
        return set()


class ConversationDataPipeline(InputPipeline):

    """
    Chat data pipeline class
    """
    
    def __init__(self, source, batch_size):
        super().__init__(source, batch_size)
        self.data = None
        self.current_batch = 0

    @staticmethod
    def default_params():
        params = InputPipeline.default_params()
        params.update({})
        return params

    def load_data(self):
        self.corpus = Corpus(filename=download("movie-corpus"))
        self.data = self.extract_conversations(self.corpus)
        return super().load_data()
    
    def extract_conversations(self, corpus):
        conversations = []
        for convo in corpus.iter_conversations():
            for utterance in convo.iter_utterances():
                input_text = utterance.text
        return conversations

    def preprocess_data(self):
        return super().preprocess_data()

    def batch_data(self):
        self.batches = None
        return super().batch_data()
    
    def get_next_batch(self):
        if self.current_batch < len(self.batches):
            batch = self.batches[self.current_batch]
            self.current_batch += 1
            return batch
        else:
            return None
        
    @property
    def feature_keys(self):
        return set(["source_tokens", "source_len"])
    
    @property
    def label_keys(self):
        return set(["target_tokens", "target_len"])
    

if __name__ == "__main__":
    pass