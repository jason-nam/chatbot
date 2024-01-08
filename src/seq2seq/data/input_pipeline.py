"""
Input pipelines
"""

import abc

import tensorflow as tf

class InputPipeline(abc.ABC):

    """
    Abstract input pipeline class
    """

    def __init__(self, source, batch_size):
        self.source = source
        self.batch_size = batch_size

    @property
    def feature_keys(self):
        return set()
    
    @property
    def label_keys(self):
        return set()

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess_data(self):
        pass

    @abstractmethod
    def batch_data(self):
        pass

    @abstractmethod
    def get_next_batch(self):
        pass

class ChatDataPipeline(InputPipeline):

    """
    Chat data pipeline class
    """
    
    def __init__(self, source, batch_size):
        super().__init__(source, batch_size)

    def load_data(self):
        return super().load_data()
    
    def preprocess_data(self):
        return super().preprocess_data()