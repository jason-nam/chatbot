"""
Unit Tests for Encoder 
"""

import unittest
import tensorflow as tf
import numpy as np

from transformer.encoder.encoder import encoder
from transformer.positional_encoding import PositionalEncoding

class TestEncoder(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.num_layers = 2
        self.units = 32
        self.d_model = 16
        self.num_heads = 4
        self.dropout = 0.1
        self.seq_len = 10
        self.batch_size = 2

        self.encoder_model = encoder(
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            units=self.units,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        self.inputs = tf.random.uniform((self.batch_size, self.seq_len), maxval=self.vocab_size, dtype=tf.int32)
        self.padding_mask = tf.random.uniform((self.batch_size, 1, 1, self.seq_len))

    def test_output_shape(self):
        outputs = self.encoder_model([self.inputs, self.padding_mask])
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_layer_connectivity(self):
        try:
            _ = self.encoder_model([self.inputs, self.padding_mask])
        except Exception as e:
            self.fail(f"Encoder failed connectivity test with error: {e}")

    def test_embedding_application(self):
        embedding_layer = self.encoder_model.get_layer(index=2)
        self.assertIsInstance(embedding_layer, tf.keras.layers.Embedding)
        self.assertEqual(embedding_layer.input_dim, self.vocab_size)
        self.assertEqual(embedding_layer.output_dim, self.d_model)

    def test_positional_encoding_application(self):
        positional_encoding_layer = self.encoder_model.get_layer(index=4)
        self.assertIsInstance(positional_encoding_layer, PositionalEncoding)

    def test_number_of_encoder_layers(self):
        encoder_layers = [layer for layer in self.encoder_model.layers if 'encoder_layer_' in layer.name]
        self.assertEqual(len(encoder_layers), self.num_layers)

if __name__ == '__main__':
    unittest.main()
