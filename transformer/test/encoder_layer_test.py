"""
Unit Tests for Encoder Layer
"""

import unittest
import tensorflow as tf
import numpy as np

from transformer.encoder.encoder_layer import encoder_layer

class TestEncoderLayer(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.num_heads = 4
        self.units = 32
        self.dropout = 0.1
        self.seq_len = 10
        self.batch_size = 2

        self.encoder = encoder_layer(units=self.units, d_model=self.d_model, num_heads=self.num_heads, dropout=self.dropout)
        self.inputs = tf.random.uniform((self.batch_size, self.seq_len, self.d_model))
        self.padding_mask = tf.random.uniform((self.batch_size, 1, 1, self.seq_len))

    def test_output_shape(self):
        outputs = self.encoder([self.inputs, self.padding_mask])
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_layer_connectivity(self):
        try:
            _ = self.encoder([self.inputs, self.padding_mask])
        except Exception as e:
            self.fail(f"Encoder layer failed connectivity test with error: {e}")

    def test_dropout_and_normalization(self):
        for layer in self.encoder.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                self.assertEqual(layer.rate, self.dropout)
            if isinstance(layer, tf.keras.layers.LayerNormalization):
                self.assertEqual(layer.epsilon, 1e-6)

if __name__ == '__main__':
    unittest.main()
