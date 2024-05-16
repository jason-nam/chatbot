"""
Unit Tests for Decoder Layer
"""

import unittest
import tensorflow as tf
import numpy as np

from transformer.decoder.decoder_layer import decoder_layer
from transformer.attention.multi_head_attention import MultiHeadAttention

class TestDecoderLayer(unittest.TestCase):
    def setUp(self):
        self.units = 32
        self.d_model = 16
        self.num_heads = 4
        self.dropout = 0.1
        self.seq_len = 10
        self.batch_size = 2

        self.decoder_layer_model = decoder_layer(
            units=self.units,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

        self.inputs = tf.random.uniform((self.batch_size, self.seq_len, self.d_model))
        self.encoder_outputs = tf.random.uniform((self.batch_size, self.seq_len, self.d_model))
        self.look_ahead_mask = tf.random.uniform((self.batch_size, 1, self.seq_len, self.seq_len))
        self.padding_mask = tf.random.uniform((self.batch_size, 1, 1, self.seq_len))

    def test_output_shape(self):
        outputs = self.decoder_layer_model([self.inputs, self.encoder_outputs, self.look_ahead_mask, self.padding_mask])
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_layer_connectivity(self):
        try:
            _ = self.decoder_layer_model([self.inputs, self.encoder_outputs, self.look_ahead_mask, self.padding_mask])
        except Exception as e:
            self.fail(f"Decoder layer failed connectivity test with error: {e}")

    def test_attention_layers(self):
        attention1 = self.decoder_layer_model.get_layer('attention_1')
        attention2 = self.decoder_layer_model.get_layer('attention_2')
        self.assertIsInstance(attention1, MultiHeadAttention)
        self.assertIsInstance(attention2, MultiHeadAttention)

    def test_dropout_and_normalization(self):
        dropout_found = False
        normalization_found = False
        for layer in self.decoder_layer_model.layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                dropout_found = True
                self.assertEqual(layer.rate, self.dropout)
            if isinstance(layer, tf.keras.layers.LayerNormalization):
                normalization_found = True
                self.assertEqual(layer.epsilon, 1e-6)
        self.assertTrue(dropout_found)
        self.assertTrue(normalization_found)

    def test_masking_effect(self):
        inputs_with_mask = [self.inputs, self.encoder_outputs, self.look_ahead_mask, self.padding_mask]
        inputs_without_mask = [self.inputs, self.encoder_outputs, None, None]
        output_with_mask = self.decoder_layer_model(inputs_with_mask)
        output_without_mask = self.decoder_layer_model(inputs_without_mask)
        self.assertFalse(tf.reduce_all(tf.equal(output_with_mask, output_without_mask)).numpy())

if __name__ == '__main__':
    unittest.main()