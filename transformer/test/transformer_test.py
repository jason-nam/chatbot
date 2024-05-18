"""
Unit Tests for Transformer
"""

import unittest
import tensorflow as tf
import numpy as np

from transformer.transformer import transformer, create_padding_mask, create_look_ahead_mask
from transformer.encoder.encoder import encoder
from transformer.decoder.decoder import decoder
from transformer.positional_encoding import PositionalEncoding
from transformer.attention.multi_head_attention import MultiHeadAttention

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.num_layers = 2
        self.units = 32
        self.d_model = 16
        self.num_heads = 4
        self.dropout = 0.1
        self.seq_len = 10
        self.batch_size = 2

        self.transformer_model = transformer(
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            units=self.units,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

        self.inputs = tf.random.uniform((self.batch_size, self.seq_len), maxval=self.vocab_size, dtype=tf.int32)
        self.decoder_inputs = tf.random.uniform((self.batch_size, self.seq_len), maxval=self.vocab_size, dtype=tf.int32)

    def test_output_shape(self):
        outputs = self.transformer_model([self.inputs, self.decoder_inputs])
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.vocab_size))

    def test_layer_connectivity(self):
        try:
            _ = self.transformer_model([self.inputs, self.decoder_inputs])
        except Exception as e:
            self.fail(f"Transformer model failed connectivity test with error: {e}")

    def test_create_padding_mask(self):
        x = tf.constant([[7, 6, 0], [1, 0, 0]])
        expected_mask = tf.constant([[[[0., 0., 1.]]], [[[0., 1., 1.]]]])
        padding_mask = create_padding_mask(x)
        self.assertTrue(tf.reduce_all(tf.equal(padding_mask, expected_mask)).numpy())

    def test_create_look_ahead_mask(self):
        x = tf.constant([[7, 6, 0], [1, 0, 0]])
        expected_look_ahead_mask = tf.constant([[[[0., 1., 1.], [0., 0., 1.], [0., 0., 0.]]],
                                                [[[0., 1., 1.], [0., 0., 1.], [0., 0., 0.]]]])
        look_ahead_mask = create_look_ahead_mask(x)
        self.assertTrue(tf.reduce_all(tf.equal(look_ahead_mask, expected_look_ahead_mask)).numpy())

    def test_embedding_and_positional_encoding(self):
        embedding_layer = self.transformer_model.get_layer(index=2)
        positional_encoding_layer = self.transformer_model.get_layer(index=3)

        self.assertIsInstance(embedding_layer, tf.keras.layers.Embedding)
        self.assertEqual(embedding_layer.output_dim, self.d_model)

        self.assertIsInstance(positional_encoding_layer, PositionalEncoding)
        self.assertEqual(positional_encoding_layer.d_model, self.d_model)

    def test_attention_layers(self):
        for i in range(self.num_layers):
            encoder_layer_instance = self.transformer_model.get_layer(f'encoder_layer_{i}')
            attention = encoder_layer_instance.get_layer('attention')
            self.assertIsInstance(attention, MultiHeadAttention)

            decoder_layer_instance = self.transformer_model.get_layer(f'decoder_layer_{i}')
            attention1 = decoder_layer_instance.get_layer('attention_1')
            attention2 = decoder_layer_instance.get_layer('attention_2')
            self.assertIsInstance(attention1, MultiHeadAttention)
            self.assertIsInstance(attention2, MultiHeadAttention)

if __name__ == '__main__':
    unittest.main()