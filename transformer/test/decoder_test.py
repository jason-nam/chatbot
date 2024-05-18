"""
Unit Tests for Decoder
"""

import unittest
import tensorflow as tf
import numpy as np

from transformer.decoder.decoder import decoder
from transformer.decoder.decoder_layer import decoder_layer
from transformer.positional_encoding import PositionalEncoding
from transformer.attention.multi_head_attention import MultiHeadAttention

class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.vocab_size = 1000
        self.num_layers = 2
        self.units = 32
        self.d_model = 16
        self.num_heads = 4
        self.dropout = 0.1
        self.seq_len = 10
        self.batch_size = 2

        self.decoder_model = decoder(
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            units=self.units,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dropout=self.dropout
        )

        self.inputs = tf.random.uniform((self.batch_size, self.seq_len), maxval=self.vocab_size, dtype=tf.int32)
        self.encoder_outputs = tf.random.uniform((self.batch_size, self.seq_len, self.d_model))
        self.look_ahead_mask = tf.random.uniform((self.batch_size, 1, self.seq_len, self.seq_len))
        self.padding_mask = tf.random.uniform((self.batch_size, 1, 1, self.seq_len))
        
        # Print the layer types
        # for i, layer in enumerate(self.decoder_model.layers):
        #     print(f"Layer {i}: {layer.name}, {type(layer)}")

    def test_output_shape(self):
        outputs = self.decoder_model([self.inputs, self.encoder_outputs, self.look_ahead_mask, self.padding_mask])
        self.assertEqual(outputs.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_layer_connectivity(self):
        try:
            _ = self.decoder_model([self.inputs, self.encoder_outputs, self.look_ahead_mask, self.padding_mask])
        except Exception as e:
            self.fail(f"Decoder model failed connectivity test with error: {e}")

    def test_embedding_and_positional_encoding(self):
        embedding_layer = self.decoder_model.layers[1]
        positional_encoding_layer = self.decoder_model.layers[3]

        self.assertIsInstance(embedding_layer, tf.keras.layers.Embedding)
        self.assertEqual(embedding_layer.output_dim, self.d_model)

        self.assertIsInstance(positional_encoding_layer, PositionalEncoding)

    def test_dropout_and_normalization(self):
        def find_layers(model, layer_type):
            layers_found = []
            for layer in model.layers:
                if isinstance(layer, layer_type):
                    layers_found.append(layer)
                elif isinstance(layer, tf.keras.Model):
                    layers_found.extend(find_layers(layer, layer_type))
            return layers_found

        dropouts = find_layers(self.decoder_model, tf.keras.layers.Dropout)
        normalizations = find_layers(self.decoder_model, tf.keras.layers.LayerNormalization)

        self.assertTrue(len(dropouts) > 0, "No Dropout layers found")
        self.assertTrue(len(normalizations) > 0, "No LayerNormalization layers found")

        for dropout in dropouts:
            self.assertEqual(dropout.rate, self.dropout)

        for normalization in normalizations:
            self.assertEqual(normalization.epsilon, 1e-6)

    def test_attention_layers(self):
        for i in range(self.num_layers):
            decoder_layer_instance = self.decoder_model.get_layer(f'decoder_layer_{i}')
            attention1 = decoder_layer_instance.get_layer('attention_1')
            attention2 = decoder_layer_instance.get_layer('attention_2')
            self.assertIsInstance(attention1, MultiHeadAttention)
            self.assertIsInstance(attention2, MultiHeadAttention)

    def test_masking_effect(self):
        no_look_ahead_mask = tf.zeros((self.batch_size, 1, self.seq_len, self.seq_len))
        no_padding_mask = tf.zeros((self.batch_size, 1, 1, self.seq_len))
        inputs_with_mask = [self.inputs, self.encoder_outputs, self.look_ahead_mask, self.padding_mask]
        inputs_without_mask = [self.inputs, self.encoder_outputs, no_look_ahead_mask, no_padding_mask]
        output_with_mask = self.decoder_model(inputs_with_mask)
        output_without_mask = self.decoder_model(inputs_without_mask)
        self.assertFalse(tf.reduce_all(tf.equal(output_with_mask, output_without_mask)).numpy())

if __name__ == '__main__':
    unittest.main()
