"""
Unit Test for Multi Head Attention
"""

import unittest
import tensorflow as tf
import numpy as np

from transformer.attention.multi_head_attention import MultiHeadAttention

class TestMultiHeadAttention(unittest.TestCase):
    def setUp(self):
        self.d_model = 16
        self.num_heads = 4
        self.seq_len = 6
        self.batch_size = 2

        self.mha = MultiHeadAttention(d_model=self.d_model, num_heads=self.num_heads)
        self.query = tf.random.uniform((self.batch_size, self.seq_len, self.d_model))
        self.key = tf.random.uniform((self.batch_size, self.seq_len, self.d_model))
        self.value = tf.random.uniform((self.batch_size, self.seq_len, self.d_model))
        self.mask = tf.random.uniform((self.batch_size, 1, 1, self.seq_len))

    def test_output_shape(self):
        inputs = {'query': self.query, 'key': self.key, 'value': self.value, 'mask': None}
        output = self.mha(inputs)
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, self.d_model))

    def test_masking_effect(self):
        inputs_with_mask = {'query': self.query, 'key': self.key, 'value': self.value, 'mask': self.mask}
        inputs_without_mask = {'query': self.query, 'key': self.key, 'value': self.value, 'mask': None}
        output_with_mask = self.mha(inputs_with_mask)
        output_without_mask = self.mha(inputs_without_mask)
        self.assertFalse(np.allclose(output_with_mask.numpy(), output_without_mask.numpy()))

    def test_split_heads(self):
        batch_size = self.batch_size
        inputs = tf.random.uniform((batch_size, self.seq_len, self.d_model))
        split = self.mha.split_heads(inputs, batch_size)
        self.assertEqual(split.shape, (batch_size, self.num_heads, self.seq_len, self.d_model // self.num_heads))

if __name__ == '__main__':
    unittest.main()
