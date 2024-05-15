"""
Scaled Dot Product Attention Unit Tests
"""

import unittest
import tensorflow as tf
import numpy as np

from transformer.attention.scaled_dot_product_attention import scaled_dot_product_attention

class TestScaledDotProductAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.num_heads = 2
        self.seq_len = 4
        self.depth = 8

        self.query = tf.random.uniform((self.batch_size, self.num_heads, self.seq_len, self.depth))
        self.key = tf.random.uniform((self.batch_size, self.num_heads, self.seq_len, self.depth))
        self.value = tf.random.uniform((self.batch_size, self.num_heads, self.seq_len, self.depth))
        self.mask = tf.random.uniform((self.batch_size, 1, 1, self.seq_len))

    def test_output_shape(self):
        output, _ = scaled_dot_product_attention(self.query, self.key, self.value, None)
        self.assertEqual(output.shape, (self.batch_size, self.num_heads, self.seq_len, self.depth))

    def test_attention_weights_shape(self):
        _, attention_weights = scaled_dot_product_attention(self.query, self.key, self.value, None)
        self.assertEqual(attention_weights.shape, (self.batch_size, self.num_heads, self.seq_len, self.seq_len))

    def test_masking_effect(self):
        output_with_mask, _ = scaled_dot_product_attention(self.query, self.key, self.value, self.mask)
        output_without_mask, _ = scaled_dot_product_attention(self.query, self.key, self.value, None)
        self.assertFalse(np.allclose(output_with_mask.numpy(), output_without_mask.numpy()))

    def test_softmax_sum(self):
        _, attention_weights = scaled_dot_product_attention(self.query, self.key, self.value, None)
        self.assertTrue(np.allclose(tf.reduce_sum(attention_weights, axis=-1).numpy(), np.ones((self.batch_size, self.num_heads, self.seq_len))))

if __name__ == '__main__':
    unittest.main()
