"""
Scaled Dot Product Attention Unit Tests
"""

import unittest
import tensorflow as tf
import numpy as np

from transformer.attention.scaled_dot_product_attention import scaled_dot_product_attention

class TestScaledDotProductAttention(unittest.TestCase):

    def setUp(self):
        # Setup with some random data
        self.batch_size = 2
        self.num_heads = 4
        self.query_len = 5
        self.key_len = 6
        self.depth = 32  # d_model/num_heads

        self.query = tf.random.normal((self.batch_size, self.num_heads, self.query_len, self.depth))
        self.key = tf.random.normal((self.batch_size, self.num_heads, self.key_len, self.depth))
        self.value = tf.random.normal((self.batch_size, self.num_heads, self.key_len, self.depth))
        self.mask = tf.ones((self.batch_size, 1, 1, self.key_len))

    def test_output_shape(self):
        """Test if the output shape is as expected."""
        output, attn_weights = scaled_dot_product_attention(self.query, self.key, self.value, None)
        self.assertEqual(output.shape, (self.batch_size, self.num_heads, self.query_len, self.depth))
        self.assertEqual(attn_weights.shape, (self.batch_size, self.num_heads, self.query_len, self.key_len))

    def test_attention_weights_sum_to_one(self):
        """Test if the attention weights across the key dimension sum to one."""
        _, attn_weights = scaled_dot_product_attention(self.query, self.key, self.value, None)
        self.assertTrue(np.allclose(tf.reduce_sum(attn_weights, axis=-1).numpy(), np.ones((self.batch_size, self.num_heads, self.query_len))))

    def test_masking_effectiveness(self):
        """Test whether the mask properly ignores the specified values."""
        self.mask = tf.concat([tf.ones((self.batch_size, 1, 1, self.key_len - 1)), tf.zeros((self.batch_size, 1, 1, 1))], axis=-1)
        _, attn_weights = scaled_dot_product_attention(self.query, self.key, self.value, self.mask)

        print(attn_weights)
        print(attn_weights[..., -1].numpy())

        # Check if the last key position is not attended to at all
        self.assertTrue(np.allclose(attn_weights[..., -1].numpy(), 0))

    def test_dtype_consistency(self):
        """Test if the data type of the output matches the input."""
        output, _ = scaled_dot_product_attention(self.query, self.key, self.value, None)
        self.assertEqual(output.dtype, self.query.dtype)

if __name__ == '__main__':
    unittest.main()
