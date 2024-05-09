"""
Unit tests for positional encoding
"""

import unittest
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from transformer.positional_encoding import PositionalEncoding

def visualize_position_by_depth():
    # sentence length 50 and embedding vector layer 128
    sample_pos_encoding = PositionalEncoding(50, 128)

    plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 128))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

class TestPositionalEncoding(unittest.TestCase):
    
    def test_shape(self):
        """Test if the positional encoding returns the correct shape"""
        position = 50
        d_model = 128
        pe = PositionalEncoding(position, d_model)
        self.assertEqual(pe.pos_encoding.shape, (1, position, d_model),
                         "Positional Encoding shape is incorrect")

    def test_values(self):
        """Test specific known values or properties (like even/odd symmetry)"""
        position = 10
        d_model = 16
        pe = PositionalEncoding(position, d_model)
        pos_encoding_array = pe.pos_encoding.numpy()

         # Fetching angles
        angles = pe.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], 
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], 
            d_model=d_model)

        # sin and cos periodicity and values check at specific positions
        sin_values = np.sin(angles[:, 0::2])
        cos_values = np.cos(angles[:, 1::2])

        # Check the differences for debugging
        diff_sin = np.abs(sin_values - pos_encoding_array[0, :, 0::2])
        diff_cos = np.abs(cos_values - pos_encoding_array[0, :, 1::2])

        # Test symmetry properties
        self.assertTrue(np.allclose(sin_values, pos_encoding_array[0, :, 0::2],
                                    atol=1e-6), f"Sine values do not match. Max diff: {np.max(diff_sin)}")
        self.assertTrue(np.allclose(cos_values, pos_encoding_array[0, :, 1::2],
                                    atol=1e-6), f"Cosine values do not match. Max diff: {np.max(diff_cos)}")

    def test_integration_with_keras_model(self):
        """Test if positional encoding can integrate with a tf.keras model"""
        position = 50
        d_model = 128
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, d_model)),
            PositionalEncoding(position, d_model)
        ])

        # Create a random input tensor of shape (batch_size, sequence_length, d_model)
        inputs = tf.random.uniform((1, 40, d_model))  # less than 'position' to test slicing
        outputs = model(inputs)
        
        # Check if outputs have the same shape as inputs
        self.assertEqual(outputs.shape, inputs.shape,
                         "Output shape of model with Positional Encoding does not match input shape")
    
    def test_dtype(self):
        """Test the data type of the positional encoding"""
        position = 10
        d_model = 16
        pe = PositionalEncoding(position, d_model)
        self.assertEqual(pe.pos_encoding.dtype, tf.float32, "Data type of positional encoding is incorrect")

if __name__ == "__main__":
    unittest.main()
