"""
Implementation of Positional Encoding
"""

import tensorflow as tf

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))

        return position * angles
    
    def positional_encoding(self, position, d_model):
        angle_rad = self.get_angles(
                        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis], 
                        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :], 
                        d_model=d_model)
        
        sin_even = tf.math.sin(angle_rad[:, 0::2])
        cos_odd = tf.math.cos(angle_rad[:, 1::2])

        pos_encoding = tf.concat([sin_even, cos_odd], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        
        return tf.cast(pos_encoding, tf.float32)
    
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]