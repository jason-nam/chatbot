"""
Transformer Model
"""

import tensorflow as tf

from src.transformer.encoder import encoder
from src.transformer.decoder import decoder

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  # (batch_size, 1, 1, length of key)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x) # including padding mask
  return tf.maximum(look_ahead_mask, padding_mask)

def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
    inputs = tf.keras.Input(shape=(None, ), name='inputs')
    decoder_inputs = tf.keras.Input(shape=(None, ), name='decoder_inputs')

    encoder_padding_mask = tf.keras.layers.Lambda(
        create_padding_mask, output_shape=(1, 1, None), name='encoder_padding_mask')(inputs)

    # mask the future tokens for decoder inputs at the 1st attention block
    look_ahead_mask = tf.keras.layers.Lambda(
        create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(decoder_inputs)
    


    return