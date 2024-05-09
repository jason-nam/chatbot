"""
Implementation of Scaled Dot Product Attention
"""

import tensorflow as tf

def scaled_dot_product_attention(query, key, value, mask):
    # query size : (batch_size, num_heads, query sentence length, d_model/num_heads)
    # key size : (batch_size, num_heads, key sentence length, d_model/num_heads)
    # value size : (batch_size, num_heads, value sentence length, d_model/num_heads)
    # padding_mask : (batch_size, 1, 1, key sentence length)

    matmul_qk = tf.matmul(query, key, transpose_b=True)
    
    # scaling
    depth = tf.cast(tf.shape(key)[-1], tf.float32)
    logits = matmul_qk / tf.math.sqrt(depth)

    # masking
    if mask is not None:
        logits += (mask * -1e9)

    print(logits)

    # softmax function depends on the final layer key's sentence length axis
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)

    return output, attention_weights