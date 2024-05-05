"""
Unit tests for positional encoding
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from transformer.positional_encoding import PositionalEncoding

# sentence length 50 and embedding vector layer 128
sample_pos_encoding = PositionalEncoding(50, 128)

plt.pcolormesh(sample_pos_encoding.pos_encoding.numpy()[0], cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.show()