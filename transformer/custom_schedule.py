"""
Custom Schedule for learning rate
"""

import tensorflow as tf
import matplotlib.pyplot as plt

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()
        # self.d_model = d_model
        # self.d_model = tf.cast(self.d_model, tf.float32)\
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        
        arg1 = tf.math.rsqrt(step)  # 1 / sqrt(step)
        arg2 = step * (self.warmup_steps**-1.5) # step / (warmup_steps ^ 1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
    
if __name__ == '__main__':
    sample_learning_rate = CustomSchedule(d_model=128)

    plt.plot(sample_learning_rate(tf.range(200000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()
