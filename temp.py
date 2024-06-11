import tensorflow as tf

def check_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("No GPUs found. Please check your CUDA and cuDNN installation.")

check_gpu()