import time
from tensorflow.keras.callbacks import TensorBoard

# Create a TensorBoard callback
log_dir = f"logs/fit/{time.strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
