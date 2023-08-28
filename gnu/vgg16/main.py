from engine import *  # Import the VGGupdated model function from engine.py
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard  # Add this line to import TensorBoard
import time  # For naming the log files uniquely

# Assuming you have set up the data processing in engine.py (train_x, test_x, train_y, test_y)

# Set parameters for training
BATCH_SIZE = 32
EPOCHS = 100
VERBOSE = 1

# Create a TensorBoard callback
log_dir = f"logs/fit/{time.strftime('%Y%m%d-%H%M%S')}"
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train the model
history = model.fit(
    train_x, train_y,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    verbose=VERBOSE,
    validation_split=0.2,  # use 20% of the training data as validation data
    callbacks=[tensorboard_callback]  # Add the callback here
)

# Evaluate the model using the test data
loss, accuracy = model.evaluate(test_x, test_y, verbose=VERBOSE)
print("Test loss:", loss)
print("Test accuracy:", accuracy)

# Optionally, you can save the trained model
model.save('proto_test.h5')

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Create a new figure
plt.figure()

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save the figure to PDF
plt.savefig("Model_Accuracy.pdf")
