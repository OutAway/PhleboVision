from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import time
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

def predict_new_images(folder_path='upload/', model_path=None):
    # Load the saved model
    if model_path is None:
        model_path = input("Enter the name of the .h5 model file you want to use for test analysis: ")
    
    model = load_model(model_path)

    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        
        # Load the image
        new_image = cv2.imread(image_path)
        
        if new_image is None:
            print(f"Could not open or find the image {image_name}.")
            continue
        
        # Preprocess the image (resize, normalize, etc.)
        new_image = cv2.resize(new_image, (300, 300))
        new_image = new_image.astype('float32') / 255.0
        
        # Add an extra dimension for the batch size
        new_image = np.expand_dims(new_image, axis=0)
        
        # Make prediction
        prediction = model.predict(new_image)
        
        # Interpret prediction (assuming a binary classification problem)
        predicted_class = np.argmax(prediction)
        
        if predicted_class == 0:
            print(f"Prediction for image {image_name}: Class 1 (echinatopharynx)")
        else:
            print(f"Prediction for image {image_name}: Class 2 (atroclavata)")

def calculate_metrics(true_labels, pred_labels):
    """
    Calculate and print classification metrics including Confusion Matrix, Classification Report.
    """
    print("Confusion Matrix:")
    print(confusion_matrix(true_labels, pred_labels))
  
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels))

def predict_and_analyze(model, test_images, true_labels):
    """
    Make predictions with the model and analyze the results.
    
    Parameters:
        model (Model): The pre-trained machine learning model
        test_images (np.array): The images to classify
        true_labels (list): The ground-truth labels
    
    Returns:
        dict: A dictionary containing various metrics and data
    """
    pred_labels = []
    prediction_time = []

    for i, new_image in enumerate(test_images):
        start_time = time.time()
        prediction = model.predict(np.expand_dims(new_image, axis=0))
        end_time = time.time()

        predicted_class = np.argmax(prediction)
        pred_labels.append(predicted_class)

        prediction_time.append(end_time - start_time)

    avg_pred_time = np.mean(prediction_time)

    metrics = {
        'avg_pred_time': avg_pred_time,
        'true_labels': true_labels,
        'pred_labels': pred_labels,
        'prediction_time': prediction_time
    }
    
    return metrics

if __name__ == "__main__":
    predict_new_images()
    metrics = predict_and_analyze(model, test_images, true_labels)
    
    # Print or further process the metrics
    print(f"Average Prediction Time: {metrics['avg_pred_time']}")
    print(f"True Labels: {metrics['true_labels']}")
    print(f"Predicted Labels: {metrics['pred_labels']}")
    print(f"Individual Prediction Times: {metrics['prediction_time']}")
    calculate_metrics()
