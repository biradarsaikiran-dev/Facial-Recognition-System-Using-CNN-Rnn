import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, UpSampling2D, GlobalAveragePooling2D, Reshape, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, clone_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

def load_data_from_csv(filepath):
    """
    Loads and preprocesses data from the fer2013.csv file.
    """
    data = pd.read_csv(filepath)
    
    # Function to parse the pixel string
    def parse_pixels(pixel_string):
        return np.array(pixel_string.split(' ')).astype('float32')

    data['pixels'] = data['pixels'].apply(parse_pixels)
    
    X = np.stack(data['pixels'].values)
    # Reshape to (num_samples, 48, 48, 1) for grayscale
    X = X.reshape(-1, 48, 48, 1).astype('float32')

    # Convert grayscale to 3-channel RGB as the model expects 3 channels
    X = np.repeat(X, 3, axis=-1)

    y = pd.get_dummies(data['emotion']).values

    # Use the 'Usage' column to split the data
    X_train = X[data['Usage'] == 'Training']
    y_train = y[data['Usage'] == 'Training']
    X_val = X[data['Usage'] == 'PublicTest']
    y_val = y[data['Usage'] == 'PublicTest']
    X_test = X[data['Usage'] == 'PrivateTest']
    y_test = y[data['Usage'] == 'PrivateTest']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

# Create the model using Transfer Learning with EfficientNetV2B0
def build_model(input_shape=(48, 48, 3), num_classes=7):
    """
    Builds a high-accuracy hybrid CNN-RNN model for emotion classification.
    - Uses EfficientNetV2B0 (pre-trained on ImageNet) as a feature extractor.
    - Upsamples input images for better performance with the pre-trained model.
    - Treats the output feature map as a sequence and feeds it to an LSTM layer.
    - Adds a classification head with a Dense layer (a simple Neural Network).
    """
    # EfficientNetV2 models perform best with larger images. We'll upsample to 96x96.
    upsampled_shape = (96, 96, 3)

    # Load the EfficientNetV2B0 model, pre-trained on ImageNet.
    # This is a powerful CNN that provides excellent features.
    base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=upsampled_shape)

    # Freeze the layers of the base model so they are not trained
    base_model.trainable = False
    base_model._name = "efficientnetv2-b0" # Explicitly name the layer

    # --- Create the new model on top ---
    inputs = Input(shape=input_shape)

    # Upsample the 48x48 images to 96x96 for the base model
    x = UpSampling2D(size=(2, 2))(inputs)

    # Get features from the base model
    x = base_model(x, training=False)

    # The output of EfficientNetV2B0 with our upsampled input is (3, 3, 1280).
    # We treat the spatial dimensions as a sequence for the RNN.
    # Reshape from (batch, 3, 3, 1280) to (batch, 9, 1280) to create a sequence of 9 steps.
    x = Reshape((-1, 1280))(x)

    # Add an RNN layer (LSTM) to process the sequence of features.
    x = LSTM(128, dropout=0.3, recurrent_dropout=0.3)(x)
    x = Dropout(0.5)(x)
    
    # Classifier part
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    return model

# Train the model
def train_model(model, X_train, y_train, X_val, y_val):
    # Callbacks for early stopping and learning rate scheduling
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False, mode='min') # Restore is handled by ModelCheckpoint
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, mode='min')

    # Calculate class weights to handle the imbalanced dataset, which improves accuracy.
    y_train_integers = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_integers),
        y=y_train_integers
    )
    class_weights_dict = dict(enumerate(class_weights))
    print("Calculated Class Weights:", class_weights)

    # Use the standard arguments for ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=15,      # degrees
        zoom_range=0.2,
        width_shift_range=0.1,  # fraction of total width
        height_shift_range=0.1) # fraction of total height

    model.compile(optimizer=Adam(learning_rate=5e-4), # Slightly lower initial learning rate
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    batch_size = 64
    initial_epochs = 50 # Max epochs for the first stage

    print("--- Starting Feature Extraction (training only the top layer) ---")
    history = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                       epochs=initial_epochs,
                       validation_data=(X_val, y_val),
                       callbacks=[early_stopping, reduce_lr, ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')],
                       class_weight=class_weights_dict)
    
    # --- Fine-Tuning Stage ---
    print("\n--- Starting Fine-Tuning (unfreezing some base model layers) ---")
    # Unfreeze the top blocks of the EfficientNetV2B0 model for fine-tuning.
    base_model = model.get_layer('efficientnetv2-b0')
    base_model.trainable = True
    for layer in base_model.layers:
        # The layers to fine-tune are typically at the end of the network.
        # We will freeze all layers except for the last few blocks.
        # For EfficientNetV2, layers in 'block6' and 'top' are good candidates for unfreezing.
        if not (layer.name.startswith('block6') or layer.name.startswith('top')):
            layer.trainable = False

    # Load the best weights from the feature extraction phase before fine-tuning
    model.load_weights('best_model.keras')

    # Re-compile the model with a very low learning rate for fine-tuning
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    fine_tune_epochs = 20 # Max epochs for fine-tuning
    total_epochs = initial_epochs + fine_tune_epochs

    history_fine = model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
                             epochs=total_epochs,
                             initial_epoch=history.epoch[-1],
                             validation_data=(X_val, y_val),
                             callbacks=[early_stopping, reduce_lr, ModelCheckpoint('best_model_fine_tuned.keras', save_best_only=True, monitor='val_loss')],
                             class_weight=class_weights_dict) # Use weights in fine-tuning too

    # You can combine histories if you want, but for plotting, the last one is fine.
    return history_fine

# Plot training history
def plot_history(history):
    """Plots accuracy and loss for training and validation sets."""
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

def detect_emotion(emotion_model, class_names):
    """
    Detects faces in real-time from a webcam, and predicts their emotion.
    Uses OpenCV's built-in Haar Cascade for minimal, file-free face detection.
    """
    # The class_names are fixed for the fer2013 dataset
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Sad", 5: "Surprised", 6: "Neutral"}


    # Use a reliable built-in Haar Cascade from OpenCV
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # Start video capture
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Convert to grayscale for the Haar Cascade detector
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            # Extract the face ROI (Region of Interest)
            face_roi = frame[y:y + h, x:x + w]
            if face_roi.size > 0:
                # The model expects a 3-channel (color) image of size 48x48
                resized_face = cv2.resize(face_roi, (48, 48))
                input_data = np.expand_dims(resized_face.astype('float32'), axis=0)
                
                prediction = emotion_model.predict(input_data, verbose=0)
                emotion = emotion_dict[np.argmax(prediction)]

                (text_width, text_height), baseline = cv2.getTextSize(emotion, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x, y - text_height - 15), (x + text_width + 10, y - 10), (0, 0, 0), -1)
                alpha = 0.6  # Transparency factor
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x + 5, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    # Use a directory path for the SavedModel format
    MODEL_PATH = 'emotion_detection_model.keras'
    DATA_PATH = 'fer2013.csv' # Assumes fer2013.csv is in your project directory
    
    force_retrain = '--retrain' in sys.argv
    # Check if the model file exists.
    if os.path.exists(MODEL_PATH) and not force_retrain:
        print("Loading existing model...")
        model = load_model(MODEL_PATH)
        class_names = None # Not needed when loading, emotion_dict is hardcoded
    else:
        if not os.path.exists(DATA_PATH):
            print(f"Error: Dataset '{DATA_PATH}' not found. Please download it and place it in the project directory.")
            return
        # Load data from CSV
        print("Loading data from CSV...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_data_from_csv(DATA_PATH)
        
        # Create and train the model
        print("Creating model...")
        model = build_model()
        model.summary()
        
        print("Training model...")
        history = train_model(model, X_train, y_train, X_val, y_val)
        
        # Plot training history
        plot_history(history)
        
        # Evaluate the model
        print("Evaluating model...")
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # Save the model
        print("Saving model...")
        model.save(MODEL_PATH)

        class_names = None # Not needed, emotion_dict is hardcoded
    
    # Start real-time detection
    print("Starting real-time detection...")
    detect_emotion(model, class_names)

if __name__ == "__main__":
    main() 
    