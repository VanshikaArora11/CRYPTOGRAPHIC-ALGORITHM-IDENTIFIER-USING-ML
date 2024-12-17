import os  # Import os for directory handling
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt  # For visualization

# Load the dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    return data

# Preprocess the data
def preprocess_data(data):
    # Encode the target labels
    label_encoder = LabelEncoder()
    data["Algorithm"] = label_encoder.fit_transform(data["Algorithm"])

    # Tokenize and pad the ciphertext data
    tokenizer = Tokenizer(char_level=True)
    tokenizer.fit_on_texts(data['Ciphertext'])
    max_length = max(data["Ciphertext"].apply(len))

    # Ensure the 'model' directory exists
    os.makedirs("model", exist_ok=True)

    # Save tokenizer for later use
    with open("model/tokenizer.pkl", "wb") as file:
        pickle.dump(tokenizer, file)

    X = tokenizer.texts_to_sequences(data['Ciphertext'])
    X = pad_sequences(X, padding='post', truncating='post', maxlen=max_length)

    y = np.array(data["Algorithm"])

    return X, y, label_encoder, max_length

# Build the model
def build_model(input_length, num_classes):
    model = Sequential([
        Embedding(input_dim=256, output_dim=128, input_length=input_length),
        Bidirectional(LSTM(256, return_sequences=True)),
        LSTM(128),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Custom Keras classifier for cross-validation
class KerasClassifierWrapper(BaseEstimator):
    def __init__(self, input_length, num_classes, epochs=30, batch_size=64):
        self.input_length = input_length
        self.num_classes = num_classes
        self.epochs = epochs
        self.batch_size = batch_size

    def fit(self, X, y):
        model = build_model(self.input_length, self.num_classes)
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=0, 
                  validation_split=0.2, callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        self.model = model

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=-1)

    def score(self, X, y):
        return self.model.evaluate(X, y, verbose=0)[1]  # Return accuracy

# Main workflow
if __name__ == "__main__":
    # Load and preprocess data
    file_path = "data2.csv"  # Update with your file path
    data = load_dataset(file_path)

    # Log the class distribution
    class_distribution = data['Algorithm'].value_counts()
    print("Class Distribution:")
    print(class_distribution)

    # Visualize the class distribution
    class_distribution.plot(kind='bar')
    plt.title("Class Distribution")
    plt.xlabel("Algorithm")
    plt.ylabel("Number of Samples")
    plt.show()

    X, y, label_encoder, max_length = preprocess_data(data)

    # Save label encoder
    np.save("model/label_encoder.npy", label_encoder.classes_)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train the model
    num_classes = len(label_encoder.classes_)
    model = build_model(X.shape[1], num_classes)

    # Learning rate scheduler
    lr_scheduler = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)

    # Train the model with EarlyStopping to prevent overfitting
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=30,
        batch_size=64,
        callbacks=[lr_scheduler, EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)],
    )

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Cross-validation with custom scoring function
    print("Performing Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Wrap Keras model into sklearn's cross-validation
    keras_model = KerasClassifierWrapper(input_length=X.shape[1], num_classes=num_classes)

    cv_scores = cross_val_score(keras_model, X, y, cv=skf)
    print(f"Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

    # Save the model
    model.save("model/encryption_classifier.keras")  # Save in new Keras format
    print("Model, tokenizer, and label encoder saved.")
