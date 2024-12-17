import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = tf.keras.models.load_model("model/encryption_classifier.keras")

# Load the saved label encoder classes
label_encoder_classes = np.load("model/label_encoder.npy", allow_pickle=True)

# Load the saved tokenizer
with open("model/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Define maximum sequence length (ensure it matches your training setup)
MAX_SEQUENCE_LENGTH = 100  # Update if necessary

def test_model(cipher_text):
    try:
        # Preprocess the input
        sequence = tokenizer.texts_to_sequences([cipher_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

        # Perform prediction
        prediction = model.predict(padded_sequence)
        predicted_label = label_encoder_classes[np.argmax(prediction)]

        print(f"Cipher Text: {cipher_text}")
        print(f"Predicted Algorithm: {predicted_label}")
        print(f"Prediction Probabilities: {prediction}")

    except Exception as e:
        print(f"Error: {e}")

# Test the model with a sample input
# Example from training data
test_input = "5fd7f5f9f47e74e3db5f4213fcbd761245aeb2fd74085c68a4abc04645845ead69b6a370405490aadb6fc51f3bdf93ac"  # Expected: AES
test_model(test_input)
