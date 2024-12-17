from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("model/encryption_classifier.keras")

# Load the saved label encoder classes
label_encoder_classes = np.load("model/label_encoder.npy", allow_pickle=True)

# Load the saved tokenizer
with open("model/tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)

# Define maximum sequence length (should match your training setup)
MAX_SEQUENCE_LENGTH = 100  # Ensure this matches the value during training

@app.route("/")
def index():
    return render_template("index2.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the cipher text from the request
        data = request.json
        cipher_text = data.get("cipher_text")
        if not cipher_text:
            return jsonify({"error": "Cipher text is missing"}), 400

        # Preprocess the cipher text using the loaded tokenizer
        sequence = tokenizer.texts_to_sequences([cipher_text])
        padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post")

        # Perform the prediction
        prediction = model.predict(padded_sequence)
        predicted_label = label_encoder_classes[np.argmax(prediction)]

        # Return the result as JSON
        return jsonify({"algorithm": predicted_label})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
