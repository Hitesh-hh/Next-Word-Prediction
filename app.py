from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the tokenizer and the model
tokenizer = pickle.load(open('tokenizer1.pkl', 'rb'))
model = tf.keras.models.load_model('nextword1.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    seed_text = data['seed_text']
    next_words = data.get('next_words', 1)

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word

    return jsonify({'predicted_text': seed_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
