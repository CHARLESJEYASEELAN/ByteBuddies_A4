from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
model_test = load_model('Bliss_Bot_ChYa_1_0_0_t6.h5')
words = pickle.load(open('words_3.pkl', 'rb'))
classes = pickle.load(open('classes_3.pkl', 'rb'))
intents = json.loads(open('BlissBot_TrainingData.json').read())

@app.route('/')
def index():
    return render_template('./templates/01_BlissBot.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['userInput']
    response = generate_chat_response(user_input)
    return jsonify({'response': response})

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag =[0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i]=1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model_test.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x:x[1],reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    if intents_list:
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])
    return "Sorry, I didn't understand that. Can you rephrase?"

def generate_chat_response(user_input):
    intents_list = predict_class(user_input)
    response = get_response(intents_list, intents)
    return response

if __name__ == '__main__':
    app.run(debug=True)
