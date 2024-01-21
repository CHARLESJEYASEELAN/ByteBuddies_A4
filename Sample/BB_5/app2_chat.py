from flask import Flask, render_template, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pandas as pd
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer

with open(r'BlissBot_TrainingData_Cleaned_4.json',encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data['intents'])
df.head(5)

dic = {"tag":[], "patterns":[], "responses":[]}
for i in range(len(df)):
    ptrns = df[df.index == i]['patterns'].values[0]
    rspns = df[df.index == i]['responses'].values[0]
    tag = df[df.index == i]['tag'].values[0]
    for j in range(len(ptrns)):
        dic['tag'].append(tag)
        dic['patterns'].append(ptrns[j])
        dic['responses'].append(rspns)
        
df = pd.DataFrame.from_dict(dic)


tokenizer = Tokenizer(lower=True, split=' ')
tokenizer.fit_on_texts(df['patterns'])
tokenizer.get_config()

vacab_size = len(tokenizer.word_index)

ptrn2seq = tokenizer.texts_to_sequences(df['patterns'])
X = pad_sequences(ptrn2seq, padding='post')

lbl_enc = LabelEncoder()
y = lbl_enc.fit_transform(df['tag'])


app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
model = load_model('Bliss_Bot_ChYa_1_0_0_t7.h5')  



@app.route('/')
def index():
    return render_template('01_BlissBot.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['userInput']
    response = generate_chat_response(user_input)
    return jsonify({'response': response})

def model_response(query): 
    text = []
    txt = re.sub('[^a-zA-Z\']', ' ', query)
    txt = txt.lower()
    
    if ' ' in txt:
        txt = txt.split()
        txt = " ".join(txt)
    
    text.append(txt)
        
    x_test = tokenizer.texts_to_sequences(text)
    x_test = np.array(x_test)
    
    if len(x_test) > 1:
        x_test = x_test.squeeze()
    
    x_test = pad_sequences(x_test, padding='post', maxlen=X.shape[1])
    y_pred = model.predict(x_test)
    y_pred = y_pred.argmax()
    tag = lbl_enc.inverse_transform([y_pred])[0]
    responses = df[df['tag'] == tag]['responses'].values[0]

    
    return random.choice(responses)


def generate_chat_response(user_input):
    response = model_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)
