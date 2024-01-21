import datetime
from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from pymongo import MongoClient
from datetime import datetime, timedelta
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import random
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import assemblyai as aai

aai.settings.api_key = "fa76543f131b4f7b9cd931b4b6267461"

objectID = ''

analyzer = SentimentIntensityAnalyzer()

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)

lemmatizer = WordNetLemmatizer()
model_test = load_model('Bliss_Bot_ChYa_1_0_0_t6.h5')
words = pickle.load(open('words_3.pkl', 'rb'))
classes = pickle.load(open('classes_3.pkl', 'rb'))
intents = json.loads(open('BlissBot_TrainingData.json').read())

global intentslist, intents_json

try:
    app.config['MONGO_URI'] = 'mongodb://localhost:27017/userdatabase'
    mongo = PyMongo(app)
    users_collection = mongo.db.users
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    mongo = None

@app.route('/')
def home():
    return render_template('loginpage.html')

@app.route('/home')
def homepage():
    return render_template('HomePage.html')
@app.route('/main_page')
def main_page():
    return render_template('MainPage_1.html')

@app.route('/trending_page')
def trending_page():
    return render_template('EDA.html')

@app.route('/register', methods=['POST'])
def register():
    global objectID
    if request.method == 'POST':
        new_username = request.form['new-username']
        new_password = request.form['new-password']

        # Check if the username already exists
        existing_user = users_collection.find_one({'username': new_username})

        if existing_user:
            return 'Username already registered. Please choose a different username.'

        # If the username doesn't exist, register the user
        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256', salt_length=8)
        result = users_collection.insert_one({'username': new_username, 'password': hashed_password})
        
        objectID += str(result.inserted_id)
        
        session['user_ID'] = objectID
        session['logged_in'] = True
        
        return redirect(url_for('analysis'))


    #return redirect(url_for('home'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    global objectID
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username exists
        user = users_collection.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            
            # Redirect to the analysis page after successful login
            objectID += str(user['_id'])
            session['user_ID'] = objectID
            session['logged_in'] = True
            session.modified = True
            
            return "Login Success"
        else:
            return redirect(url_for('home'))
        
@app.route('/logoutPageRoute',methods=['GET', 'POST'])        
def logoutPageRoute():
    return render_template('loginpage.html')

    
@app.route('/analysis')
def analysis():
    return render_template('Analysisform.html')

def normalize(value):
    return value / 10.0  # Normalize values to the range [0, 1]

def weightsCalculation(factor):
    temp = 0
    if 0.0 <= factor <= 0.3:
        temp = 0.15
    elif 0.4 <= factor <= 0.7:
        temp = 0.55
    elif 0.8 <= factor <= 1.0:
        temp = 0.9
    return temp

def calculate_emotional_score(depression, anxiety, sleep_disturbance, substance_intake):
    normalized_depression = normalize(depression)
    normalized_anxiety = normalize(anxiety)
    normalized_sleep = normalize(sleep_disturbance)
    normalized_substance = normalize(substance_intake)
    
    weight_depression = weightsCalculation(normalized_depression)
    weight_sleep = weightsCalculation(normalized_sleep)
    weight_anxiety = weightsCalculation(normalized_anxiety)
    weight_substance = weightsCalculation(normalized_substance)

    emotional_score = (weight_depression * normalized_depression +weight_sleep * normalized_sleep +weight_anxiety * normalized_anxiety + weight_substance * normalized_substance) / (weight_depression+ weight_sleep + weight_anxiety + weight_substance)

    return emotional_score

@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        
        depression = float(request.form['depression'])
        anxiety = float(request.form['anxiety'])
        sleepDisturbance = float(request.form['sleepDisturbance'])
        substanceUse = float(request.form['substanceUse'])
        
        emotional_score = calculate_emotional_score(depression, anxiety, sleepDisturbance, substanceUse)
        
        form_data = {
            'user_ID':objectID,
            'fullName': request.form['fullName'],
            'dob': request.form['dob'],
            'gender': request.form['gender'],
            'address': request.form['address'],
            'phoneNumber': request.form['phoneNumber'],
            'emergencyContact': request.form['emergencyContact'],
            'reasonForAssessment': request.form['reasonForAssessment'],
            'durationOfSymptoms': request.form['durationOfSymptoms'],
            'currentMedications': request.form['currentMedications'],
            'pastMedicalHistory': request.form['pastMedicalHistory'],
            'familyHistory': request.form['familyHistory'],
            'educationLevel': request.form['educationLevel'],
            'occupation': request.form['occupation'],
            'livingSituation': request.form['livingSituation'],
            'relationships': request.form['relationships'],
            'symptomsDescription': request.form['symptomsDescription'],
            'depression': request.form['depression'],
            'anxiety': request.form['anxiety'],
            'sleepDisturbance': request.form['sleepDisturbance'],
            'substanceUse': request.form['substanceUse'],
            'emotional_score': emotional_score,
            'otherSymptoms': request.form['otherSymptoms'],
            'suicidalIdeation': request.form['suicidalIdeation'],
            'previousTreatments': request.form['previousTreatments'],
            'stressors': request.form['stressors'],
            'culturalReligious': request.form['culturalReligious'],
            'legalEthical': request.form['legalEthical'],
            'clinicalObservations': request.form['clinicalObservations'],
            'recommendedInterventions': request.form['recommendedInterventions'],
            'followUpPlan': request.form['followUpPlan'],
            'patientSignature': request.form['patientSignature'],
            'signatureDate': request.form['signatureDate'],
            
        }

        # Insert the form data into the MongoDB collection
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['userdatabase']
        forms_collection = db['record']

        forms_collection.insert_one(form_data)

        return 'Form submitted successfully!'
   

@app.route('/aboutUs')
def aboutUs():
    return render_template('aboutus.html')


# -------------------------------- Main and HOME PAGE -----------------------------------------


#blissactivity one
@app.route('/bliss_activity_one', methods=['POST'])
def bliss_activity_one():
    # Get the user's session ID or create a new one
    #session_id = request.cookies.get('session_id') or str(ObjectId())
      # Print the ObjectId for debugging purposes
    #print(f"Received session_id: {session_id}")
    now = datetime.utcnow()
    now = now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    
    
    affirmation = request.form.get('affirmation', '')
    
    try:
        mongo.db.affCounts.update_one(
            {'user_ID': objectID},
            {
                '$inc': {'affirmation_count': 1},
                '$set': {
                    'affirmation': affirmation,
                    'last_submission': datetime.utcnow()
                }
            },
            upsert=True
        )
    except Exception as e:
        return jsonify({'error': f"Error updating/saving points for Affirmation Count: {str(e)}"}), 500

    # Check if the user has already submitted within the last 24 hours
    user_data = mongo.db.points.find_one({'user_ID': objectID, 'last_submission': {'$gte': now}})

    if user_data:
        return jsonify({'error': 'User already submitted within the last 24 hours for the points cycle, How ever, you can fill affirmation still to improvise your mood'}), 400

    # Check the length of the affirmation
    if len(affirmation) > 0:
        # Award one point
        points = 1
    else:
        points = 0

    try:
        # Update MongoDB with the points in both collections using the session_id
        
        
        mongo.db.points.insert_one({
            'user_ID': objectID, 
            'bliss_activity': 'one', 
            'points': points,
            'last_submission': datetime.utcnow()
        })
        
        # Set the session_id as a cookie for the user
        response = jsonify({'points': points})
        #response.set_cookie('session_id', session_id)
        
    except Exception as e:
        return jsonify({'error': f"Error updating/saving points: {str(e)}"}), 500
    return render_template('HomePage.html')




#blissactivity two
@app.route('/bliss_activity_two', methods=['POST'])
def bliss_activity_two():
   # Get the user's session ID or create a new one
    #session_id = request.cookies.get('session_id') or str(ObjectId())
  # Print the ObjectId for debugging purposes
    #print(f"Received session_id: {session_id}")
    gratitude_message = request.form.get('gratitudeTextArea', '')

    # Check if the user has already submitted within the last 24 hours for bliss_activity_two
    user_data = mongo.db.points.find_one({'user_ID': objectID, 'last_submission_two': {'$gt': datetime.utcnow() - timedelta(days=1)}})

    if user_data:
        return jsonify({'error': 'User already submitted for bliss_activity_two within the last 24 hours'}), 400

    # Check the length of the gratitude message
    if len(gratitude_message) == 0:
        # No points if the message is empty
        points = 0
    elif len(gratitude_message) >= 5:
        # Award two points if the message is at least 5 characters
        points = 2
    else:
        # No points if the message is less than 5 characters
        points = 0

    try:
        # Update MongoDB with the points in both collections using the session_id for bliss_activity_two
        
        mongo.db.points.insert_one({
            'user_ID': objectID, 
            'bliss_activity': 'two', 
            'points': points,
            'last_submission_two': datetime.utcnow()
        })
        
        # Set the session_id as a cookie for the user
        response = jsonify({'points': points})
        #response.set_cookie('session_id', session_id)
        return response
    except Exception as e:
        return jsonify({'error': f"Error updating/saving points: {str(e)}"}), 500






    
#bliss activity three 
@app.route('/bliss_activity_three', methods=['POST'])
def bliss_activity_three():
    if request.method == 'POST':
        # Get the user's session ID or create a new one
        #session_id = request.cookies.get('session_id') or str(ObjectId())
        # Print the ObjectId for debugging purposes
        #print(f"Received session_id: {session_id}")
        # Check if the user has already submitted within the last 24 hours
        user_data = mongo.db.points.find_one({'user_ID': objectID, 'last_submission_three': {'$gt': datetime.utcnow() - timedelta(days=1)}})

        if user_data:
            return jsonify({'error': 'User already submitted for bliss_activity_three within the last 24 hours'}), 400

        # Extract data from the form
        day_rating = int(request.form.get('day', 1))
        activities_marks = sum(int(request.form.get(activity, 0)) if request.form.get(activity) != 'on' else 0 for activity in ['bliss', 'meditation', 'sleep', 'recreational'])
        memorable_moments_marks = len(request.form.get('memorableMoments', '').split('\n'))
        act_of_kindness_marks = len(request.form.get('actOfKindness', '').split('\n'))
        problems_faced_marks = len(request.form.get('problemsFaced', '').split('\n'))
        express_yourself_marks = len(request.form.get('expressContent', '').split('\n'))
        express_yourself = request.form.get('expressContent','')

        # Calculate total marks (max 10)
        total_marks = min(day_rating + activities_marks + memorable_moments_marks + act_of_kindness_marks + problems_faced_marks + express_yourself_marks, 12)
        
        try:
        # Get text from the POST request
            

        # Perform sentiment analysis using VADER
            sentiment_scores = analyzer.polarity_scores(express_yourself)

        # Return the sentiment scores as a dictionary
            result = {
                'positive': sentiment_scores['pos'],
                'negative': sentiment_scores['neg'],
                'neutral': sentiment_scores['neu']
            }


        except Exception as e:
            return jsonify({'error': str(e)})

        # Update MongoDB with the points in both collections
        try:
            
            mongo.db.points.insert_one({
                'user_ID': objectID, 
                'bliss_activity': 'three', 
                'points': total_marks,
                'positive': result['positive'],
                'negative': result['negative'],
                'neutral': result['neutral'],
                'last_submission_three': datetime.utcnow()
            })
            # Set the session_id as a cookie for the user
            response = jsonify({'total_marks': total_marks})
            #response.set_cookie('session_id', session_id)
            return response
        except Exception as e:
            return jsonify({'error': f"Error updating/saving marks: {str(e)}"}), 500


import os
from datetime import datetime
import time

@app.route('/bliss_activity_four', methods=['POST'])
def bliss_activity_four():
    # Check if the 'audio' file is in the request
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']

    # Create an 'uploads' directory if it doesn't exist
    uploads_dir = 'uploads'
    if not os.path.exists(uploads_dir):
        os.makedirs(uploads_dir)

    # Generate a unique filename using timestamp
    unique_filename = f"received_audio_{int(time.time())}.wav"
    
    audio_path = os.path.join(uploads_dir, unique_filename)

    # Save the audio file locally
    audio_file.save(audio_path)

    # Transcribe audio using AssemblyAI
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_path)

    # Perform sentiment analysis using VADER
    sentiment_analyzer = SentimentIntensityAnalyzer()
    sentiment_scores_audio = sentiment_analyzer.polarity_scores(transcript.text)
    
    user_data = mongo.db.points.find_one({'user_ID': objectID, 'last_submission_four': {'$gt': datetime.utcnow() - timedelta(days=1)}})
    
    if user_data:
        return jsonify({'error': 'User already submitted for bliss_activity_three within the last 24 hours'}), 400
    
    try:
        # Get text from the POST request
            

        # Perform sentiment analysis using VADER
            

        # Return the sentiment scores as a dictionary
            result = {
                'positive': sentiment_scores_audio['pos'],
                'negative': sentiment_scores_audio['neg'],
                'neutral': sentiment_scores_audio['neu']
            }


    except Exception as e:
        return jsonify({'error': str(e)})

        # Update MongoDB with the points in both collections
    try:
            
        mongo.db.points.insert_one({
            'user_ID': objectID, 
            'bliss_activity': 'four',
            'positive': result['positive'],
            'negative': result['negative'],
            'neutral': result['neutral'],
            'last_submission_four': datetime.now()
        })
            # Set the session_id as a cookie for the user
        return render_template('Congratulations_Page.html')
    except Exception as e:
        return jsonify({'error': f"Error updating/saving marks: {str(e)}"}), 500
    
    
'''@app.route('/bliss_activity_three', methods=['POST'])
def bliss_activity_three():
    if request.method == 'POST':
        # Get the user's session ID or create a new one
        #session_id = request.cookies.get('session_id') or str(ObjectId())
        # Print the ObjectId for debugging purposes
        #print(f"Received session_id: {session_id}")
        # Check if the user has already submitted within the last 24 hours
        user_data = mongo.db.points.find_one({'user_ID': objectID, 'last_submission_three': {'$gt': datetime.utcnow() - timedelta(days=1)}})

        if user_data:
            return jsonify({'error': 'User already submitted for bliss_activity_three within the last 24 hours'}), 400

        # Extract data from the form
        day_rating = int(request.form.get('day', 1))
        activities_marks = sum(int(request.form.get(activity, 0)) if request.form.get(activity) != 'on' else 0 for activity in ['bliss', 'meditation', 'sleep', 'recreational'])
        memorable_moments_marks = len(request.form.get('memorableMoments', '').split('\n'))
        act_of_kindness_marks = len(request.form.get('actOfKindness', '').split('\n'))
        problems_faced_marks = len(request.form.get('problemsFaced', '').split('\n'))
        express_yourself_marks = len(request.form.get('expressContent', '').split('\n'))
        express_yourself = request.form.get('expressContent','')

        # Calculate total marks (max 10)
        total_marks = min(day_rating + activities_marks + memorable_moments_marks + act_of_kindness_marks + problems_faced_marks + express_yourself_marks, 12)
        
        try:
        # Get text from the POST request
            

        # Perform sentiment analysis using VADER
            sentiment_scores = analyzer.polarity_scores(express_yourself)

        # Return the sentiment scores as a dictionary
            result = {
                'positive': sentiment_scores['pos'],
                'negative': sentiment_scores['neg'],
                'neutral': sentiment_scores['neu']
            }


        except Exception as e:
            return jsonify({'error': str(e)})

        # Update MongoDB with the points in both collections
        try:
            
            mongo.db.points.insert_one({
                'user_ID': objectID, 
                'bliss_activity': 'three', 
                'points': total_marks,
                'positive': result['positive'],
                'negative': result['negative'],
                'neutral': result['neutral'],
                'last_submission_three': datetime.utcnow()
            })
            # Set the session_id as a cookie for the user
            response = jsonify({'total_marks': total_marks})
            #response.set_cookie('session_id', session_id)
            return response
        except Exception as e:
            return jsonify({'error': f"Error updating/saving marks: {str(e)}"}), 500

    # Return sentiment scores
    return jsonify({
        'transcript': transcript.text,
        'sentiment_scores': sentiment_scores
    })'''

    



# ------------------------------------- BOT PAGE ---------------------------------------------------------------------

@app.route('/bliss_boat_page')
def bliss_boat_page():
    return render_template('01_BlissBot.html')

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
    chatSentiScores = analyzer.polarity_scores(response)
    lstScores = [chatSentiScores['pos'], chatSentiScores['neu'], chatSentiScores['neg']]

    user_object_id = session.get('user_ID')

    emotionalScore = mongo.db.record.find_one(
        {'user_ID': user_object_id},
        projection={'emotional_score': 1}
    )
    eScore = emotionalScore['emotional_score']
    
    print(eScore)

    t = 0

    if 0.0 <= eScore <= 0.3:
        t = 0
    elif 0.4 <= eScore <= 0.7:
        t = 1
    elif 0.8 <= eScore <= 1.0:
        t = 2
        
    ie = 0

    while ie <= 6:
        if t == 2:
            if lstScores[0] > lstScores[1] and lstScores[0] > lstScores[2]:
                return response
            else:
                response = get_response(intents_list, intents)
        elif t == 1:
            if lstScores[0] > lstScores[2] and lstScores[1] > lstScores[2]:
                return response
            else:
                response = get_response(intents_list, intents)
        else:
            if lstScores[0] > lstScores[2] and lstScores[1] > lstScores[2]:
                return response
            else:
                response = get_response(intents_list, intents)
        ie += 1
    else:
        return response 



# ----------------------------------------------- Improviser Page -------------------------------------------------

@app.route('/improviser_page')
def improviser_page():
    return render_template('Improviser.html')

#Recommendation Section Begins Here

# Load your music recommendation model
knn_model = joblib.load('ChYa_Music_Recommender_1_0_0.joblib')
book_model = joblib.load('model.pkl')

# Assuming df_user_input is the DataFrame for user input features
# Replace this with your actual DataFrame and preprocessing logic
df_user_input = pd.read_csv('Modified_Data.csv')
df_music = pd.read_csv('music_dataset.csv')

df = pd.read_csv('BooksByPopularity.csv')

new_df=df[df['User-ID'].map(df['User-ID'].value_counts()) > 200]  # Drop users who vote less than 200 times.
users_pivot=new_df.pivot_table(index=["User-ID"],columns=["Book-Title"],values="Book-Rating")
users_pivot.fillna(0,inplace=True)

def users_choice(id):
    
    users_fav=new_df[new_df["User-ID"]==id].sort_values(["Book-Rating"],ascending=False)[0:5]
    return users_fav

def user_based(new_df,id):
    if id not in new_df["User-ID"].values:
        print("❌ User NOT FOUND ❌")
        
        
    else:
        index=np.where(users_pivot.index==id)[0][0]
        similarity=cosine_similarity(users_pivot)
        similar_users=list(enumerate(similarity[index]))
        similar_users = sorted(similar_users,key = lambda x:x[1],reverse=True)[0:5]
    
        user_rec=[]
    
        for i in similar_users:
                data=df[df["User-ID"]==users_pivot.index[i[0]]]
                user_rec.extend(list(data.drop_duplicates("User-ID")["User-ID"].values))
        
    return user_rec

def common(new_df,user,user_id):
    x=new_df[new_df["User-ID"]==user_id]
    recommend_books=[]
    user=list(user)
    for i in user:
        y=new_df[(new_df["User-ID"]==i)]
        books=y.loc[~y["Book-Title"].isin(x["Book-Title"]),:]
        books=books.sort_values(["Book-Rating"],ascending=False)[0:5]
        recommend_books.extend(books["Book-Title"].values)
        
    return recommend_books[0:5]
def get_user_recommendations(user_input, num_recommendations=5):
    # Create a DataFrame for the user input
    user_df = pd.DataFrame([user_input], columns=df_user_input.columns)
    numerical_features = ['valence', 'danceability', 'energy']
    user_df[numerical_features] = (user_df[numerical_features] - df_user_input[numerical_features].mean()) / (df_user_input[numerical_features].max() - df_user_input[numerical_features].min())
    user_df = user_df.fillna(0)
    print(user_df)
    # Find the nearest neighbors to the user input
    _, indices = knn_model.kneighbors(user_df.values, n_neighbors=num_recommendations+1)  # +1 to exclude the user itself

    # Get recommended tracks
    recommended_tracks = pd.DataFrame(df_music.iloc[indices[0][1:]])
    return recommended_tracks

@app.route('/get_music_recommendations', methods=['POST'])
def get_music_recommendations():
    genre = request.form['genre']
    user_input = {
        'valence': float(request.form['valence']),
        'danceability': float(request.form['danceability']),
        'energy': float(request.form['energy']),
        genre: 1
    }

    recommendations = get_user_recommendations(user_input)

    return render_template('Improviser.html', music_recommendations=recommendations.to_dict('records'))
def popular_books():
    user_id=random.choice(new_df["User-ID"].values)
    user_choice_df=pd.DataFrame(users_choice(user_id))
    user_favorite=users_choice(user_id)
    n=len(user_choice_df["Book-Title"].values)
    print("🟦 USER: {} ".format(user_id))
    


    user_based_rec=user_based(new_df,user_id)
    books_for_user=common(new_df,user_based_rec,user_id)
    books_for_userDF=pd.DataFrame(books_for_user,columns=["Book-Title"])
    return books_for_userDF

@app.route('/get_popular_books', methods=['POST'])
def get_popular_books():
    # Example: Get popular books
    popular_books_result = popular_books()
    return render_template('Improviser.html', popular_books=popular_books_result.to_dict('records'))
    

#-------------------------------------------------------------Rest of the PAGE -------------------------------------------------------


@app.route('/resources_page')
def resources_page():
    return render_template('ResourcesPage.html')

@app.route('/contribute_page')
def contribute_page():
    return render_template('FundPage.html')


#------------------------------------------------------------DASH BOARD --------------------------------------------------------------------------------
import plotly.express as px
import plotly

@app.route('/graphs',methods=['GET', 'POST'],endpoint='graphs')
def barGraph():
    pass

@app.route('/dashboard')
def dashboard():
    global objectID
    # Retrieve user's Object ID from session
    user_object_id = objectID
    print(user_object_id)
    
    # Query MongoDB to get data for all bliss activities
    user_data_one = list(mongo.db.points.find(
        {'user_ID': user_object_id, 'bliss_activity': 'one'},
        projection={'last_submission': 1, 'points': 1}
    ).limit(5))
    
    print("User Data_1", user_data_one)
    
    
    
    dates_one = [item['last_submission'] for item in user_data_one]
    points_one = [item['points'] for item in user_data_one]
    dates_one = pd.to_datetime(dates_one).date
    
    print("Total Dates_1: ",dates_one)
    print("Total Points_1: ",points_one)
    
    user_data_two = list(mongo.db.points.find(
        {'user_ID': user_object_id, 'bliss_activity': 'two'},
        projection={'last_submission_two': 1, 'points': 1}
    ).limit(5))
    
    
    print("User Data_2", user_data_two)
    
    
    
    dates_two = [i['last_submission_two'] for i in user_data_two]
    points_two = [i['points'] for i in user_data_two]
    dates_two = pd.to_datetime(dates_two).date
    
    print("Total Dates_2: ",dates_two)
    print("Total Points_2: ",points_two)
    
    user_data_three = list(mongo.db.points.find(
        {'user_ID': user_object_id, 'bliss_activity': 'three'},
        projection={'last_submission_three': 1, 'points': 1}
    ).limit(5))
    print("User Data Three: ",user_data_three)
    
    dates_three = [j['last_submission_three'] for j in user_data_three]
    points_three = [j['points'] for j in user_data_three]
    dates_three = pd.to_datetime(dates_three).date
    
    print("Total Dates_3: ",dates_three)
    print("Total Points_3: ",points_three)
    
    #-------------Dataframe MERGING CODE -----------------
    df_1 = pd.DataFrame({'Dates': dates_one, 'Points_1': points_one})
    df_2 = pd.DataFrame({'Dates': dates_two, 'Points_2': points_two})
    df_3 = pd.DataFrame({'Dates': dates_three, 'Points_3': points_three})
    
    # Merge DataFrames on 'Dates'
    merged_df = pd.merge(df_1, df_2, on='Dates', how='outer').merge(df_3, on='Dates', how='outer')

    # Fill NaN values with 0
    merged_df = merged_df.fillna(0)

    # Sum the points for each date
    merged_df['Total_Points'] = merged_df['Points_1'] + merged_df['Points_2'] + merged_df['Points_3']

    print(list(merged_df['Total_Points']))
    # Display the final DataFrame
    final_Bar_Df = merged_df[['Dates','Total_Points']]
    
    
    
    sentimentData = list(mongo.db.points.find(
        {'user_ID':user_object_id,'bliss_activity':'four'},
        projection = {'last_submission_four': 1,'positive':1,'negative':1,'neutral':1}
    ).limit(5))
    
    print("Sentiment Data", sentimentData)
    # Convert MongoDB data to DataFrame for easier manipulation
    df_one = pd.DataFrame([user_data_one]) if user_data_one else pd.DataFrame()
    df_two = pd.DataFrame([user_data_two]) if user_data_two else pd.DataFrame()
    df_three = pd.DataFrame([user_data_three]) if user_data_three else pd.DataFrame()
    df_sentiment = pd.DataFrame([sentimentData]) if sentimentData else pd.DataFrame()
    print(df_sentiment)
    # Concatenate DataFrames for all bliss activities
    frames = [df_one, df_two, df_three]
    points_df = pd.concat(frames, ignore_index=True)
    #print(points_df)
    # Aggregate points based on the date
    #aggregated_points_df = points_df.groupby('last_submission')['points'].sum().reset_index()
    #print(aggregated_points_df)
    
    #print(df_points_for_the_day.head())
    # Create Plotly visualization for Bar Graph (Date vs Points)
    bar_fig = px.bar(final_Bar_Df, x='Dates', y='Total_Points', title='Date vs Points')
    graph_html = bar_fig.to_html(full_html = False)
    
    #-------------------------PIE CHART-----------------------------------
    
    # Convert the list of dictionaries to a DataFrame
    sentimentDf = pd.DataFrame(sentimentData)
    print('Sentiment Data')

    
    # Filter data for the present day
    present_day_data = sentimentDf[sentimentDf['last_submission_four'].dt.date == datetime.now().date()]
    print("Present", present_day_data)

    # Extract positive, negative, and neutral scores for the present day
    present_day_scores_df = present_day_data[['positive', 'negative', 'neutral']].transpose().reset_index()
    present_day_scores_df.columns = ['Scores', 'Points']
    
    pieFig = px.pie(present_day_scores_df,names='Scores', values='Points', title="Today's Mood")
    pie_html = pieFig.to_html(full_html=False)
    
    #----------------------------- Line Graph --------------------------------
    
    positive_scores_df = sentimentDf[['last_submission_four', 'positive']]
    positive_scores_df.columns = ['Dates', 'Positive score']

    positive_scores_df['Dates'] = positive_scores_df['Dates'].dt.date
    
    lineFig = px.line(positive_scores_df,x='Dates',y='Positive score',markers=True)
    line_html = lineFig.to_html(full_html=False)
    
    #bar_fig_json = json.dumps(bar_fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    
    
    #--------------------------------------- Affirmation Bar Graph-----------------------------------------------------------------------
    
    affirmation_data = list(mongo.db.affCounts.find(
        {'user_ID': user_object_id},
        projection={'affirmation_count': 1, 'last_submission': 1, '_id': 0}
    ))
    
    affirmationData = pd.DataFrame(affirmation_data)
    
    affirmationData['Date'] = affirmationData['last_submission'].dt.date

    # Keep only the necessary columns
    affirmationData = affirmationData[['Date', 'affirmation_count']]

    # Rename the 'affirmation_count' column to 'Count'
    affirmationData.columns = ['Date', 'Count']
    
    
    
    print("Affirmation Data", affirmationData)
    
    affirmationBar = px.bar(affirmationData, x='Date', y='Count', title='Date vs Affirmations')
    graph_html_affirmation = affirmationBar.to_html(full_html = False)
    
    return render_template('DashBoard.html',graph_html=graph_html,pie=pie_html,line=line_html,affirmation=graph_html_affirmation)

    # Render the dashboard template with the Plotly figure
    


from flask import redirect, url_for

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    global objectID
 # Clear the session or perform any necessary cleanup
    session.clear()
    objectID = ''
    session.pop('user_ID', None)
    session['logged_in'] = False
    session.modified = True
    return redirect(url_for('logoutPageRoute')) # replace 'login' with the name of your login route function

if __name__ == '__main__':
    app.run(debug=True)
