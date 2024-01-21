from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import secrets
from pymongo import MongoClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(16)
analyzer = SentimentIntensityAnalyzer()


try:
    app.config['MONGO_URI'] = 'mongodb://localhost:27017/userdatabase'
    mongo = PyMongo(app)
    users_collection = mongo.db.users
except Exception as e:
    print(f"Error connecting to MongoDB: {str(e)}")
    mongo = None

@app.route('/')
def home():
    return render_template('MainPage_1.html')

@app.route('/register', methods=['POST'])
def register():
    if request.method == 'POST':
        new_username = request.form['new-username']
        new_password = request.form['new-password']

        # Check if the username already exists
        existing_user = users_collection.find_one({'username': new_username})

        if existing_user:
            return 'Username already registered. Please choose a different username.'

        # If the username doesn't exist, register the user
        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256', salt_length=8)
        users_collection.insert_one({'username': new_username, 'password': hashed_password})

        return 'Registration successful!'


    return redirect(url_for('home'))

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if the username exists
        user = users_collection.find_one({'username': username})

        if user and check_password_hash(user['password'], password):
            session['user'] = username
            # Redirect to the analysis page after successful login
            return redirect(url_for('analysis'))
        else:
            return 'Invalid username or password'

    return redirect(url_for('home'))
@app.route('/analysis')
def analysis():
    return render_template('Analysisform.html')
@app.route('/submit_form', methods=['POST'])
def submit_form():
    if request.method == 'POST':
        form_data = {
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
            'signatureDate': request.form['signatureDate']
        }

        # Insert the form data into the MongoDB collection
        # Connect to MongoDB
        client = MongoClient('mongodb://localhost:27017/')
        db = client['userdatabase']
        forms_collection = db['record']

        forms_collection.insert_one(form_data)

        return 'Form submitted successfully!'
   

@app.route('/home')
def homepage():
    return render_template('HomePage.html')
@app.route('/main_page')
def main_page():
    return render_template('MainPage_1.html')

@app.route('/bliss_activity_one', methods=['POST'])
def bliss_activity_one():
    affirmation = request.form.get('affirmation', '')

    # Check the length of the affirmation
    if len(affirmation) > 0:
        # Award one point
        points = 1
    else:
        points = 0

    try:
        # Update MongoDB with the points in both collections
        users_collection.insert_one({'affirmation': affirmation, 'points': points})
        mongo.db.points.insert_one({'bliss_activity': 'one', 'points': points})
        
        return jsonify({'points': points})
    except Exception as e:
        return jsonify({'error': f"Error updating/saving points: {str(e)}"}), 500


@app.route('/bliss_activity_two', methods=['POST'])
def bliss_activity_two():
    gratitude_message = request.form.get('gratitudeTextArea', '')

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
        # Update MongoDB with the points in both collections
        users_collection.insert_one({'gratitude_message': gratitude_message, 'points': points})
        mongo.db.points.insert_one({'bliss_activity': 'two', 'points': points})
        
        return jsonify({'points': points})
    except Exception as e:
        return jsonify({'error': f"Error updating/saving points: {str(e)}"}), 500



    
@app.route('/bliss_activity_three', methods=['POST'])
def bliss_activity_three():
    if request.method == 'POST':
        # Extract data from the form
        day_rating = int(request.form.get('day', 1))
        activities_marks = sum(int(request.form.get(activity, 0)) if request.form.get(activity) != 'on' else 0 for activity in ['bliss', 'meditation', 'sleep', 'recreational'])
        memorable_moments_marks = len(request.form.get('memorableMoments', '').split('\n'))
        act_of_kindness_marks = len(request.form.get('actOfKindness', '').split('\n'))
        problems_faced_marks = len(request.form.get('problemsFaced', '').split('\n'))
        express_yourself_marks = len(request.form.get('expressContent','').split('\n'))
        express_yourself = request.form.get('expressContent','')
        # Calculate total marks (max 10)
        total_marks = min(day_rating + activities_marks + memorable_moments_marks + act_of_kindness_marks + problems_faced_marks+express_yourself_marks, 12)

        # Update MongoDB with the points in both collections
        
        
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
        
        try:
            users_collection.insert_one({'daily_form_points': total_marks})
            mongo.db.points.insert_one({'bliss_activity': 'three', 'points': total_marks})
            return jsonify({'total_marks': total_marks})
        except Exception as e:
            return jsonify({'error': f"Error updating/saving marks: {str(e)}"}), 500




@app.route('/bliss_boat_page')
def bliss_boat_page():
    return render_template('01_BlissBot.html')

@app.route('/improviser_page')
def improviser_page():
    return render_template('02_Improviser.html')

@app.route('/resources_page')
def resources_page():
    return render_template('ResourcesPage.html')

@app.route('/contribute_page')
def contribute_page():
    return render_template('FundPage.html')
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
