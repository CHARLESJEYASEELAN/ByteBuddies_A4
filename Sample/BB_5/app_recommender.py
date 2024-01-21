from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import joblib
import random
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('Improviser.html')  # Update with your actual HTML file

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
if __name__ == '__main__':
    app.run(debug=True)
