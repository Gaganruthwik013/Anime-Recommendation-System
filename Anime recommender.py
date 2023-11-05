#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas scikit-learn


# In[3]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load the dataset
data = pd.read_csv('anime.csv')

# Drop rows with missing genre values
data = data.dropna(subset=['genre'])

# Preprocess the data
label_encoder = LabelEncoder()
data['genre_encoded'] = label_encoder.fit_transform(data['genre'])

# Create a TF-IDF vectorizer to convert the genre column into numerical features
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['genre'])

# Calculate the cosine similarity between anime based on genre
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Define a function to get recommendations based on genre and ratings
def get_recommendations(title, genre_encoded, n=50):
    sim_scores = list(enumerate(cosine_sim[genre_encoded]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]
    anime_indices = [i[0] for i in sim_scores]
    recommended_anime = data.iloc[anime_indices]
    recommended_anime = recommended_anime.sort_values(by='rating', ascending=False)
    return recommended_anime.head(n)

# Input genre number
genre_number = int(input("Enter the genre number (0 for Drama, 1 for Romance, 2 for Comedy, 3 for Action, 4 for Thriller, 5 for Sci-fi): "))

# Get recommendations
recommendations = get_recommendations(data.iloc[genre_number]['name'], genre_number)
print(recommendations[['name', 'rating']])


# In[ ]:




