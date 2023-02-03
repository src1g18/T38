import os
import re
import string
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def get_similar_movies(desc):
    # Read the movies file
    with open("movies.txt") as f:
        movie_list = f.readlines()

    # Preprocess the movie descriptions
    preprocessed_movies = []
    for movie in movie_list:
        movie = movie.lower()
        movie = re.sub(r"[{}]".format(string.punctuation), "", movie)
        preprocessed_movies.append(movie)

    # Compute the tf-idf vectors for each description
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_movies)

    # Get the cosine similarity between the target description and each movie
    desc = desc.lower()
    desc = re.sub(r"[{}]".format(string.punctuation), "", desc)
    desc_vector = tfidf_vectorizer.transform([desc])
    similarity = cosine_similarity(desc_vector, tfidf_matrix).flatten()

    # Get the index of the most similar movie
    most_similar_index = np.argmax(similarity)

    # Return the title of the most similar movie
    return movie_list[most_similar_index].strip()


# Example usage
desc = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk land on the planet Sakaar where he is sold into slavery and trained as a gladiator."
print("You might want to watch:", get_similar_movies(desc))
