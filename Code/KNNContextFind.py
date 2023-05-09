from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


path = 'NewsQA_SPAN.feather'

df = pd.read_feather(path)


# vectorize text data using TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['question'])

# train KNN model with cosine similarity metric
k = 20
knn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine')
knn.fit(X)


def predict_context(question):
    question_vect = vectorizer.transform([question])
    distances, indices = knn.kneighbors(question_vect)
    top_k_contexts = df['paragraph'].iloc[indices[0]]
    avg_similarities = top_k_contexts.apply(lambda x: cosine_similarity(question_vect, vectorizer.transform([x]))[0][0]).values
    top_context = top_k_contexts.iloc[avg_similarities.argmax()]
    return top_context