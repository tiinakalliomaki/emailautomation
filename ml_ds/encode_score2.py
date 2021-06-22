from sentence_transformers import SentenceTransformer
from email_cleaning import EmailCleaning
from scipy import spatial
import pickle



"""Unpickle avergae vectpr pf thank-you vectors"""
with open('../positives_average.pickle', 'rb') as f:
    average_vector = pickle.load(f)

def get_score(input_text):
    cleaned_text=EmailCleaning.full_clean(input_text)
    model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    encoded_vector=model.encode(cleaned_text)
    score=spatial.distance.cosine(average_vector, encoded_vector)
    return score