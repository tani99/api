import spacy
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
import nltk

pd.set_option('display.max_columns', 7)

nlp = spacy.load("en_core_web_sm")


# Utility function for generating sentence embedding from the text
def get_embeddings(text, embedding, d):
    # Change embeddings
    return embedding(text, d)


def spacy(text, d):
    return nlp(text).vector


def doc2vec(text, d):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(d.to_numpy())]
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100)
    words = nltk.word_tokenize(text.lower())
    return model.infer_vector(words)


def bert(text, example):
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    return sbert_model.encode([text])[0]

