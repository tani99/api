import numpy
import math
import pandas as pd
import nltk
from nltk.corpus import stopwords

def cosine_distance(u, v):
    """
    Returns 1 minus the cosine of the angle between vectors v and u. This is
    equal to 1 - (u.v / |u||v|).
    """
    return 1 - (numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v))))

def tokenize_remove_stopwords(list_of_sentences):
    text_tokens = nltk.word_tokenize(" ".join(list_of_sentences))
    return [word for word in text_tokens if not word in stopwords.words()]

def get_cluster_similarity(cluster1, cluster2):

    cluster1_words = tokenize_remove_stopwords(cluster1)
    cluster2_words = tokenize_remove_stopwords(cluster2)

    return len(list(set(cluster1_words) & set(cluster2_words)))