import numpy
import math
import pandas as pd
import nltk
from nltk.corpus import stopwords
import time

from final_files.visualisation.embeddings import spacy_embedding


def cosine_distance(u, v):
    """
    Returns 1 minus the cosine of the angle between vectors v and u. This is
    equal to 1 - (u.v / |u||v|).
    """
    return 1 - (numpy.dot(u, v) / (math.sqrt(numpy.dot(u, u)) * math.sqrt(numpy.dot(v, v))))


import nltk
from nltk.corpus import stopwords
import nltk
stopwords_set = set(stopwords.words('english'))


def tokenize_remove_stopwords(list_of_sentences):
    tokens = nltk.word_tokenize(list_of_sentences)
    all_words = set(tokens)

    words_without_stopwords = all_words - stopwords_set
    return words_without_stopwords
    # text_tokens = nltk.word_tokenize(" ".join(list_of_sentences))
    # return [word for word in text_tokens if not word in stopwords.words()]


def get_cluster_similarity(cluster1, cluster2):
    # cluster1_vector = spacy_embedding(cluster1)
    # cluster2_vector = spacy_embedding(cluster2)
    #
    # return cosine_distance(cluster1_vector, cluster2_vector)
    cluster1_words = tokenize_remove_stopwords(cluster1)

    cluster2_words = tokenize_remove_stopwords(cluster2)

    return len(list(cluster1_words & cluster2_words))
