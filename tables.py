import spacy
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
import tensorflow_hub as hub
import numpy as np
from nltk.cluster import KMeansClusterer
import nltk
from tabulate import tabulate

pd.set_option('display.max_columns', 7)

nlp = spacy.load("en_core_web_sm")
# data = pd.read_csv('sentence_clustering.csv')

example1 = pd.read_csv('example1.csv')
example2 = pd.read_csv('example2.csv')


# print(example1)
# print(example2)


# default
# embedding = spacy

# Utility function for generating sentence embedding from the text
def get_embeddinngs(text, embedding, example):
    # Change embeddings
    return embedding(text, example)


def spacy(text, example):
    return nlp(text).vector


def doc2vec(text, example):
    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(example['Text'].to_numpy())]
    model = Doc2Vec(tagged_data, vector_size=20, window=2, min_count=1, epochs=100)
    words = nltk.word_tokenize(text.lower())
    return model.infer_vector(words)


def bert(text, example):
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    return sbert_model.encode([text])[0]


module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_model = hub.load(module_url)


def universal_sentence_encoder(text, example):
    return use_model([text])[0]


def clustering_question(data, NUM_CLUSTERS=3):
    # Generating sentence embedding from the text

    sentences = data['Text']

    X = np.array(data['emb'].tolist())

    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance,
        repeats=25, avoid_empty_clusters=True)

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    data['predicted'] = pd.Series(assigned_clusters, index=data.index)
    data['centroid'] = data['predicted'].apply(lambda x: kclusterer.means()[x])

    return data, assigned_clusters


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')


def get_clusters(results):
    real = [[], [], []]
    predicted = [[], [], []]
    for index, row in results.iterrows():
        text = row["Text"]
        cluster_real = row["Real"]
        cluster_predicted = row["predicted"]
        real[cluster_real].append(text)
        predicted[cluster_predicted].append(text)
    print(real, predicted)
    return real, predicted


def evaluate_clusters(results):
    real, predicted = get_clusters(results)

    similarities = []
    for real_cluster in real:
        max_similarity, matching_cluster = get_matching_cluster(real_cluster, predicted)

        # max_similarity = -1
        # matching_cluster = None
        # for predicted_cluster in predicted:
        #     similarity = get_cluster_similarity(real_cluster, predicted_cluster)
        #     if similarity > max_similarity:
        #         max_similarity = similarity
        #         matching_cluster = predicted_cluster
        similarities.append(max_similarity)

    # print(results[["Real", "predicted"]])
    # print("Similarities: ", similarities)
    # score is average of similarities
    return np.average(similarities)


def match_clusters(clusters1, clusters2):
    matches = []
    for cluster in clusters1:
        sim, match = get_matching_cluster(cluster, clusters2)
        clusters2.remove(match)
        string1 = " ".join(cluster)
        string2 = " ".join(match)
        matches.append([string1, string2])
    return matches


def get_matching_cluster(cluster, all_clusters):
    max_similarity = -1
    matching_cluster = None
    for c in all_clusters:
        similarity = get_cluster_similarity(cluster, c)
        if similarity > max_similarity:
            max_similarity = similarity
            matching_cluster = c
    print("matching cluster", matching_cluster)
    return max_similarity, matching_cluster


def get_cluster_similarity(cluster1, cluster2):
    return len(list(set(cluster1) & set(cluster2)))


#################
# Testing
#################

# Example 1

def print_score(example):
    results1, assigned_clusters = clustering_question(example)
    score1 = evaluate_clusters(results1)
    # print("Scores: ", results1)
    print("Score: ", score1)


EMBEDDINGS = [spacy, doc2vec, bert, universal_sentence_encoder]


def run_all_embeddings(example):
    for emb in EMBEDDINGS:
        print("Embedding: ", emb.__name__)
        embedding = emb
        example['emb'] = example["Text"].apply(get_embeddinngs, embedding=embedding, example=example)
        # example['emb'] = example['Text'].apply(get_embeddinngs(embedding=embedding))
        print_score(example)


# print("Example 1")
# run_all_embeddings(example1)
# print("Example 2")
# run_all_embeddings(example2)

########################

# def get_most_similar_cluster():

# Format: (["", ""], ["", ""])
def match_texts(t1, t2):
    # Get embeddings
    t1['emb'] = t1["Text"].apply(get_embeddinngs, embedding=doc2vec, example=t1)
    t2['emb'] = t2["Text"].apply(get_embeddinngs, embedding=doc2vec, example=t2)
    # Cluster
    results1, assigned_clusters = clustering_question(t1)
    results2, assigned_clusters = clustering_question(t2)
    # Collect clusters as arrays
    real1, pred1 = get_clusters(results1)
    real2, pred2 = get_clusters(results2)

    return match_clusters(pred1, pred2)


text1 = """The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13th Lok Sabha, Bhartiya Janata Party lost a no-confidence motion by one vote and had to resign. 
The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545. """

text2 = """The Rajya Sabha is a permanent House. It cannot be dissolved. When the Lok Sabha is not in session or is 
dissolved, the permanent house still functions. However, each member of the Rajya Sabha enjoys a six-year tcrm. Every 
two years one-third of its members retire by rotation. The total strength of the Rajya Sabha cannot be more than 250 
of which 238 are elected while 12 arc nominated by the President of India. """

pairs = match_texts(example1, example2)

print(tabulate(pairs, headers=['Text 1', 'Text 2']))
