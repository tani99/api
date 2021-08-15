from multiprocessing.dummy import freeze_support

import spacy
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from nltk.corpus import stopwords
from tables.kmeans import KMeansClusterer

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

def create_dataframe_from_text(text):
    tokenized = nltk.sent_tokenize(text)
    d = pd.DataFrame({"Text": tokenized})
    return d


def cluster_text(text, NUM_CLUSTERS, embedding, distance, create=None):
    # text to dataframe

    data = create_dataframe_from_text(text)

    # Generating sentence embedding from the text
    data['emb'] = data["Text"].apply(get_embeddings, embedding=embedding, d=data)
    matrix = distance_matrix(data)

    texts = np.array(data['Text'].tolist())
    embeddings = np.array(data['emb'].tolist())
    # X = np.array(data['Text'].tolist())
    X = embeddings #zip(embeddings, texts)
    print("X")
    print(X)
    # X = np.array(data['emb'].tolist())

    # print("features")
    # print(X)
    # print(X.shape)
    kclusterer = KMeansClusterer(
        NUM_CLUSTERS, distance=distance,
        repeats=25, avoid_empty_clusters=True)

    assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

    data['predicted'] = pd.Series(assigned_clusters, index=data.index)
    data['centroid'] = data['predicted'].apply(lambda x: kclusterer.means()[x])
    print("assigned clusters ", assigned_clusters)
    return data, assigned_clusters


def k_means_cluster(t1, t2, NUM_CLUSTERS, embedding=doc2vec, distance=nltk.cluster.util.cosine_distance, dataframe=True):
    results1, assigned_clusters1 = cluster_text(t1, NUM_CLUSTERS, embedding, distance)
    results2, assigned_clusters2 = cluster_text(t2, NUM_CLUSTERS, embedding, distance)

    # Collect clusters as arrays
    pred1 = clusters_predicted(results1, NUM_CLUSTERS)
    pred2 = clusters_predicted(results2, NUM_CLUSTERS)

    if not dataframe:
        return match_clusters(pred1, pred2)
    return match_clusters_dataframe(pred1, pred2)
    # return match_clusters_dataframe(pred1, pred2).to_json()
    # return match_clusters(pred1, pred2)

def clustering_question(data, NUM_CLUSTERS):
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


def clusters_predicted(results, NUM_CLUSTERS):
    predicted = [[] for i in range(NUM_CLUSTERS)]
    for index, row in results.iterrows():
        text = row["Text"]
        cluster_predicted = row["predicted"]
        predicted[cluster_predicted].append(text)
    return predicted


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


def match_clusters_dataframe(clusters1, clusters2):
    matches = pd.DataFrame()
    for i, cluster in enumerate(clusters1):
        sim, match = get_matching_cluster(cluster, clusters2)
        clusters2.remove(match)
        string1 = " ".join(cluster)
        string2 = " ".join(match)
        # Simplify clusters
        # string1_simplified = summarise_text(string1, simplifier='muss', percentage=1)
        # string2_simplified = summarise_text(string2, simplifier='muss', percentage=1)
        # matches = matches.append({'Text1': string1_simplified, 'Text2': string2_simplified}, ignore_index=True)
        matches = matches.append({'Text1': string1, 'Text2': string2}, ignore_index=True)
    return matches


def get_matching_cluster(cluster, all_clusters):
    max_similarity = -1
    matching_cluster = None
    for c in all_clusters:
        similarity = get_cluster_similarity(cluster, c)
        if similarity > max_similarity:
            max_similarity = similarity
            matching_cluster = c
    # print("matching cluster", matching_cluster)
    return max_similarity, matching_cluster



def tokenize_remove_stopwords(list_of_sentences):
    text_tokens = nltk.word_tokenize(" ".join(list_of_sentences))
    return [word for word in text_tokens if not word in stopwords.words()]

def get_cluster_similarity(cluster1, cluster2):

    cluster1_words = tokenize_remove_stopwords(cluster1)
    cluster2_words = tokenize_remove_stopwords(cluster2)

    # print(set(cluster1_words))
    # print(set(cluster2_words))
    # print(set(cluster1_words) & set(cluster2_words))
    # print("Similarity", len(list(set(cluster1_words) & set(cluster2_words))))
    return len(list(set(cluster1_words) & set(cluster2_words)))


#################
# Testing
#################

# Example 1

def print_score(example):
    results1, assigned_clusters = clustering_question(example)
    score1 = evaluate_clusters(results1)
    # print("Scores: ", results1)
    print("Score: ", score1)


EMBEDDINGS = [spacy, doc2vec, bert]


def run_all_embeddings(example):
    for emb in EMBEDDINGS:
        print("Embedding: ", emb.__name__)
        embedding = emb
        example['emb'] = example["Text"].apply(get_embeddings, embedding=embedding, example=example)
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
    t1['emb'] = t1["Text"].apply(get_embeddings, embedding=doc2vec, example=t1)
    t2['emb'] = t2["Text"].apply(get_embeddings, embedding=doc2vec, example=t2)
    # Cluster
    results1, assigned_clusters = clustering_question(t1)
    results2, assigned_clusters = clustering_question(t2)
    # Collect clusters as arrays
    real1, pred1 = get_clusters(results1)
    real2, pred2 = get_clusters(results2)

    return match_clusters(pred1, pred2)


text1 = """The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency.
It can be dissolved earlier than its term by the President on the advice of the Prime Minister.
It can be voted out of power by a debate and vote on a no-confidence motion.
During the 13" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.
The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community.
At present, the strength of the Lok Sabha is 545. 
Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha.
The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individual.
The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. 
Whats More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back.
"""

text2 = """
The Rajya Sabha is a permanent House. It cannot be dissolved. Whcn the Lok Sabha is not in session or is dissolved, the permanent house still functions.
However, each member of the Rajya Sabha enjoys a six-year tcrm.
Every two years one-third of its members retire by rotation. 
The total strength of the Rajya Sabha cannot be more than 250 of which 238 are elected while 12 arc nominated by the President of India. 
Election to the Rajya Sabha is done indirectly.
The members of the state legislature elect the state representatives to the Rajya Sabha in accordance with the system of proportional representation by means of a single transferable vote.
The seats in the Rajya Sabha for each state and Union Territory arc fixed on the basis of its population.
A constituency is an area demarcated for the purpose of election.
In other words, it is an area or locality with a certain number of people who choose a person to represent them in the Lok Sabha.
Each State and Union Territory is divided into territorial constituencies.The division is not based on area but on population.
Let us consider Mizoram, Rajasthan and Uttar Pradesh. Uttar Pradesh, a large state with dense population, has 80 constituencies.
"""


# pairs = match_texts(example1, example2)

# tabulated = k_means_cluster(text1, text2, NUM_CLUSTERS=3)
# tabulated.to_csv('out.csv', index=False)
# print(tabulated)
# print(tabulated.to_json())
# print(tabulate(pairs, headers=['Text 1', 'Text 2']))

def custom_distance(u, v):
    """
    Returns 1 minus the cosine of the angle between vectors v and u. This is
    equal to 1 - (u.v / |u||v|).
    """
    cosine_distance = 1 - (np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v))))
    # print(cosine_distance)
    return cosine_distance


# print(cluster_text(text1, NUM_CLUSTERS=3, embedding=spacy, distance=custom_distance))


#   1 2 3 4
# 1 0 1 2 3
# 2   0 1 2
# 3     0 1
# 4       0
def distance_matrix_old(d):
    sentences = d["Text"]
    d = d.set_index('Text')

    print("Print after resetting index")
    print(d)
    matrix = {}
    for i in range(0, len(sentences)):
        # print(d.at[sentences[i], 'emb'])
        x = sentences[i]
        matrix[x] = {}
        matrix[x][x] = 0
        distance = 1
        for j in range(0, len(sentences)):
            # y = d[(d.index == sentences[j])]['emb']
            y = sentences[j]
            matrix[x][y] = distance
            distance = distance + 1

    return matrix

def distance_matrix(d):
    print("printing data here")
    print(d)
    sentences = d["Text"]
    # d = d.set_index('Text')
    print(sentences)
    print("Print after resetting index")
    print(d)
    matrix = {}
    for i in range(0, len(sentences)):
        curr = sentences[i]
        if i == 0:
            next_elem = sentences[i + 1]
            matrix[curr] = [next_elem]
        elif i == len(sentences)-1:
            prev = sentences[i - 1]
            matrix[curr] = [prev]
        else:
            prev = sentences[i-1]
            next_elem = sentences[i + 1]
            matrix[curr] = [prev, next_elem]

    return matrix

president = """To be able to stand for election for the position of President of India a candidate must be a citizen of India, be at least 35 years of age, be qualified to be elected as a Lok Sabha member, be enrolled on the electoral list, be of sound mental health, not be an economic offender, not be a proven criminal The first President of India
and not hold an office of profit under the Government.

To stop all and sundry from standing for election for the office of President which reduces
the dignity of the office, each nominated candidate requires to have his/her name proposed
and seconded by at least 50 members of Electoral College, deposit € 15,000 as security with
the Election Commission, which the candidate will lose if he/she fails to get one-sixth of the

‘The President is elected for a term of 5 years. He/she can be re-elected for only another term.
The President continues in office till such time as his/her successor is elected and is able to take charge. On being appointed the President has to take the oath of office. It is administered to him/her by the Chief Justice of India.

‘The presidential election is conducted by the Election Commission. It is an indirect election,
in other words the people of India do not elect their President. Instead, this is done through a
system of proportional representation by means of a single transferable vote and secret ballot.
‘The President is elected by an Electoral College consisting of members of both the Houses of
Parliament and State Legislatures."""

vice_president = """A person seeking election for the post of the Vice President must be a citizen of India the post of the President, Not less than 35 years of age, Qualified to stand for election to the Rajya Sabha. Should not hold any office of profit under the union or state
government.

‘The office of the Vice President is for a term of five years. He/she can be
re-elected as many times as required. He/she can resign by submitting f 4
his/her letter of resignation to the President. He/she can also be vs
removed from office by a resolution passed in the Rajya Sabha by an
absolute majority and agreed to by the Lok Sabha. As the Chairman of
the Rajya Sabha, he/she draws a salary of € 40,000 a month.

He/she is indirectly elected by an Electoral College consisting
of members of both the Houses of Parliament by a system of
proportional representation by means of a single transferable"""
# k_means_cluster(text1, text2, NUM_CLUSTERS=3)
if __name__ == '__main__':
    freeze_support()
    # text1 = president
    # text2 = vice_president
    # text1_simplified = summarise_text(text1, simplifier="muss", percentage=1)
    # text2_simplified = summarise_text(text2, simplifier="muss", percentage=1)
    test = pd.DataFrame()
    # clusters_list = k_means_cluster(text1, text2, NUM_CLUSTERS=3, dataframe=False)

    # print("Clusters list")
    # print(clusters_list)
    # test = test.append(clusters_df)
    # this
    test = test.append(k_means_cluster(text1, text2, NUM_CLUSTERS=3))
    # test = test.append(k_means_cluster(text1_simplified, text2_simplified, NUM_CLUSTERS=3))
    # print("evaluate: " + evaluate_clusters(clusters_df))

    test.to_csv('test.csv', index=False)

    # dataframe = create_dataframe_from_text(text1)
    # print("data", dataframe)
    # print(distance_matrix(dataframe))
# data['emb'] = data["Text"].apply(get_embeddings, embedding=doc2vec, data=data)
# dist_matrix = distance_matrix(data)
# data = list(dist_matrix.items())
# an_array = np.array(data)
# print(an_array)
