import nltk
import pandas as pd

text1 = """
The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency.
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
The Rajya Sabha is a permanent House.
It cannot be dissolved. Whcn the Lok Sabha is not in session or is dissolved, the permanent house still functions.
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

# Embeddings
from tables import spacy, doc2vec, bert, universal_sentence_encoder, k_means_cluster

EMBEDDINGS = [spacy, doc2vec, bert, universal_sentence_encoder]


def run_all_embeddings(t1, t2):
    data = pd.DataFrame()
    for emb in EMBEDDINGS:
        print("Embedding: ", emb.__name__)
        data = data.append(k_means_cluster(t1, t2, NUM_CLUSTERS=3))
    data.to_csv('embeddings.csv', index=False)


# run_all_embeddings(text1, text2)

# Similarity Metric
DISTANCE_METRICS = [nltk.cluster.util.cosine_distance, nltk.cluster.util.euclidean_distance]


def run_all_simlaritymetrics(t1, t2):
    data = pd.DataFrame()
    for dist in DISTANCE_METRICS:
        print("Distance: ", dist.__name__)
        data = data.append(k_means_cluster(t1, t2, NUM_CLUSTERS=3, distance=dist))
    data.to_csv('distances.csv', index=False)

# run_all_simlaritymetrics(text1, text2)
