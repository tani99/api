import math

from nltk import sent_tokenize

from tables import get_embeddings, spacy
from util import cosine_distance


def distance(sent1, sent2):
    emb1 = get_embeddings(sent1, spacy, None)
    emb2 = get_embeddings(sent2, spacy, None)

    return cosine_distance(emb1, emb2)

def weight(i, j):
    mod = abs(i - j)
    if mod <= 2:
        return 1
    else:
        return 1/(math.sqrt(mod-1))

def distances_in_text(text):
    tokenized = sent_tokenize(text)
    distances = {}
    for i, sent in enumerate(tokenized):
        if i >= len(tokenized) - 1:
            break
        sent1 = tokenized[i]
        sent2 = tokenized[i + 1]
        dist = distance(sent1, sent2)
        print("Sent 1: ", sent1)
        print("Sent 2: ", sent2)
        print("Distance between sentences: ", dist)
        distances[(sent1, sent2)] = dist

    print("Mininum: ", min(distances, key=distances.get))
    print("Sorted: ")
    print(dict(sorted(distances.items(), key=lambda item: item[1])))


def segment_similarity(tok1, tok2):
    sum_sentence_similarities = 0
    # tok1 = sent_tokenize(seg1)
    # tok2 = sent_tokenize(seg2)
    for i, sent_seg_1 in enumerate(tok1):
        for j, sent_seg_2 in enumerate(tok2):
            # weighted
            sum_sentence_similarities += distance(sent_seg_1, sent_seg_2) * weight(i, j)

    return sum_sentence_similarities / (len(tok1) * len(tok2))

MinSegmentSize = 2
def weakest_segments(text):
    tok = sent_tokenize(text)
    last_index = len(tok) - 1
    for i, sent, in enumerate(tok):
        if 0 == i or i == last_index:
            continue
        else:
            print(i)
            seg1 = tok[0:i]
            seg2 = tok[i:last_index]
            if len(seg1) > MinSegmentSize and len(seg2) > MinSegmentSize:
                print(seg1)
                print(seg2)
                print(segment_similarity(seg1, seg2))


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

segment1 = """The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency.
It can be dissolved earlier than its term by the President on the advice of the Prime Minister.
It can be voted out of power by a debate and vote on a no-confidence motion.
During the 13" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.
The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community.
At present, the strength of the Lok Sabha is 545. """

segment2 = """Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha.
The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individual.
The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. 
Whats More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back.
"""
# distances_in_text(text1)

# print(segment_similarity(seg1, seg2))
weakest_segments(text1)
