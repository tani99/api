import math
import sys
from copy import copy
from multiprocessing.dummy import freeze_support

from nltk import sent_tokenize

from final_files.simplification.muss_simp.muss_simplification import simplify_muss
from final_files.summarisation.evaluate_edmundsons import rouge_score
from final_files.util import sentence_tokenizer
from final_files.visualisation.embeddings import get_embeddings, spacy_embedding, bert, doc2vec
from final_files.visualisation.util import cosine_distance, get_cluster_similarity
import pandas as pd


def distance(sent1, sent2, embedding):
    emb1 = get_embeddings(sent1, embedding, None)
    emb2 = get_embeddings(sent2, embedding, None)

    return cosine_distance(emb1, emb2)


def weight(i, j):
    mod = abs(i - j)
    if mod <= 2:
        return 1
    else:
        return 1 / (math.sqrt(mod - 1))


# def distances_in_text(text, embedding):
#     tokenized = sent_tokenize(text)
#     distances = {}
#     for i, sent in enumerate(tokenized):
#         if i >= len(tokenized) - 1:
#             break
#         sent1 = tokenized[i]
#         sent2 = tokenized[i + 1]
#         dist = distance(sent1, sent2, embedding)
#         print("Sent 1: ", sent1)
#         print("Sent 2: ", sent2)
#         print("Distance between sentences: ", dist)
#         distances[(sent1, sent2)] = dist
#
#     print("Mininum: ", min(distances, key=distances.get))
#     print("Sorted: ")
#     print(dict(sorted(distances.items(), key=lambda item: item[1])))


def segment_similarity(tok1, tok2, embedding):
    sum_sentence_similarities = 0
    for i, sent_seg_1 in enumerate(tok1):
        for j, sent_seg_2 in enumerate(tok2):
            # weighted
            sum_sentence_similarities += distance(sent_seg_1, sent_seg_2, embedding) * weight(i, j)

    return sum_sentence_similarities / (len(tok1) * len(tok2))


def weakest_segments(text, embedding, min_seg_size=2):
    tok = sentence_tokenizer(text)
    if len(tok) <= min_seg_size:
        print("Can't be broken down anymore")
        return None, None, None

    last_index = len(tok) - 1

    min_seg_similarity = sys.float_info.max
    min_segments = None, None

    for i, sent, in enumerate(tok):
        if min_seg_size <= i <= last_index - min_seg_size:
            print("valid i: ", i)
            seg1 = tok[0:i]
            seg2 = tok[i:last_index]
            similarity = segment_similarity(seg1, seg2, embedding)
            if similarity < min_seg_similarity:
                min_segments = seg1, seg2
                min_seg_similarity = similarity

    s1, s2 = min_segments

    if min_segments == (None, None):
        print("Wasn't broken!")
        return None, None, None

    return " ".join(s1), " ".join(s2), min_seg_similarity


def get_clusters(text, n, embedding):
    segments = [text]
    while len(segments) < n:
        text_broken_into_segs = None
        min_segments = None, None
        min_segments_sim = sys.float_info.max
        for segs in segments:
            s1, s2, sim = weakest_segments(segs, embedding)
            if s1 == None and s2 == None:
                print("Seg is too small")
                continue
            if sim < min_segments_sim:
                min_segments_sim = sim
                min_segments = s1, s2
                text_broken_into_segs = segs

        if min_segments != (None, None):
            seg1, seg2 = min_segments
            index = segments.index(text_broken_into_segs)
            segments.insert(index, seg2)
            segments.insert(index, seg1)
            segments.remove(text_broken_into_segs)

    print("Segments size: ", len(segments))
    return segments


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


def match_clusters_dataframe(clusters1, clusters2):
    matches = pd.DataFrame()
    for i, cluster in enumerate(clusters1):
        sim, match = get_matching_cluster(cluster, clusters2)
        clusters2.remove(match)
        # Simplify clusters
        # string1_simplified = simplify_muss(cluster)
        # string2_simplified = simplify_muss(match)
        # matches = matches.append({'Text1': string1_simplified, 'Text2': string2_simplified}, ignore_index=True)
        matches = matches.append({'Text1': cluster, 'Text2': match}, ignore_index=True)

    return matches


def tabulate_text(t1, t2, n, embedding):
    s1 = get_clusters(t1, n, embedding)
    print("s1: ", s1)
    s2 = get_clusters(t2, n, embedding)
    print("s2: ", s2)
    table_output = match_clusters_dataframe(copy(s1), copy(s2))
    return s1, s2, table_output


def evaluate(clusters_real, clusters_pred):
    scores = []
    print("test: ", len(clusters_real), len(clusters_pred))
    for i, real in enumerate(clusters_real):
        pred = clusters_pred[i]
        scores.append(rouge_score(real, pred)['rouge-1']['f'])
        # scores.append(rouge_score(real, pred)['rouge-2']['f'])
    return scores

if __name__ == '__main__':
    freeze_support()
    # dataframe = pd.read_csv('../examples/edmundsons_testing_dataset.csv')

    example = "The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13\" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.  The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545.  Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha. The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an indicvidual. The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. What's More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back. "
              # "The Rajya Sabha is a permanent House. It cannot be dissolved. When the Lok Sabha is not in session or is dissolved, the permanent house still functions. However, each member of the Rajya Sabha enjoys a six-year term. Every two years one-third of its members retire by rotation. The total strength of the Rajya Sabha cannot be more than 250 of which 238 are elected while 12 arc nominated by the President of India. Election to the Rajya Sabha is done indirectly. The members of the state legislature elect the state representatives to the Rajya Sabha in accordance with the system of proportional representation by means of a single transferable vote. The seats in the Rajya Sabha for each state and Union Territory arc fixed on the basis of its population. A constituency is an area demarcated for the purpose of election. In other words, it is an area or locality with a certain number of people who choose a person to represent them in the Lok Sabha. Each State and Union Territory is divided into territorial constituencies.The division is not based on area but on population. Let us consider Mizoram, Rajasthan and Uttar Pradesh. Uttar Pradesh, a large state with dense population, has 80 constituencies."
    # min_size, max_size = get_seg_constraints(example, 3, 2)
    # print(weakest_segments(example, spacy_embedding, min_size, max_size))

    clusters = (get_clusters(example, 6, bert))
    for seg in clusters:
        print(seg)
        print("--------------")

    # # Example 1
    # t1_org = dataframe['original'][0]
    # t2_org = dataframe['original'][1]
    #
    # clus_real1 = dataframe['paragraphs'][0].strip().split("\n")
    # clus_real2 = dataframe['paragraphs'][1].strip().split("\n")
    #
    # clus_pred1, clus_pred2, table = tabulate_text(t1_org, t2_org, 3, doc2vec)
    #
    # print(table.transpose().to_json())
    # print(evaluate(clus_real1, clus_pred1))
    # print(evaluate(clus_real2, clus_pred2))
    #
    # table.to_csv("table-1-bert.csv")
    # # Example 2
    # t1_org = dataframe['original'][4]
    # t2_org = dataframe['original'][5]
    #
    # clus_real1 = dataframe['paragraphs'][4].strip().split("\n")
    # clus_real2 = dataframe['paragraphs'][5].strip().split("\n")
    #
    # clus_pred1, clus_pred2, table = tabulate_text(t1_org, t2_org, 2, doc2vec)
    #
    # print(table.transpose().to_json())
    # print(evaluate(clus_real1, clus_pred1))
    # print(evaluate(clus_real2, clus_pred2))
    # table.to_csv("table-2-bert.csv")
