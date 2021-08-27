import itertools
import math
from copy import copy
import time
from multiprocessing.dummy import freeze_support
# import itertoolsx
from itertools import permutations
import pandas as pd
from final_files.visualisation.embeddings import spacy_embedding
from final_files.visualisation.weakest_link_method import get_clusters
from tables.tables import get_cluster_similarity
import unittest

def sim(pairs):
    # print("pairs", pairs)
    total = 0
    for (a, b) in pairs:
        start = time.time()
        total += get_cluster_similarity(a, b)
        end = time.time()
        print("Time - sim: ", end - start)

    return total


def best_matches(cluster1, cluster2):
    # create empty list to store the
    # combinations
    unique_combinations = []

    # Getting all permutations of list_1
    # with length of list_2
    permut = itertools.permutations(cluster1, len(cluster2))

    # zip() is called to pair each permutation
    # and shorter list element into combination
    for comb in permut:
        zipped = zip(comb, cluster2)
        # print(list(zipped))
        unique_combinations.append(list(zipped))

        # unique_combinations[function(list(zipped))] = list(zipped)

    # print(unique_combinations)
    dict = {}
    for combination in unique_combinations:
        print("got one ", combination)
        start = time.time()
        dict[sim(combination)] = combination
        end = time.time()
        print("Time: ", end - start)

    print("FINAL")
    max_key = list(reversed(sorted(dict.keys())))[0]

    matches = pd.DataFrame(index=range(0, len(dict[max_key])), columns=["Text1", "Text2"])
    i = 0
    for (text1, text2) in dict[max_key]:
        print("i")
        matches.loc[i, "Text1"] = text1
        matches.loc[i, "Text2"] = text2
        i += 1

    return matches

    # return dict
    # return [{multiply(combination): combination[0:50]} for combination in unique_combinations]
    # return unique_combinations

    # combos = []
    #
    #
    # copy1 = copy(cluster1)
    # copy2 = copy(cluster2)
    # for i in range(0, len(copy1)):
    #     for j in range(0, len(copy2)):
    #         combos.append((copy1[i], copy2[j]))
    #
    # return combos


dataframe = pd.read_csv('../examples/edmundsons_testing_dataset.csv')


# Example 1
# n = 3
# t1_org = dataframe['original'][0]
# t2_org = dataframe['original'][1]
#
# clus1 = get_clusters(t1_org, n, spacy_embedding)
# clus2 = get_clusters(t2_org, n, spacy_embedding)
#
# result = best_matches(clus1,clus2)
# print(result)
# max_key = list(reversed(sorted(result.keys())))[0]
# print(max_key)
# print("MAX: ", max_key, result[max_key])

#
# def calcSumPairProd(arr, n):
#     pairs = []
#     maxSum = 0
#     arr.sort()
#     print(arr)
#     i = 0
#     j = n - 1
#     while i < n and arr[i] < 0:
#         print("loop")
#         if i != n - 1 and arr[i + 1] <= 0:
#             maxSum = (maxSum + (arr[i] * arr[i + 1]))
#             print("Pair: ", arr[i], ", ", arr[i + 1])
#             print(maxSum)
#             i += 2
#         else:
#             break
#
#     while j >= 0 and arr[j] > 0:
#         print("loop2")
#         if j != 0 and arr[j - 1] > 0:
#             print("First")
#             print(maxSum)
#             maxSum = maxSum + (arr[j] * arr[j - 1])
#             print("Pair: ", arr[i], ", ", arr[j - 1])
#             print(maxSum)
#             j = j - 2
#         else:
#             break
#
#     if (j > i):
#         print("loop3")
#         maxSum = (maxSum + (arr[i] * arr[j]))
#         print("Pair: ", arr[i], ", ", arr[j])
#         print(maxSum)
#     elif (i == j):
#         print("loop4")
#         maxSum = (maxSum + arr[i])
#         print("Pair: ", maxSum, ", ", arr[i])
#         print(maxSum)
#
#     print("the fuck", maxSum)
#     return maxSum
#

def num_segs_possible(num_sents, min_seg_size):
    return math.floor(num_segs_possible_util(num_sents, min_seg_size, 0)/2)


def num_segs_possible_util(num_sents, min_seg_size, count):
    # print(num_sents)

    if num_sents < min_seg_size:
        return 1

    seg_1 = math.floor(num_sents / 2)
    seg_2 = num_sents - seg_1

    # print(seg_1, ", ", seg_2)

    return num_segs_possible_util(seg_1, min_seg_size, count) + num_segs_possible_util(seg_2, min_seg_size, (count))

def max_segs(n, min_seg_size):
    return math.floor(n/min_seg_size)

class TestClass(unittest.TestCase):

    def test(self):
        self.assertEqual(max_segs(3, 3), 1)
        self.assertEqual(max_segs(4, 3), 1)
        self.assertEqual(max_segs(5, 3), 1)
        self.assertEqual(max_segs(6, 3), 2)
        self.assertEqual(max_segs(7, 3), 2)
        self.assertEqual(max_segs(8, 3), 2)
        self.assertEqual(max_segs(9, 3), 3)
        self.assertEqual(max_segs(10, 3), 3)

if __name__ == '__main__':
    freeze_support()
    unittest.main()
    # arr = [1, 2, 3]
    # arr2 = [4, 5, 6]
    # n = len(arr)
    # print(best_matches(arr, arr2))

    # assertEqual(num_segs_possible(3,3), )
    # print("3:", num_segs_possible(3, 3))
    # print("4:", num_segs_possible(3, 3))
    # print("5:", num_segs_possible(3, 3))
    # print("6:", num_segs_possible(3, 3))
    # print("7:", num_segs_possible(3, 3))
    # print("8:", num_segs_possible(3, 3))
    # print("9:", num_segs_possible(3, 3))
    # print("10:", num_segs_possible(3, 3))
    # print("11:", num_segs_possible(3, 3))
    # print("12:", num_segs_possible(3, 3))
    # print("13:", num_segs_possible(3, 3))

