from nltk import sent_tokenize
from tables.tables import get_matching_cluster


def count_correct(clusters_predicted, clusters_real):

    real_clusters = sent_tokenize(real)
    pred_clusters = sent_tokenize(predicted)

    similarities = []
    for real_cluster in real:
        max_similarity, matching_cluster = get_matching_cluster(real_clusters, pred_clusters)

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

