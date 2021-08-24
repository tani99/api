from multiprocessing.dummy import freeze_support
import pandas as pd

from final_files.visualisation.embeddings import spacy_embedding, bert, doc2vec
from final_files.visualisation.weakest_link_method import tabulate_text, evaluate

if __name__ == '__main__':
    freeze_support()
    dataframe = pd.read_csv('../examples/edmundsons_testing_dataset.csv')

    embeddings = [spacy_embedding, bert, doc2vec]

    results = pd.DataFrame()

    for embedding in embeddings:
        # Example 1
        t1_org = dataframe['original'][0]
        t2_org = dataframe['original'][1]

        clus_real1 = dataframe['paragraphs'][0].strip().split("\n")
        clus_real2 = dataframe['paragraphs'][1].strip().split("\n")

        clus_pred1, clus_pred2, table = tabulate_text(t1_org, t2_org, 3, embedding)

        print(embedding.__name__)
        print(table)
        # results.append(table)
        # print(embedding.__name__)
        # print(table.transpose().to_json())
        # print(evaluate(clus_real1, clus_pred1))
        # print(evaluate(clus_real2, clus_pred2))

        # Example 2
        # t1_org = dataframe['original'][4]
        # t2_org = dataframe['original'][5]
        #
        # clus_real1 = dataframe['paragraphs'][4].strip().split("\n")
        # clus_real2 = dataframe['paragraphs'][5].strip().split("\n")
        #
        # clus_pred1, clus_pred2, table = tabulate_text(t1_org, t2_org, 2, embedding)
        #
        # results.append(table)
        # print(embedding.__name__)
        # print(table.transpose().to_json())
        # print(evaluate(clus_real1, clus_pred1))
        # print(evaluate(clus_real2, clus_pred2))

