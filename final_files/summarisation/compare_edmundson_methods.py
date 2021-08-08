import os

import pandas as pd

from final_files.util import sentence_tokenizer

edmundsons_results_file = 'edmundsons_results.csv'
def similarity(extract1, combination_extract):

    sen_ex_1 = set(sentence_tokenizer(extract1))
    sen_ex_2 = set(sentence_tokenizer(combination_extract))

    # Pure values
    # return len(sen_ex_1 & sen_ex_2)

    # Probabilities
    return len(sen_ex_1 & sen_ex_2)/ len(sentence_tokenizer(combination_extract))

def update_csv_with_similarity_scores():
    dataframe = pd.read_csv(edmundsons_results_file)

    sims_dataframe = pd.DataFrame(index=range(0,10), columns = [])

    # opening the file with w+ mode truncates the file
    # dataframe.insert(loc=0, column='Test', value=["TEST",2,3,4,5,6,7,8,9,10])
    cue_similarities, key_similarities, title_similarities, length_similarities = [], [], [], []
    for i, row in dataframe.iterrows():

        extract_final_edmundsons = dataframe["Edmundsons"][i]

        # CUE
        cue_similarities.append(similarity(dataframe["EdmundsonsCue"][i], extract_final_edmundsons))
        # KEY
        key_similarities.append(similarity(dataframe["EdmundsonsKey"][i], extract_final_edmundsons))
        # TITLE
        title_similarities.append(similarity(dataframe["EdmundsonsTitle"][i], extract_final_edmundsons))
        # LENGTH
        length_similarities.append(similarity(dataframe["CustomEdmundsonsLength"][i], extract_final_edmundsons))

    # Add columns with similarities to the middles
    dataframe.insert(loc=3, column='Cue Sim', value=cue_similarities)
    dataframe.insert(loc=5, column='Key Sim', value=key_similarities)
    dataframe.insert(loc=7, column='Title Sim', value=title_similarities)
    dataframe.insert(loc=9, column='Length Sim', value=length_similarities)

    # Similarities overview dataframe
    sims_dataframe['original'] = dataframe['original']
    sims_dataframe['originalNumSentences'] = [len(sentence_tokenizer(text)) for text in dataframe['original']]
    sims_dataframe['Edmundsons'] = dataframe['Edmundsons']
    sims_dataframe['EdmundsonsNumSentences'] = [len(sentence_tokenizer(text)) for text in dataframe['Edmundsons']]
    sims_dataframe['CueSim'] = cue_similarities
    sims_dataframe['KeySim'] = key_similarities
    sims_dataframe['TitleSim'] = title_similarities
    sims_dataframe['LengthSim'] = length_similarities

    # print(sims_dataframe['EdmundsonsNumSentences'])
    # sims_dataframe.to_csv('results_csv/edmundsons_similarities_overview.csv')

    # Can be probabilities if the correct bit is removed.
    dataframe.to_csv('results_csv/edmundsons_similarity_probabilities.csv')


if __name__ == '__main__':
    update_csv_with_similarity_scores()
