import os

from rouge import Rouge
import pandas as pd

RESULTS_COLUMNS = ["Group " + str(i) for i in range(2, 5)]

def rouge_score(real, prediction):
    rouge = Rouge()
    scores = rouge.get_scores(prediction, real)
    return scores[0]

TEST_DATASET_FILEPATH = '../../examples/edmundsons_testing_dataset.csv'

def populate_evaluation_results(results_file, write_to_file, results_columns, real_summary_column='human_summary'):
    dataframe = pd.read_csv(results_file)
    print(os.getcwd())
    human_summarised_dataframe = pd.read_csv(TEST_DATASET_FILEPATH)
    rouge_scores = pd.DataFrame(index=range(0, 10), columns=[])

    for result_column in results_columns:
        rouge_results_column = []
        for i, row in dataframe.iterrows():
            real = human_summarised_dataframe[real_summary_column][i]
            summary = dataframe[result_column][i]

            # Rouge-1&2 F1 score
            rouge_results_column.append(rouge_score(real, summary)['rouge-1']['f'])
            # rouge_results_column.append(rouge_score(real, summary)['rouge-2']['f'])
        rouge_scores[result_column] = rouge_results_column

    print(rouge_scores)
    # rouge_scores.to_csv(write_to_file)

if __name__ == '__main__':
    populate_evaluation_results('results/control-simp-results.csv',
                                'control_simp/results/control_simp-rouge-1-scores.csv',
                                RESULTS_COLUMNS, 'merged')