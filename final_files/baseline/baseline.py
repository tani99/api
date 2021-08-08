import math
import os
from multiprocessing.spawn import freeze_support

from final_files.baseline.abstractive_summarisation.bartTransformers import BartTransformersSummariser
from final_files.baseline.abstractive_summarisation.gpt2 import Gpt2Summariser
from final_files.baseline.abstractive_summarisation.t5transformers import T5TransformersSummariser
from final_files.baseline.abstractive_summarisation.xlm_transformers import XlmSummariser
from final_files.baseline.extractive_summarisation.klsum import KLSummariser
from final_files.baseline.extractive_summarisation.lexrank import LexrankSummariser
from final_files.baseline.extractive_summarisation.lsa import LsaSummariser
from final_files.baseline.extractive_summarisation.luhn import LuhnSummariser
from final_files.baseline.extractive_summarisation.textrank import TextrankSummariser

from nltk import sent_tokenize, word_tokenize
import pandas as pd

from final_files.baseline.util import pre_process_text

pd.options.mode.chained_assignment = None

directory = '../examples/'

SUMMARIZERS_EXTRACTIVE = [LexrankSummariser, TextrankSummariser, LuhnSummariser, LsaSummariser, KLSummariser]
SUMMARIZERS_ABSTRACTIVE = [T5TransformersSummariser, XlmSummariser, Gpt2Summariser, BartTransformersSummariser]
PERCENTAGE_RETAINED = 0.8


# Get baseline average summary based on sentence count
# Ties are broken with random selection
def average_summary(filename, summarisers):
    df = pd.read_csv(filename)

    for i, row in df.iterrows():
        map = {}
        text = df['original'][i]
        print(text)
        sentences_count = int(len(sent_tokenize(text)) * PERCENTAGE_RETAINED)
        for sum in summarisers:
            text = df[sum.__name__][i]
            for sent in sent_tokenize(text):
                count = map.get(sent, 0)
                map[sent] = count + 1
        average_summary = list(dict(sorted(map.items(), key=lambda item: item[1], reverse=True)).keys())[
                          :sentences_count]
        df['average'][i] = "\n".join(average_summary)
        print(df)

    df.to_csv(filename)


# Used to generate results on a set of examples and store them to baseline_results.csv
def generate_results_extractive():
    dataframe = pd.DataFrame(index=range(0, 10), columns=['original', 'LexrankSummariser','KLSummariser','LsaSummariser','TextrankSummariser','LuhnSummariser','average'])

    for index in range(0, 10):
        filename = f"example{str(index+1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = file.read() #.replace('\n', '')
                text = pre_process_text(text)
                dataframe['original',index] = text
                sentences_count = len(sent_tokenize(text)) * PERCENTAGE_RETAINED
                for sum in SUMMARIZERS_EXTRACTIVE:
                    summariser = sum(text, sentences_count)
                    summary = summariser.summarise()
                    print(index)
                    print(sum.__name__)
                    print(dataframe)
                    dataframe[sum.__name__,index] = summary
                    print(dataframe)
                    print(filename)
                    print(text)
                    print(summary)
                    print("\n-----------------------\n")
            continue
        else:
            continue

    print(dataframe)
    dataframe.to_csv('baseline_results.csv')

# Used to generate results on a set of examples and store them to baseline_results_abstractive.csv
def generate_results_abstractive():
    dataframe = pd.DataFrame(index=range(0, 10), columns=['original', 'T5TransformersSummariser', 'XlmSummariser', 'Gpt2Summariser', 'BartTransformersSummariser','average'])

    print(dataframe)
    for index in range(0, 10):
        filename = f"example{str(index+1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = file.read() #.replace('\n', '')
                text = pre_process_text(text)
                dataframe['original',index] = text
                length = int(len(word_tokenize(text)) * PERCENTAGE_RETAINED)
                for sum in SUMMARIZERS_ABSTRACTIVE:
                    summariser = sum(text, length)
                    summary = summariser.summarise()
                    print(sum.__name__)
                    dataframe.loc[sum.__name__, index] = summary
                    # dataframe[sum.__name__][index] = summary
                    print(filename)
                    print(text)
                    print(summary)
                    print("\n-----------------------\n")
            continue
        else:
            continue

    print(dataframe)
    dataframe.to_csv('baseline_results_abstractive.csv')

if __name__ == '__main__':
    freeze_support()
    # generate_results_extractive()
    # average_summary('baseline_results.csv', SUMMARIZERS_EXTRACTIVE)
    generate_results_abstractive()
    average_summary('baseline_results_abstractive.csv', SUMMARIZERS_ABSTRACTIVE)
