import time
from multiprocessing.dummy import freeze_support

import pandas as pd
import spacy

from nltk.corpus import stopwords

from final_files.simplification.control_simp.controllable_simp import controllable_simp
from final_files.simplification.util import pre_process_text
from final_files.util import sentence_tokenizer

STOP_WORDS = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

directory = '../../examples/'


groups = ["Group " + str(i) for i in range(2, 5)]
def generate_results():

    columns = ['original'] + groups

    dataframe = pd.DataFrame(index=range(0, 10), columns=columns)

    # dataframe = pd.read_csv('results/control_simp-results.csv')

    for index in range(0, 10):
        filename = f"example{str(index+1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = pre_process_text(file.read()) #.replace('\n', '')
                print("TEXTTT ", text)
                dataframe['original'][index] = text
                for group in groups:
                    n = int(group.split(" ")[1])
                    summary = controllable_simp(text, n)
                    print("SUMMARYYY ", summary)
                    dataframe[group][index] = summary
                    print("Summary length: ", len(sentence_tokenizer(summary)))

                print(dataframe)
            continue
        else:
            continue

    print(dataframe)
    dataframe.to_csv('results/control_simp-results.csv')

def filter_seprators(filename):
    dataframe = pd.read_csv(filename)
    for index in range(0, 10):
        filename = f"example{str(index+1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                for group in groups:
                    value = dataframe[group][index]
                    dataframe.loc[index, group] = value.replace("<SEP> ", "")
            continue
        else:
            continue

    print(dataframe)
    dataframe.to_csv(filename)

def run_and_get_time(text, n):
    start_time = time.time()
    summary_no_group = controllable_simp(text, n=n)
    end_time = time.time()

    print("N: ", n)
    print("TIME: ", end_time-start_time)
    return  summary_no_group, end_time-start_time

def time_test():
    columns = []
    for i in range(0,10):
        columns.append("example " + str(i))
        columns.append("time " + str(i))

    df = pd.DataFrame(index=range(0, 10), columns=columns)

    for index in range(0, 10):
        filename = f"example{str(index + 1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = pre_process_text(file.read())
                sentences = sentence_tokenizer(text)
                for i in range(1, len(sentences)):
                    if i == len(sentences)-1:
                        sum_all, time_all = run_and_get_time(text, -1)
                        df.loc[i, "example " + str(index)] = sum_all
                        df.loc[i, "time " + str(index)] = time_all
                    sum_curr, time_curr = run_and_get_time(text, i)
                    df.loc[i, "example " + str(index)] = sum_curr
                    df.loc[i, "time " + str(index)] = time_curr

                df.to_csv("temp_control.csv")
                print(df)
                print(df[["example " + str(index), "time " + str(index)]])
                # df.to_csv("temp.csv")
    df.to_csv("time_test_control_simp.csv")

if __name__ == '__main__':
    freeze_support()
    time_test()



    # example = """The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.  The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545.  Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha. The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individual. The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. What's More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back."""
    #
    # print("RESULT: ", run_and_get_time(example, 2))
    # print("RESULT: ", run_and_get_time(example, 3))
    # generate_results()
    # filter_seprators("results/control_simp-results.csv")

#
# if __name__ == '__main__':
#     freeze_support()
#
#     print("FINAL ANSWER")
#     print(controllable_simp(example))
