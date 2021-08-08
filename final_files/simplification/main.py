from multiprocessing.dummy import freeze_support

import pandas as pd
import spacy

from nltk.corpus import stopwords

from final_files.simplification.controllable_simp import controllable_simp
from final_files.util import sentence_tokenizer

STOP_WORDS = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

directory = '../examples/'

def pre_process_text(text):
    new_lines_removed = text.replace('\n', '')
    # Make sure sentences are separated by a space
    sentences = sentence_tokenizer(new_lines_removed)

    return " ".join(sentences)

def generate_results():
    columns = ['original'] + ["Controllable-Simp"]

    dataframe = pd.DataFrame(index=range(0, 10), columns=columns)

    # dataframe = pd.read_csv('results/control-simp-results.csv')

    for index in range(0, 10):
        filename = f"example{str(index+1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = pre_process_text(file.read()) #.replace('\n', '')
                print("TEXTTT ", text)
                dataframe['original'][index] = text
                summary = controllable_simp(text)
                print("SUMMARYYY ", summary)
                dataframe["Controllable-Simp"][index] = summary
                print("Summary length: ", len(sentence_tokenizer(summary)))

                print(dataframe)
            continue
        else:
            continue

    print(dataframe)
    dataframe.to_csv('results/control-simp-results.csv')

if __name__ == '__main__':
    freeze_support()
    # text = """The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.  The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545.  Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha. The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individual. The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. What's More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back."""

    generate_results()
    # generate_results_edmundsons_combination(combinations)

    # generate_results_edmundsons_basic()

#
# if __name__ == '__main__':
#     freeze_support()
#
#     print("FINAL ANSWER")
#     print(controllable_simp(text))
