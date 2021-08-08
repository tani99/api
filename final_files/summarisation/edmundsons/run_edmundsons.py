import os
from multiprocessing.dummy import freeze_support

import pandas as pd
import spacy

from nltk.corpus import stopwords
from sumy.parsers.plaintext import PlaintextParser

from final_files.baseline.extractive_summarisation.extractive_summarisation import ExtractiveSummariser
from final_files.summarisation.edmundsons.edmundson import EdmundsonSummarizer
from final_files.summarisation.nlp.tokenizers import Tokenizer
from final_files.util import sentence_tokenizer

PERCENTAGE_RETAINED = 0.8

STOP_WORDS = set(stopwords.words('english'))

nlp = spacy.load("en_core_web_sm")

directory = '../examples/'


class Edmundsons(ExtractiveSummariser):

    def __init__(self, original_text, sentences_count, cue_weight=0.0, key_weight=0.0,
                 title_weight=0.0, location_weight=0.0, length_weight=0.0):
        super(Edmundsons, self).__init__(original_text, sentences_count)

        self.doc = nlp(self.original_text)

        self.summariser = EdmundsonSummarizer(cue_weight=cue_weight, key_weight=key_weight,
                                              title_weight=title_weight, location_weight=location_weight,
                                              length_weight=length_weight)
        self.summariser.bonus_words = self.get_bonus_words()
        # print(self.summariser.bonus_words)
        # print("Stigma words: ", self.get_stigma_words())
        self.summariser.stigma_words = self.get_stigma_words()
        self.summariser.null_words = STOP_WORDS

    def get_bonus_words(self):
        if len(self.doc.ents) == 0:
            # print("Empty bonus word list")
            return ["EMPTY BONUS WORD LIST!"]
        return self.doc.ents

    def get_null_words(self):
        return STOP_WORDS

    def get_stigma_words(self):
        count_map = {}

        for token in self.doc:
            if not token.is_punct and not token.is_stop:
                count = count_map.get(token.text, 0)
                count_map[token.text] = count + 1

        most_frequent_words = []
        for word, value in count_map.items():  # for name, age in dictionary.iteritems():  (for Python 2.x)
            if value >= 5:
                most_frequent_words.append(word)
        if len(most_frequent_words) == 0:
            # print("Empty most frequent word list")
            return ["EMPTY MOST FREQUENT LIST!"]
        return most_frequent_words

    def summarise(self):
        # Initializing the parser
        my_parser = PlaintextParser.from_string(self.original_text, Tokenizer('english'))

        # Creating a summary of 3 sentences.
        print("SENTENCES COUNT ACTUALLY CALLED: ", self.sentences_count)
        edmundson_summary = self.summariser(my_parser.document, sentences_count=self.sentences_count)

        return self.join_sentences(edmundson_summary)

def pre_process_text(text):
    new_lines_removed = text.replace('\n', '')
    # Make sure sentences are separated by a space
    sentences = sentence_tokenizer(new_lines_removed)

    return " ".join(sentences)

def generate_results_edmundsons_basic():
    dataframe = pd.read_csv('edmundsons/edmundsons_results_template.csv')

    for index in range(0, 10):
        filename = f"example{str(index+1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = pre_process_text(file.read()) #.replace('\n', '')

                dataframe['original'][index] = text
                sentences_count = int(len(sentence_tokenizer(text)) * PERCENTAGE_RETAINED)
                print("TESTING SENTENCE COUNT")
                print(len(sentence_tokenizer(text)), sentences_count)

                # CUE
                summariser = Edmundsons(text, sentences_count, cue_weight=1.0)
                summary = summariser.summarise()
                dataframe["EdmundsonsCue"][index] = summary
                print("Summary length: ", len(sentence_tokenizer(summary)))

                # KEY
                summariser = Edmundsons(text, sentences_count, key_weight=1.0)
                summary = summariser.summarise()
                dataframe["EdmundsonsKey"][index] = summary
                print("Summary length: ", len(sentence_tokenizer(summary)))

                # TITLE
                summariser = Edmundsons(text, sentences_count, title_weight=1.0)
                summary = summariser.summarise()
                dataframe["EdmundsonsTitle"][index] = summary
                print("Summary length: ", len(sentence_tokenizer(summary)))

                # LENGTH
                summariser = Edmundsons(text, sentences_count, length_weight=1.0)
                summary = summariser.summarise()
                dataframe["CustomEdmundsonsLength"][index] = summary
                print("Summary length: ", len(sentence_tokenizer(summary)))

                # ALL
                summariser = Edmundsons(text, sentences_count, cue_weight=1.0, key_weight=1.0,
                                        title_weight=1.0, length_weight=1.0)
                summary = summariser.summarise()

                dataframe["Edmundsons"][index] = summary
                print("Summary length: ", len(sentence_tokenizer(summary)))

                print(dataframe)
            continue
        else:
            continue

    print(dataframe)
    # dataframe.to_csv('edmundsons_results.csv')

def generate_results_edmundsons_combination(combinations):
    columns = ['original'] + ["_".join(args) for args in combinations]

    dataframe = pd.DataFrame(index=range(0, 10), columns=columns)

    print(dataframe)
    for index in range(0, 10):
        filename = f"example{str(index+1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = pre_process_text(file.read())

                dataframe.loc[index,'original'] = text

                for args in combinations:

                    sentences_count = int(len(sentence_tokenizer(text)) * PERCENTAGE_RETAINED)
                    summariser = Edmundsons(text, sentences_count, cue_weight=args.get("cue", 0.0), key_weight=args.get("key", 0.0),
                                            title_weight=args.get("title", 0.0), length_weight=args.get("length", 0.0))
                    summary = summariser.summarise()
                    column_name = "_".join(args)
                    dataframe.loc[index, column_name] = summary

                print(dataframe)
            continue
        else:
            continue

    print(dataframe)
    dataframe.to_csv('edmundsons_results_combinations_with_cue.csv')

if __name__ == '__main__':
    freeze_support()
    text = """The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.  The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545.  Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha. The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individual. The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. What's More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back."""

    # Without Cue
    # test1 = {"key":1.0, "length":1.0}
    # test2 = {"key":1.0, "title":1.0}
    # test3 = {"length": 1.0, "title": 1.0}
    # test4 = {"key":1.0, "title":1.0, "length":1.0}

    # With Cue
    test1 = {"cue":1.0, "key":1.0}
    test2 = {"cue":1.0, "title":1.0}
    test3 = {"cue": 1.0, "length": 1.0}
    test4 = {"cue":1.0, "title":1.0, "length":1.0}
    test5 = {"cue": 1.0, "key": 1.0, "length": 1.0}
    test6 = {"cue": 1.0, "key": 1.0, "title": 1.0}

    combinations = [test1, test2, test3, test4 , test5, test6]

    generate_results_edmundsons_combination(combinations)

    # generate_results_edmundsons_basic()

