from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.edmundson import EdmundsonSummarizer
from sumy.utils import get_stop_words
from nltk.tokenize import sent_tokenize

import math

LANGUAGE = "english"
stop_words = get_stop_words(LANGUAGE)

def edmundsons(text, percentage=0.8):
    tokenized_sentences = sent_tokenize(text)
    stemmer = Stemmer(LANGUAGE)
    summarizer = EdmundsonSummarizer(stemmer)
    summarizer.null_words = stop_words
    summarizer.bonus_words = ["word"]
    summarizer.stigma_words = ["word"]

    parser = PlaintextParser.from_string(text, Tokenizer(LANGUAGE))
    extracted = ""
    for sentence in summarizer(parser.document, math.floor(percentage * len(tokenized_sentences))):
        extracted = extracted + str(sentence) + "\n"

    return extracted