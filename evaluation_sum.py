# Summarisation

# Noun phrases

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import RegexpParser
from nltk import Tree
import pandas as pd

# Defining a grammar & Parser
NP = "NP: {(<V\w+>|<NN\w?>)+.*<NN\w?>}"
chunker = RegexpParser(NP)
import spacy

DETERMINERS = ["a", "an", "the"]


def noun_phrases(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    phrases = []
    for chunk in doc.noun_chunks:
        phrases.append(chunk.text)
        # print(chunk.root.text, chunk.root.head.text)
        # print(filter_determiners(chunk.text))
        # print(chunk.text, chunk.root.text, chunk.root.dep_,
        #       chunk.root.head.text)
    return phrases


def filter_determiners(text):
    for det in DETERMINERS:
        text = text.replace(det + " ", "")
    return text


def noun_phrase_count(text):
    return len(noun_phrases(text))


def score_words(text):
    nlp = spacy.load("en_core_web_sm")
    # Merge noun phrases and entities for easier analysis
    nlp.add_pipe("merge_entities")
    nlp.add_pipe("merge_noun_chunks")
    doc = nlp(text)
    ents = doc.ents
    doc_length = 0
    ne_score = 0
    nc_score = 0
    verb_score = 0
    stopword_score = 0
    count_passive = 0
    for token in doc:
        # print(token.dep_)
        if token.dep_ == "nsubjpass" or token.dep_ == "auxpass" or token.dep_ == "csubjpass":
            # print(token, " - passive")
            count_passive += 1
        # else:
        #     print(token, " - active")
        if token.text in [e.text for e in ents]:
            # print("Named entity!")
            ne_score += 1
        # elif token.text in [nc.text for nc in doc.noun_chunks]:
        #     # print("Noun phrase!")
        #     nc_score += 0.8
        elif token.pos_ == "VERB":
            # print("Verb!")
            verb_score += 0.5
        elif token.is_stop:
            # print("Stopword!")
            stopword_score -= 0.5
        if not token.is_punct:
            doc_length += 1
    print("passive count: " + str(count_passive))
    print("doc length: " + str(doc_length))
    print(ne_score, nc_score, verb_score, stopword_score)
    score = ne_score + nc_score + verb_score + stopword_score
    return score / doc_length, count_passive

def score(org, sum):
    score_org, passive_count_org = score_words(org)
    score_sum, passive_count_sum = score_words(sum)

    print("original: ", score_org)
    print("summary: ", score_sum)

    print("Passive improvement: " + str(passive_count_org) + " -> " + str(passive_count_sum))

def improvement(org, sum):
    return sum / (org+sum)

example = "The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency."

original = "The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency.  It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13th Lok Sabha, Bhartiya Janata Party lost a no-confidence motion by one vote and had to resign."
summary = "Lok Sabha members are elected for five years. Its life can be extended if there is a national emergency for one year.It can be dissolved earlier than its term. The President can ask the Prime Minister to do this. The government can be removed from office by a debate or a no-confidence vote. During the 13th Lok Sabha, the Prime Minister had to resign because of a no-confidence motion."

# print(noun_phrases(example))

# print("original: ", score_words(original))
# print("summary: ", score_words(summary))

score(original, summary)
# Tables
