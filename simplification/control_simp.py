
# import argparse
# import math

# from muss.simplify import simplify_sentences
# from sumy.nlp.stemmers import Stemmer
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.parsers.plaintext import PlaintextParser
# from sumy.summarizers.edmundson import EdmundsonSummarizer
# from sumy.utils import get_stop_words

# from controllable_simplification.generate_candidates import main as generate_candidates
# from controllable_simplification.ranking.main import main as rank
# import os
# from nltk.tokenize import sent_tokenize
# import torch

def controllable_simplification(input):
#     # Remove all new line characters
#     print("INPUT", input)
#     text = input.replace("\n"," ")
#     print("TEXT ADTER NEWLINES", text)
#     # text = input
#     f = open("input.txt", "w")
#     f.truncate(0)
#     f.write(text)
#     f.close()

#     print("First")

#     args = argparse.Namespace(input='input.txt', output='candidates.txt')
#     generate_candidates(args)

#     print("generated")
#     args = argparse.Namespace(model='controllable_simplification/ranking/model.bin', input='input.txt',
#                               candidates='candidates.txt', output='output.txt')
#     rank(args)
#     print("simplified")
#     simplified = ""
#     with open('output.txt') as f:
#         print("printing contents")
#         contents = f.read()
#         simplified += contents
#         print(contents)

#     return simplified 
      return None 