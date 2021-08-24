from multiprocessing import Pool

from muss.simplify import simplify_sentences
from nltk.tokenize import sent_tokenize

from final_files.simplification.util import groups_of_n
from final_files.util import sentence_tokenizer


def simplify_muss(extracted, n=1):
    text = groups_of_n(n, sentence_tokenizer(extracted))
    # text = " ".join(sentence_tokenizer(extracted))
    simplified = simplify(text)
    summary = "\n".join(simplified)
    return summary

def simplify(text):
    source_sentences = text.split("\n")
    pred_sentences = simplify_sentences(source_sentences, model_name="muss_en_wikilarge_mined")
    for c, s in zip(source_sentences, pred_sentences):
        print('-' * 80)
        print(f'Original:   {c}')
        print(f'Simplified: {s}')

    return pred_sentences

def simplify_muss_multi(extracted, n=1):
    text = groups_of_n(n, sentence_tokenizer(extracted))
    # text = " ".join(sentence_tokenizer(extracted))
    simplified = simplify_multi(text)
    # summary = "\n".join(simplified)
    return simplified

def f(sentence):
    return simplify_sentences([sentence], model_name="muss_en_wikilarge_mined")

def simplify_multi(text):
    source_sentences = text.split("\n")
    simplified = ""

    print("POOLED")
    with Pool(len(source_sentences)) as p:
        print(p.map(f, source_sentences))

    # for sentence in source_sentences:
    #     pred_sentences = simplify_sentences([sentence], model_name="muss_en_wikilarge_mined")
    #     simplified += "\n".join(pred_sentences)
    #     simplified += " "

    # for c, s in zip(source_sentences, pred_sentences):
    #     print('-' * 80)
    #     print(f'Original:   {c}')
    #     print(f'Simplified: {s}')

    return "TESTING"
    # return simplified