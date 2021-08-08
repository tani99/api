from muss.simplify import simplify_sentences
from nltk.tokenize import sent_tokenize

from final_files.util import sentence_tokenizer


def simplify_muss(extracted):
    text = " ".join(sentence_tokenizer(extracted))
    simplified = simplify(text)
    summary = "\n".join(simplified)
    return summary

def simplify(text):
    print(text)
    source_sentences = sent_tokenize(text)
    pred_sentences = simplify_sentences(source_sentences, model_name="muss_en_wikilarge_mined")
    for c, s in zip(source_sentences, pred_sentences):
        print('-' * 80)
        print(f'Original:   {c}')
        print(f'Simplified: {s}')

    return pred_sentences