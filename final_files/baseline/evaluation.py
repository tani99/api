from nltk import word_tokenize
from gensim.summarization import keywords
import yake


def keywords_lost(original, summary):
    keywords_org = keywords_gensim(original)
    keywords_sum = keywords_gensim(summary)

    # print(len(list(set(keywords_org) - set(keywords_sum))))
    print(keywords_org)
    print(keywords_sum)

    # print(len(keywords_org))
    # print(len(keywords_sum))
    ratio_org = len(keywords_org) / len(word_tokenize(original))
    ratio_sum = len(keywords_sum) / len(word_tokenize(summary))

    keywords_lost = len(list(set(keywords_org) - set(keywords_sum)))
    return keywords_lost


def keywords_gensim(text):
    return keywords(text).split('\n')


def keywords_yake(text):
    kw_extractor = yake.KeywordExtractor()
    language = "en"
    max_ngram_size = 3
    deduplication_threshold = 0.9
    numOfKeywords = 20
    custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram_size, dedupLim=deduplication_threshold,
                                                top=numOfKeywords, features=None)
    keywords = custom_kw_extractor.extract_keywords(text)
    for kw in keywords:
        ...
        print(kw)
    return [k for (k, p) in keywords]
