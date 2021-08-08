# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

from itertools import chain
from operator import attrgetter
from ._compat import ffilter
from ._summarizer import AbstractSummarizer


class EdmundsonLengthMethod(AbstractSummarizer):
    def __init__(self, stemmer, null_words):
        super(EdmundsonLengthMethod, self).__init__(stemmer)
        self._null_words = null_words

    def __call__(self, document, sentences_count):
        ratings = self._rate_sentences(document)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    def _rate_sentence(self, sentence, document):
        return self._normalised_sentence_length(sentence, document)

    def rate_sentences(self, document):
        return self._rate_sentences(document)

    def _rate_sentences(self, document):
        rated_sentences = {}
        for sentence in document.sentences:
            rated_sentences[sentence] = self._rate_sentence(sentence, document)

        return rated_sentences

    def _longest_sentence_length(self, document):
        longest_sentence_length = 0
        for sentence in document.sentences:
            sentence_length = len(sentence.words)
            if sentence_length > longest_sentence_length:
                longest_sentence_length = sentence_length
        return longest_sentence_length

    # Normalised sentence length is ratio of the length of a sentence to the length of the longest sentence in the document
    def _normalised_sentence_length(self, sentence, document):
        return len(sentence.words)/self._longest_sentence_length(document)