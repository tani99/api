from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.nlp.tokenizers import Tokenizer

from final_files.baseline.extractive_summarisation.extractive_summarisation import ExtractiveSummariser


class TextrankSummariser(ExtractiveSummariser):
    def summarise(self):
        parser = PlaintextParser.from_string(self.original_text, Tokenizer('english'))

        #  Creating the summarizer
        textrank_summarizer = TextRankSummarizer()
        textrank_summary = textrank_summarizer(parser.document, sentences_count=self.sentences_count)

        return self.join_sentences(textrank_summary)
