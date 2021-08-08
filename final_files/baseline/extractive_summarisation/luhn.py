from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

from final_files.baseline.extractive_summarisation.extractive_summarisation import ExtractiveSummariser


class LuhnSummariser(ExtractiveSummariser):
    def summarise(self):
        parser = PlaintextParser.from_string(self.original_text, Tokenizer('english'))

        #  Creating the summarizer
        luhn_summarizer = LuhnSummarizer()
        luhn_summary = luhn_summarizer(parser.document, sentences_count=self.sentences_count)

        return self.join_sentences(luhn_summary)
