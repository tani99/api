from sumy.summarizers.kl import KLSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser

from final_files.baseline.extractive_summarisation.extractive_summarisation import ExtractiveSummariser


class KLSummariser(ExtractiveSummariser):

    def summarise(self):
        parser = PlaintextParser.from_string(self.original_text, Tokenizer('english'))
        # Instantiating the  KLSummarizer
        kl_summarizer = KLSummarizer()
        kl_summary = kl_summarizer(parser.document, sentences_count=self.sentences_count)

        return self.join_sentences(kl_summary)
