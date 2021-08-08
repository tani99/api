from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

from final_files.baseline.extractive_summarisation.extractive_summarisation import ExtractiveSummariser


class LexrankSummariser(ExtractiveSummariser):
    def summarise(self):
        # Initializing the parser
        my_parser = PlaintextParser.from_string(self.original_text, Tokenizer('english'))

        # Creating a summary of 3 sentences.
        lex_rank_summarizer = LexRankSummarizer()
        lexrank_summary = lex_rank_summarizer(my_parser.document, sentences_count=self.sentences_count)

        return self.join_sentences(lexrank_summary)
