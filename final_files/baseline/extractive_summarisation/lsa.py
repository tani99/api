from sumy.summarizers.lsa import LsaSummarizer

from final_files.baseline.extractive_summarisation.extractive_summarisation import ExtractiveSummariser


class LsaSummariser(ExtractiveSummariser):
    def summarise(self):
        # Parsing the text string using PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.parsers.plaintext import PlaintextParser
        parser=PlaintextParser.from_string(self.original_text,Tokenizer('english'))

        # creating the summarizer
        lsa_summarizer=LsaSummarizer()
        lsa_summary= lsa_summarizer(parser.document, sentences_count=self.sentences_count)

        return self.join_sentences(lsa_summary)