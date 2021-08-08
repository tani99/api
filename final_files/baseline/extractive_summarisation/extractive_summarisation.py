class ExtractiveSummariser:
    def __init__(self, original_text, sentences_count):
        self.original_text = original_text
        self.sentences_count = sentences_count

    def join_sentences(self, summary_sentences):
        return  "\n".join(str(s) for s in summary_sentences)

    def summarise(self):
        raise NotImplementedError("Please Implement the summarise() method")
