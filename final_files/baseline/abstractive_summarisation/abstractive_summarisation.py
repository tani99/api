class AbstractiveSummariser:
    def __init__(self, original_text, length):
        self.original_text = original_text
        self.length = length

    def summarise(self):
        raise NotImplementedError("Please Implement the summarise() method")

