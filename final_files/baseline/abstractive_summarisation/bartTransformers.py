# Importing the model
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
# Loading the model and tokenizer for bart-large-cnn
from final_files.baseline.abstractive_summarisation.abstractive_summarisation import AbstractiveSummariser

tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

class BartTransformersSummariser(AbstractiveSummariser):
    def summarise(self):
        # Encoding the inputs and passing them to model.generate()
        inputs = tokenizer.batch_encode_plus([self.original_text], return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'], early_stopping=True)

        # Decoding and printing the summary
        bart_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return bart_summary
