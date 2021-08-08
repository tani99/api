# Importing model and tokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer

# Instantiating the model and tokenizer
from final_files.baseline.abstractive_summarisation.abstractive_summarisation import AbstractiveSummariser

tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
model = XLMWithLMHeadModel.from_pretrained('xlm-mlm-en-2048')


class XlmSummariser(AbstractiveSummariser):

    def summarise(self):
        # Encoding text to get input ids & pass them to model.generate()
        inputs = tokenizer.batch_encode_plus([self.original_text], return_tensors='pt', max_length=self.length,
                                             truncation=True)

        summary_ids = model.generate(inputs['input_ids'], early_stopping=True)

        # Decode and print the summary
        XLM_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(XLM_summary)
        return XLM_summary
