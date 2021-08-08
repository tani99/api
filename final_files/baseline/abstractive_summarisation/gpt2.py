# Importing model and tokenizer
from transformers import GPT2Tokenizer,GPT2LMHeadModel
from final_files.baseline.abstractive_summarisation.abstractive_summarisation import AbstractiveSummariser


class Gpt2Summariser(AbstractiveSummariser):
    def summarise(self):
        # Instantiating the model and tokenizer with gpt-2
        tokenizer=GPT2Tokenizer.from_pretrained('gpt2')
        model=GPT2LMHeadModel.from_pretrained('gpt2')

        # Encoding text to get input ids & pass them to model.generate()
        inputs=tokenizer.batch_encode_plus([self.original_text], return_tensors='pt', max_length=self.length, truncation=True)
        summary_ids=model.generate(inputs['input_ids'],early_stopping=True)

        # Decoding and printing summary
        GPT_summary=tokenizer.decode(summary_ids[0],skip_special_tokens=True)
        return GPT_summary
