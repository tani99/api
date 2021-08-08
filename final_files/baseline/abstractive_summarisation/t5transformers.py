# # Importing requirements
# from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
#
# # text to summarize
from final_files.baseline.abstractive_summarisation.abstractive_summarisation import AbstractiveSummariser

# original_text = "The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time " \
#                 "during a national emergency. It can be dissolved earlier than its term by the President on the " \
#                 "advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence " \
#                 "motion. During the 13th Lok Sabha, Bhartiya Janata Party lost a no-confidence motion by one vote and " \
#                 "had to resign. "
#
# # Instantiating the model and tokenizer
# my_model = T5ForConditionalGeneration.from_pretrained('t5-small')
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
#
# # Concatenating the word "summarize:" to raw text
# text = "summarize:" + original_text
# print(text)
#
# # encoding the input text
# input_ids=tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
#
# # Generating summary ids
# summary_ids = my_model.generate(input_ids)
# print(summary_ids)
#
# # Decoding the tensor and printing the summary.
# t5_summary = tokenizer.decode(summary_ids[0])
# print(t5_summary)


import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config


class T5TransformersSummariser(AbstractiveSummariser):
    def summarise(self):
        model = T5ForConditionalGeneration.from_pretrained('t5-small')
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        device = torch.device('cpu')

        preprocess_text = self.original_text.strip().replace("\n", "")
        t5_prepared_Text = "summarize: " + preprocess_text

        tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

        # summmarize
        summary_ids = model.generate(tokenized_text,
                                     num_beams=4,
                                     no_repeat_ngram_size=2,
                                     min_length=0,
                                     max_length=self.length,
                                     early_stopping=True)

        output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        return output
