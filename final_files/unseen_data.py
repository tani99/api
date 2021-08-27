from multiprocessing.dummy import freeze_support
import pandas as pd

from final_files.extraction.extraction import process_text
from final_files.simplification.muss_simp.muss_simplification import simplify_muss
from final_files.summarisation.edmundsons.run_edmundsons import Edmundsons
from final_files.util import sentence_tokenizer

percent = 80
def get_results(text):
        sentences_count = max(1, int((float(percent) / 100) * len(sentence_tokenizer(text))))
        summariser = Edmundsons(text, sentences_count, cue_weight=1.0, key_weight=1.0,
                                title_weight=1.0, length_weight=1.0)
        summarised = summariser.summarise()
        simplified = simplify_muss(text, 2)
        points = process_text(text, True)
        complete = process_text(text, False)

        return summarised, simplified, points, complete

if __name__ == '__main__':
    freeze_support()

    data = pd.read_csv('examples/unseen_data.csv')
    results_unseen = pd.DataFrame(index=range(0,3), columns=["Original", "Summarised", "Simplified", "Points", "Complete"])

    for i in range(0,3):
        print("Processing ", i)

        summarised, simplified, points, complete = get_results(data.loc[i, "Original"])

        results_unseen.loc[i, "Original"] = data.loc[i, "Original"]
        results_unseen.loc[i, "Summarised"] = summarised
        results_unseen.loc[i, "Simplified"] = simplified
        results_unseen.loc[i, "Points"] = points
        results_unseen.loc[i, "Complete"] = complete

        print(results_unseen)

    results_unseen.to_csv("results_unseen.csv")

