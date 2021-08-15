from multiprocessing.dummy import freeze_support

from final_files.simplification.control_simp.controllable_simp import controllable_simp
from final_files.simplification.muss_simp.muss_simplification import simplify_muss
from final_files.summarisation.edmundsons.run_edmundsons import Edmundsons, pre_process_text
from final_files.util import sentence_tokenizer

import pandas as pd

directory = "../examples/"


def sum_simp():
    dataframe = pd.DataFrame(index=range(0, 10),
                             columns=["original", "sum", "sum_simp_muss", "sum_simp_control"])

    for index in range(0, 10):
        filename = f"example{str(index + 1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                print("index ", index)
                text = pre_process_text(file.read())

                sentences_count = 0.8 * len(sentence_tokenizer(text))
                # Summary -> Simplified
                summariser = Edmundsons(text, sentences_count, cue_weight=0, key_weight=1,
                                        title_weight=1, length_weight=1)
                summarised = summariser.summarise()
                print("----- summarised ---------")
                print(summarised)
                simplified_muss = simplify_muss(summarised)
                simplified_control_simp = controllable_simp(summarised, "../../controllable_simplification")

                print("--------------- summary -> simplified --------------------")
                print(simplified_muss)
                print(simplified_control_simp)

                dataframe.loc[index, "original"] = text
                dataframe.loc[index, "sum"] = summarised
                dataframe.loc[index, "sum_simp_muss"] = simplified_muss
                dataframe.loc[index, "sum_simp_control"] = simplified_control_simp
                print(dataframe)

            continue
        else:
            continue

    print(dataframe)
    dataframe.to_csv("sum-simp-results.csv")
    return dataframe

def simp_sum():
    dataframe = pd.DataFrame(index=range(0, 10),
                             columns=["original", "simp_muss", "simp_control", "simp_sum_muss", "simp_sum_control"])

    for index in range(0, 10):
        filename = f"example{str(index + 1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                print("index ", index)
                text = pre_process_text(file.read())

                # Simplified -> Summary
                pre_simplified_muss = simplify_muss(text)
                pre_simplified_control_simp = controllable_simp(text, "../../controllable_simplification")

                summariser_muss = Edmundsons(pre_simplified_muss, 0.8 * len(sentence_tokenizer(pre_simplified_muss)), cue_weight=0, key_weight=1,
                                         title_weight=1, length_weight=1)
                summariser_control = Edmundsons(pre_simplified_control_simp, 0.8 * len(sentence_tokenizer(pre_simplified_control_simp)), cue_weight=0, key_weight=1,
                                         title_weight=1, length_weight=1)
                summarised_muss = summariser_muss.summarise()
                summarised2_control = summariser_control.summarise()

                dataframe.loc[index, "original"] = text
                dataframe.loc[index, "simp_muss"] = pre_simplified_muss
                dataframe.loc[index, "simp_control"] = pre_simplified_control_simp
                dataframe.loc[index, "simp-sum_muss"] = summarised_muss
                dataframe.loc[index, "simp-sum_control"] = summarised2_control
                print(dataframe)

            continue
        else:
            continue

    print(dataframe)
    dataframe.to_csv("simp-sum-results.csv")
    return dataframe

if __name__ == '__main__':
    freeze_support()
    # sum_simp()

    simp_sum()

    # results = pd.DataFrame(index=range(0, 10), columns=["original", "sum", "sum_simp_muss", "sum_simp_control"])
    # texty = """The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.  The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545.  Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha. The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individual. The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. What's More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back."""
    # sentences_county = 0.8 * len(sentence_tokenizer(texty))
    #
    # # Summary -> Simplified
    # summariser = Edmundsons(texty, sentences_county, cue_weight=0, key_weight=1,
    #                         title_weight=1, length_weight=1)
    # summarised = summariser.summarise()
    # print("----- summarised ---------")
    # print(summarised)
    # simplified_muss = simplify_muss(summarised)
    # simplified_control_simp = controllable_simp(summarised, "../../controllable_simplification")
    #
    # print("--------------- summary -> simplified --------------------")
    # print(simplified_muss)
    # print(simplified_control_simp)
    #
    # results.loc[0, "original"] = texty
    # results.loc[0, "sum"] = summarised
    # results.loc[0, "sum_simp_muss"] = simplified_muss
    # results.loc[0, "sum_simp_control"] = simplified_control_simp
    #
    # print(results)

    # Simplified -> Summary
    # pre_simplified_muss = simplify_muss(text)
    # pre_simplified_control_simp = controllable_simp(text)
    # summariser1 = Edmundsons(pre_simplified_muss, sentences_count, cue_weight=0, key_weight=1,
    #                          title_weight=1, length_weight=1)
    # summariser2 = Edmundsons(pre_simplified_control_simp, sentences_count, cue_weight=0, key_weight=1,
    #                          title_weight=1, length_weight=1)
    # summarised1 = summariser1.summarise()
    # summarised2 = summariser2.summarise()

    # print("--------------- simplified -> summary --------------------")
    # print(summarised1)
    # print(summarised2)
