# from multiprocessing.dummy import freeze_support
import multiprocessing
from multiprocessing.dummy import freeze_support

from final_files.simplification.muss_simp.muss_simplification import simplify_muss, simplify_muss_multi
from final_files.simplification.util import pre_process_text
from final_files.util import sentence_tokenizer
import pandas as pd
import time

groups = ["Group " + str(i) for i in range(1, 5)]
directory = '../../examples/'


def generate_results():
    columns = ['original'] + groups
    dataframe = pd.DataFrame(index=range(0, 10), columns=columns)
    dataframe_time = pd.DataFrame(index=range(0, 10), columns=columns)

    for index in range(0, 10):
        filename = f"example{str(index + 1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = pre_process_text(file.read())
                print(text)
                dataframe['original'][index] = text
                for group in groups:
                    n = int(group.split(" ")[1])

                    start = time.time()
                    summary = simplify_muss(text, n)
                    end = time.time()

                    dataframe[group][index] = summary
                    dataframe_time[group][index] = end - start
                    print("RESULT")
                    print(summary)
                    print(end-start)

                print(dataframe)
                print(dataframe_time)
            continue
        else:
            continue

    print(dataframe)
    print(dataframe_time)
    # dataframe.to_csv("results/muss-simp-rults.csv")
    # dataframe_time.to_csv("results/muss-simp-results-time.csv")

def run_and_get_time(text, n):
    start_time = time.time()
    summary_no_group = simplify_muss(text, n=n)
    end_time = time.time()

    print("N: ", n)
    print("TIME: ", end_time-start_time)
    return  summary_no_group, end_time-start_time

def time_test():
    columns = []
    for i in range(0,10):
        columns.append("example " + str(i))
        columns.append("time " + str(i))

    df = pd.DataFrame(index=range(0, 10), columns=columns)

    for index in range(0, 10):
        filename = f"example{str(index + 1)}.txt"
        print(filename)
        filepath = directory + filename
        if filename.endswith(".txt"):
            with open(filepath, 'r') as file:
                text = pre_process_text(file.read())
                sentences = sentence_tokenizer(text)
                for i in range(1, len(sentences)):
                    if i == len(sentences)-1:
                        sum_all, time_all = run_and_get_time(text, -1)
                        df.loc[i, "example " + str(index)] = sum_all
                        df.loc[i, "time " + str(index)] = time_all
                    sum_curr, time_curr = run_and_get_time(text, i)
                    df.loc[i, "example " + str(index)] = sum_curr
                    df.loc[i, "time " + str(index)] = time_curr

                df.to_csv("temp.csv")
                print(df)
                print(df[["example " + str(index), "time " + str(index)]])
                # df.to_csv("temp.csv")
    df.to_csv("final_test.csv")

if __name__ == '__main__':
    freeze_support()
    text = "It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13\" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.  The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545.  Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha. The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individual. The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. What\'s More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back."
    # start = time.time()
    # print("FINAL 1: ", simplify_muss_multi(text, 2))
    # end = time.time()
    # print("Time 1: ", end-start)

    start = time.time()
    print("FINAL 2: ", simplify_muss(text, 2))
    print(multiprocessing.cpu_count())
    end = time.time()
    print("Time 2: ", end - start)


    # example = """The Lok Sabha is elected for a term of five years. Its life can be extended for one year at a time during a national emergency. It can be dissolved earlier than its term by the President on the advice of the Prime Minister. It can be voted out of power by a debate and vote on a no-confidence motion. During the 13" Lok Sabha, Bhartiya Janata Party lost a no~confidence motion by one vote and had to resign.  The House may have not more than 552 members; 530 elected from the states, 20 from Union Territories and not more than 2 members nominated from the Anglo-Indian Community. At present, the strength of the Lok Sabha is 545.  Election to the Lok Sabha is by universal adult franchise. Every Indian citizen above the age of 18 can vote for his/her representative in the Lok Sabha. The election is direct but by secret ballot, so that nobody is threatened or coerced into voting for a particular party or an individual. The Election Commission, an autonomous body elected by the President of India, organises, manages and oversees the entire process of election. What's More The provision for the Anglo-Indian community was included at the behest of the British Government to protect their nationals who had decided to stay back."""
    #
    # print("--------Original--------")
    # print("\n".join(sentence_tokenizer(example)))
    # print("--------1--------")
    # print(simplify_muss(example, 1))
    # print("--------2--------")
    # print(simplify_muss(example, 2))
    # print("--------3--------")
    # print(simplify_muss(example, 3))
    # print("--------4--------")
    # print(simplify_muss(example, 4))

    # start = time.time()
    # summary = simplify_muss(example, n=-1)
    # end = time.time()
    # print("TIME: ", end-start)
    # generate_results("results/muss-simp-results.csv")
    # df = pd.DataFrame(index=range(1, 13), columns=["summary", "time"])
    #
    # for i in range(1,12):
    #
    #     sum_curr, time_curr = run_and_get_time(example, 1)
    #     df.loc[i, "summary"] = sum_curr
    #     df.loc[i, "time"] = time_curr
    #
    # sum_all, time_all = run_and_get_time(example, -1)
    # df.loc[12, "summary"] = sum_all
    # df.loc[12, "time"] = time_all
    #
    # print(df)
    # df.to_csv("time_test.csv")

    # time_test()
    # run_and_get_time(example, 2)
    # run_and_get_time(example, 3)
    # run_and_get_time(example, 4)
    # run_and_get_time(example, 5)
    # run_and_get_time(example, 6)
    # run_and_get_time(example, 7)
    # run_and_get_time(example, 8)
    # run_and_get_time(example, 9)
    # run_and_get_time(example, 10)
    # run_and_get_time(example, 11)
    # run_and_get_time(example, -1)
    #


