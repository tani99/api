
import pandas as pd
directory = '../examples'

def get_sum_simp():
    # Sum: edmundsons key_length_title
    # Simp: muss group 2
    edmundsons_results = pd.read_csv("../summarisation/edmundsons_results_combinations_without_cue.csv")
    sum_simp_results = pd.read_csv("../sum_and_simp/sum-simp-results.csv")
    muss_results = pd.read_csv("../simplification/muss_simp/results/muss-simp-results.csv")
    simp_sum_results = pd.read_csv("../sum_and_simp/simp-sum-results.csv")

    df = pd.DataFrame(index=range(0, 10), columns=["original", "sum", "simp", "sum_simp", "simp_sum"])

    for i in range(0, 10):
        df.loc[i, "original"] = muss_results.loc[i, "original"]
        df.loc[i, "sum"] = edmundsons_results.loc[i, "key_title_length"]
        df.loc[i, "simp"] = muss_results.loc[i, "Group 2"]
        df.loc[i, "sum_simp"] = sum_simp_results.loc[i, "sum_simp_muss"]
        print(simp_sum_results.columns)
        df.loc[i, "simp_sum"] = simp_sum_results.loc[i, "simp-sum_muss"]

    print(df)
    df.to_csv("sum_simp_combined-form.csv")

def get_muss_control():
    # Only groups of 2
    muss_results = pd.read_csv("../simplification/muss_simp/results/muss-simp-results.csv")
    control_results = pd.read_csv("../simplification/control_simp/results/control-simp-results.csv")

    df = pd.DataFrame(index=range(0,10), columns=["original", "muss", "control"])

    for i in range(0, 10):
        df.loc[i, "original"] = muss_results.loc[i, "original"]
        df.loc[i, "muss"] = muss_results.loc[i, "Group 2"]
        df.loc[i, "control"] = control_results.loc[i, "Group 2"]

    print(df)
    df.to_csv("muss-control-form.csv")

def get_muss_groups():
    muss_results = pd.read_csv("../simplification/muss_simp/results/muss-simp-results.csv")
    print(muss_results)
    muss_results.to_csv("muss-groups-form.csv")


def get_control_groups():
    control_results = pd.read_csv("../simplification/control_simp/results/control-simp-results.csv")
    print(control_results)
    control_results.to_csv("control-groups-form.csv")

if __name__ == '__main__':
    get_sum_simp()
    get_muss_control()
    get_muss_groups()
    get_control_groups()
