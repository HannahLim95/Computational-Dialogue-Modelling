import pickle as pkl
import numpy as np
import os
import statsmodels.api as sm
import argparse
import more_itertools.more

from collections import Counter
import alignment_utils as au


def prep_apl(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between, lexical=True)

    if ARGS.analysis == 2:
        return pf_apl, pm_apl

    apl = pf_apl + pm_apl

    return apl


def calculate_alignment(apl, target_gender, prime_gender="f"):
    print(f"Calculating alignment for experiment {ARGS.analysis}, ",  
        f"t_{target_gender}" if ARGS.analysis == 1 else f"p_{prime_gender}_t_{target_gender}")

    z, p, b = au.calculate_beta(apl, ARGS.analysis)

    adj_pair_type = "between" if ARGS.between else "plain"
    results_filename = f"{SAVEDIR}/results_{adj_pair_type}.txt"
    betas_filename = f"{SAVEDIR}/betas_{adj_pair_type}.txt"

    au.print_results(z, p, results_filename, ARGS.analysis, target_gender, prime_gender)
    au.print_betas(b, betas_filename, ARGS.analysis, target_gender, prime_gender)


def main():
    au.prep_dataset(ARGS.dataset, between=ARGS.between, lexical=True)

    os.makedirs(SAVEDIR, exist_ok=True)

    for target_gender in ["m", "f"]:

        if ARGS.analysis == 1:
            adj_pair_list = prep_apl(target_gender)
            
            calculate_alignment(adj_pair_list, target_gender)

        else:
            female_prime_apl, male_prime_apl = prep_apl(target_gender)

            calculate_alignment(female_prime_apl, target_gender)

            calculate_alignment(male_prime_apl, target_gender, prime_gender="m")

    print("Done! Enjoy the results!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for lexical alignment.')

    parser.add_argument('dataset', type=str, choices=["AMI", "ICSI"],
						help="\"AMI\" or \"ICSI\", name of the dataset being analysed.")
    parser.add_argument('analysis', type=int, default=1, choices=[1, 2],
						help="1 or 2, experiment to perform")
    parser.add_argument('--between', type=bool, default=False,
                        help="bool to include the intermiediate adjacency pairs or not, default False")


    ARGS = parser.parse_args()

    print("Attention: Make sure you're in the 'Alignment' directory before running code!")
    print(ARGS)

    SAVEDIR = f"./lexical_{ARGS.dataset}"

    main()