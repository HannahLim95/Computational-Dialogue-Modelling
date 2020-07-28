import os
import argparse
import pickle as pkl
from collections import Counter
import alignment_utils as au
import statsmodels.api as sm
import numpy as np



def create_liwc(vocab):
    """
    Parameters:
        vocab - set of vocab

    Returns:
        dictionary of liwc categories (key) and list of its words (value)
    """
    with open("liwc.pkl", "rb") as f:
        liwc = pkl.load(f)

    for cat in liwc:
        extension = list()
        asterisk = set()
        
        for w in liwc[cat]:
            if "*" in w:
                asterisk.add(w)
                extension.extend([vw for vw in vocab if (w == vw or vw.startswith(w[:-1]))])
        liwc[cat] |= (set(extension))
        liwc[cat] -= asterisk     

    return liwc, ['articles', 'pronoun', 'prepositions', 'negations', 'tentative', 'certainty', 'discrepancy', 'exclusive', 'inclusive']


def create_apl(liwc, cat_set, orig_apl):
    print("Creating LIWC adjacency pairs...")
    
    inv_ind = {w: cat for cat in liwc for w in liwc[cat]}

    liwc_apl = list()
    for g, p, t, plen in orig_apl:
        new_p = Counter([inv_ind[w] for w in p.elements() if w in inv_ind])
        for m in cat_set ^ set(new_p.elements()):
            new_p[m] = 0

        new_t = Counter([inv_ind[w] for w in t.elements() if w in inv_ind])
        for m in cat_set ^ set(new_t.elements()):
            new_t[m] = 0

        liwc_apl.append((g, new_p, new_t, plen))

    return liwc_apl


def prep_apl(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between)
    
    apl = pf_apl + pm_apl
    
    vocab = au.create_vocab(apl)

    liwc, cat = create_liwc(vocab)

    liwc_apl = create_apl(liwc, set(cat), apl)

    combined_apl = list()
    for g, p, t, l in liwc_apl:
        for k in cat:
            combined_apl.append((g, p[k], t[k], l))

    return combined_apl


def prep_apl_two(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between)
    
    female_prime_vocab = au.create_vocab(pf_apl)
    male_prime_vocab = au.create_vocab(pm_apl)

    female_prime_liwc, cat = create_liwc(female_prime_vocab)
    male_prime_liwc, cat = create_liwc(male_prime_vocab)


    female_prime_stylistic_apl = create_apl(female_prime_liwc, set(cat), pf_apl)
    male_prime_stylistic_apl = create_apl(male_prime_liwc, set(cat), pm_apl)

    combined_apl_f = list()
    for g, p, t, l in female_prime_stylistic_apl:
        for k in cat:
            combined_apl_f.append((g, p[k], t[k], l))

    combined_apl_m = list()
    for g, p, t, l in male_prime_stylistic_apl:
        for k in cat:
            combined_apl_m.append((g, p[k], t[k], l))


    return combined_apl_f, combined_apl_m


def prep_apl_per_cat(target_gender):
    pf_apl, pm_apl = au.prime_lists(target_gender, ARGS.dataset, ARGS.between)
    
    apl = pf_apl + pm_apl
    
    vocab = au.create_vocab(apl)

    liwc, cat = create_liwc(vocab)

    liwc_apl = create_apl(liwc, set(cat), apl)

    return liwc_apl


def calculate_alignment(apl, target_gender, prime_gender="f"):
    print(f"Calculating alignment for experiment {ARGS.analysis} ",  
        f"t_{target_gender}" if ARGS.analysis == 1 else f"p_{prime_gender}_t_{target_gender}")
    
    z, p, b = au.calculate_beta(apl, ARGS.analysis)

    adj_pair_type = "between" if ARGS.between else "plain"
    results_filename = f"{SAVEDIR}/results_{adj_pair_type}.txt"
    betas_filename = f"{SAVEDIR}/betas_{adj_pair_type}.txt"

    au.print_zscores(z, p, results_filename, ARGS.analysis, target_gender, prime_gender)
    au.print_betas(b, betas_filename, ARGS.analysis, target_gender, prime_gender)


def calculate_alignment_per_cat(apl, target_gender, prime_gender="f"):
    print(f"Calculating alignment per category for experiment {ARGS.analysis} t_{target_gender}")

    z, p, b = au.calculate_beta_per_cat(apl)

    adj_pair_type = "between" if ARGS.between else "plain"
    results_filename = f"{SAVEDIR}/cat_results_{adj_pair_type}.txt"
    betas_filename = f"{SAVEDIR}/cat_betas_{adj_pair_type}.txt"
    
    au.print_zscores_per_cat(z, p,results_filename, ARGS.analysis, target_gender, prime_gender)
    au.print_betas_per_cat(b, betas_filename, ARGS.analysis, target_gender, prime_gender)


def main():
    au.prep_dataset(ARGS.dataset, between=ARGS.between)

    os.makedirs(SAVEDIR, exist_ok=True)

    for target_gender in ["m", "f"]:
        if ARGS.analysis == 1:
            if ARGS.cat:
                adj_pair_list = prep_apl_per_cat(target_gender)

                calculate_alignment_per_cat(adj_pair_list, target_gender)
            else:
                adj_pair_list = prep_apl(target_gender)

                calculate_alignment(adj_pair_list, target_gender)
        else:
            female_prime_apl, male_prime_apl = prep_apl_two(target_gender)

            calculate_alignment(female_prime_apl, target_gender)
            calculate_alignment(male_prime_apl, target_gender, prime_gender="m")

    print("Done! Enjoy the results!")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments for stylistic alignment.')
    parser.add_argument('dataset', type=str, choices=["AMI", "ICSI"],
                        help="\"AMI\" or \"ICSI\", name of the dataset being analysed")
    parser.add_argument('analysis', type=int, default=1, choices=[1, 2],
						help="1 or 2, experimet to perform")
    parser.add_argument('--between', type=bool, default=False,
                        help="bool to include the intermiediate utterances or not, default False")
    parser.add_argument('--cat', type=bool, default=False,
                        help='bool to compute beta values per LIWC category, default False -- only for experiment 1')

    ARGS = parser.parse_args()
    
    print("Attention: Make sure you're in the 'Alignment' directory before running code!")
    print(ARGS)

    SAVEDIR = f"./stylistic_{ARGS.dataset}"

    main()