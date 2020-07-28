import statsmodels.api as sm
import numpy as np
import pickle as pkl
import os

CATEGORIES = ['articles', 'pronoun', 'prepositions', 'negations', 'tentative', 'certainty', 'discrepancy', 'exclusive', 'inclusive']

######################################################
########## PREPROCESSING STEP FOR Xu et al. ##########
######################################################
def __between_split_list(split_list, prime_gender, lexical=False):
    expanded_split = list()
    b_str = f"list_{prime_gender}b"

    for adj_pair in split_list:
        if sum(adj_pair["b"].values()) > 0:
            if sum(adj_pair["a"].values()) > 0:
                if lexical:
                    keyz = list(adj_pair['a'].keys()) + list(adj_pair['b'].keys())
                    for k in keyz:
                        expanded_split.append((prime_gender, adj_pair['a'][k], adj_pair['b'][k], sum(adj_pair["a"].values())))
                else:
                    expanded_split.append((prime_gender, adj_pair["a"], adj_pair["b"], sum(adj_pair["a"].values())))

            for ab in adj_pair[b_str]:
                if sum(ab.values()) > 0:
                    if lexical:
                        keyz = list(ab.keys()) + list(adj_pair['b'].keys())
                        for k in keyz:
                            expanded_split.append((prime_gender, ab[k], adj_pair['b'][k], sum(adj_pair["a"].values())))
                    else:
                        expanded_split.append((prime_gender, ab, adj_pair["b"], sum(ab.values())))

    return expanded_split


def __plain_split_list(split_list, prime_gender, lexical=False):
    clean_split = list()

    for adj_pair in split_list:
        if sum(adj_pair["b"].values()) > 0 and sum(adj_pair["a"].values()) > 0:
            if lexical:
                keyz = list(adj_pair['a'].keys()) + list(adj_pair['b'].keys())
                for k in keyz:
                    clean_split.append((prime_gender, adj_pair['a'][k], adj_pair['b'][k], sum(adj_pair["a"].values())))
            else:
                clean_split.append((prime_gender, adj_pair["a"], adj_pair["b"], sum(adj_pair["a"].values())))

    return clean_split


def __prep_between_split_list(dataset, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"

    print("Please wait while I first prepare between split list...")
    with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    e_mf = __between_split_list(mf, "m", lexical)
    e_mm = __between_split_list(mm, "m", lexical)
    e_fm = __between_split_list(fm, "f", lexical)
    e_ff = __between_split_list(ff, "f", lexical)
    
    with open(f"../Corpora/{dataset}/{alignment_type}_processed_between_split.pkl", "wb") as f:
        pkl.dump((e_mf, e_mm, e_fm, e_ff), f)


def __prep_plain_split_list(dataset, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"

    print("Please wait while I first prepare plain split list...")
    with open(f"../Corpora/{dataset}/final_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    c_mf = __plain_split_list(mf, "m", lexical)
    c_mm = __plain_split_list(mm, "m", lexical)
    c_fm = __plain_split_list(fm, "f", lexical)
    c_ff = __plain_split_list(ff, "f", lexical)
    
    with open(f"../Corpora/{dataset}/{alignment_type}_processed_plain_split.pkl", "wb") as f:
        pkl.dump((c_mf, c_mm, c_fm, c_ff), f)


def prep_dataset(dataset, between, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"
    adj_pair_type = "between" if between else "plain"

    c_filename = f"../Corpora/{dataset}/{alignment_type}_processed_plain_split.pkl"
    e_filename = f"../Corpora/{dataset}/{alignment_type}_processed_between_split.pkl"

    if between and not os.path.exists(e_filename):
        __prep_between_split_list(dataset, lexical)
        
    elif not os.path.exists(c_filename):
        __prep_plain_split_list(dataset, lexical)


#############################################################################
def create_predictors(apl, analysis):
    print("Creating predictors...")

    y = [ 1 if target > 0 else 0 for _, _, target, _ in apl]

    if analysis == 1:
        c_count = [prime for _, prime, _, _ in apl]
        c_gender = [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl]
        c_plen = [plen for _, _, _, plen in apl]

        return c_count, c_gender, c_plen, y
    
    else:
        c_count = [prime for _, prime, _, _ in apl]

        return c_count, y


def calculate_beta(apl, analysis):
    if analysis == 1:
        n = 8
        c_count, c_gender, c_plen, y = create_predictors(apl, analysis)
    else:
        n = 2
        c_count, y = create_predictors(apl, analysis)
    
    pvalue_dict = {i: "undefined" for i in range(n)}
    zscores_dict = {i: "undefined" for i in range(n)}
    betas_dict = {i: "undefined" for i in range(n)}

    print("Calculating betas...")
    if sum(y) > 0:
        c_w = np.array(c_count)

        if analysis == 1:
            g_w = np.array(c_gender)
            X = np.array([np.ones(len(c_w)), c_w, g_w, c_plen, c_w*g_w, c_w*c_plen, g_w*c_plen, c_w*g_w*c_plen]).T
        else:
            X = np.array([np.ones(len(c_w)), c_w]).T

        y_w = np.array(y)

        res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
        
        p_values = res.pvalues
        for i, p in enumerate(p_values):
            pvalue_dict[i] = p

        z_scores = res.tvalues
        for i, z in enumerate(z_scores):
            zscores_dict[i] = z

        betas = res.params
        for i, b in enumerate(betas):
            betas_dict[i] = b

    return zscores_dict, pvalue_dict, betas_dict


def create_predictors_per_cat(apl):
    print("Creating predictors...")
    y = {w: [ 1 if target[w] > 0 else 0 for _, _, target, _ in apl] for w in CATEGORIES}

    c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in CATEGORIES}
    c_gender = {w: [ 1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl] for w in CATEGORIES}
    c_plen = [plen for _, _, _, plen in apl]

    return c_count, c_gender, c_plen, y


def calculate_beta_per_cat(apl):
    c_count, c_gender, c_plen, y = create_predictors_per_cat(apl)

    pvalue_dict = {w: {i: "undefined" for i in range(8)} for w in CATEGORIES}
    zscores_dict = {w: {i: "undefined" for i in range(8)} for w in CATEGORIES}
    betas_dict = {w: {i: "undefined" for i in range(8)} for w in CATEGORIES}

    print("Calculating betas...")
    for w in CATEGORIES:
        c_w = c_count[w]
        g_w = c_gender[w]
        y_w = y[w]

        if sum(y_w) > 0:
            c_w = np.array(c_w)
            g_w = np.array(g_w)
            X = np.array([np.ones(len(c_w)), c_w, g_w, c_plen, c_w*g_w, c_w*c_plen, g_w*c_plen, c_w*g_w*c_plen]).T
    
            y_w = np.array(y_w)

            res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
            
            p_values = res.pvalues
            for i, p in enumerate(p_values):
                pvalue_dict[w][i] = p

            z_scores = res.tvalues
            for i, z in enumerate(z_scores):
                zscores_dict[w][i] = z

            betas = res.params
            for i, b in enumerate(betas):
                betas_dict[w][i] = b
    return zscores_dict, pvalue_dict, betas_dict


def print_betas(b, filename, analysis, target_gender, prime_gender):
    betas = [0, 1] if analysis == 2 else list(range(8))

    with open(filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EXPERIMENT: {analysis}\nTARGET GENDER: {target_gender}\n")

        if analysis == 2: f.write(f"PRIME GENDER: {prime_gender}\n")

        f.write("==================================\n")

        for bb in betas:
            f.write(f"\tBETA_{bb}")

        f.write("\n----------------------------------\n")

        for bb in betas:
            f.write(f"\t{b[bb]:.3f}")
        f.write(f"\n")


def print_zscores(z, p, filename, analysis, target_gender, prime_gender):
    betas = [0, 1] if analysis == 2 else list(range(8))

    with open(filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EXPERIMENT: {analysis}\nTARGET GENDER: {target_gender}\n")

        if analysis == 2: f.write(f"PRIME GENDER: {prime_gender}\n")

        f.write("==================================\n")

        f.write("BETA:\tZ-SCORES\tP-VALUES\n")
        f.write("----------------------------------\n")

        for bb in betas:
            if p[bb] < 0.001: 
                f.write(">>> ")
            elif p[bb] < 0.01:
                f.write(">> ")
            elif p[bb] < 0.05:
                f.write("> ")

            f.write(f"{bb}\t{z[bb]:.3f}\t{p[bb]:.3f}\n")


def print_zscores_per_cat(z, p, filename, analysis, target_gender, prime_gender):
    betas = [0, 1] if analysis == 2 else list(range(8))

    with open(filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EXPERIMENT: {analysis}\nTARGET GENDER: {target_gender}\n")

        if analysis == 2: f.write(f"PRIME GENDER: {prime_gender}\n")

        f.write("==================================\n")

        for bb in betas:
            f.write(f"\nBETA: {bb}\n")

            f.write("CATEGORIES:\tZ-SCORES\tP-VALUES\n")
            f.write("----------------------------------\n")

            for w in z:
                if p[w][bb] < 0.05: f.write(">> ")
                f.write(f"{w}:\t{z[w][bb]:.3f}\t{p[w][bb]:.3f}\n")


def print_betas_per_cat(b, filename, analysis, target_gender, prime_gender):
    betas = [0, 1] if analysis == 2 else list(range(8))

    with open(filename, "a") as f:
        f.write("\n==================================\n")
        f.write(f"EXPERIMENT: {analysis}\nTARGET GENDER: {target_gender}\n")

        if analysis == 2: f.write(f"PRIME GENDER: {prime_gender}\n")

        f.write("==================================\n")

        f.write("CATEGORIES:")
        for bb in betas:
            f.write(f"\tBETA_{bb}")

        f.write("\n----------------------------------\n")

        for w in b:
            f.write(f"{w}")
            for bb in betas:
                f.write(f"\t{b[w][bb]:.3f}")
            f.write(f"\n")


def create_vocab(adj_pair_list):
    print("Creating vocab...")
    vocab = set()

    for _, prime, target, _ in adj_pair_list:
        vocab |= set(prime.keys())
        vocab |= set(target.keys())

    return vocab


def prime_lists(target_gender, dataset, between, lexical=False):
    alignment_type = "lexical" if lexical else "stylistic"
    adj_pair_type = "between" if between else "plain"

    with open(f"../Corpora/{dataset}/{alignment_type}_processed_{adj_pair_type}_split.pkl", "rb") as f:
        mf, mm, fm, ff = pkl.load(f)

    if target_gender == "m":
        return fm, mm
    else:
        return ff, mf
