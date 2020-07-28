from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle as pkl
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from operator import itemgetter
from gekko import GEKKO
from numpy import inf


from sklearn.neural_network import MLPClassifier
CATEGORIES = ['articles', 'pronoun', 'prepositions', 'negations', 'tentative', 'certainty', 'discrepancy', 'exclusive', 'inclusive']

def create_predictors(apl, eq, normalise=False):
    y = {w: [1 if target[w] > 0 else 0 for _, _, target, _ in apl] for w in CATEGORIES}

    if eq == 1:
        c_count = {w: [prime[w] / plen if normalise else prime[w] for _, prime, _, plen in apl] for w in CATEGORIES}
        c_gender = {w: [1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl] for w in CATEGORIES}

        return c_count, c_gender, y

    elif eq == 2:
        c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in CATEGORIES}

        return c_count, y

    elif eq == 3:
        c_count = {w: [prime[w] for _, prime, _, _ in apl] for w in CATEGORIES}
        c_gender = {w: [1 if prime_gender == "m" else 0 for prime_gender, _, _, _ in apl] for w in CATEGORIES}
        c_plen = [plen for _, _, _, plen in apl]

        return c_count, c_gender, c_plen, y

# THIS IS ONLY FOR TARGET MALE

def get_dataset(adjacency_pairs_list):
    normalise = False
    c_count, c_gender, y = create_predictors(adjacency_pairs_list, 1, normalise)
    betas={}
    for w in CATEGORIES:
        c_w = c_count[w]
        g_w = c_gender[w]
        y_w = y[w]

        if sum(y_w) > 0:
            c_w = np.array(c_w)
            g_w = np.array(g_w)
            X = np.array([np.ones(len(c_w)), c_w, g_w, c_w * g_w]).T

            y_w = np.array(y_w)

            res = sm.GLM(y_w, X, family=sm.families.Binomial()).fit()
            # print('params', w, res.params)
            betas[w] = res.params
    return c_count, c_gender, y, betas

def get_predictor(c_count, c_gender, y, betas, category):
    # print(betas[category])
    # To create balanced datasets
    indices_male = [i for i, x in enumerate(c_gender[category]) if x == 1]
    indices_female = [i for i, x in enumerate(c_gender[category]) if x == 0]

    #to make the amount of males and females in the dataset even
    indices_male=indices_male[:len(indices_female)]

    len_train_male = int(np.round(0.8 * len(indices_male)))
    len_test_male = len(indices_male) - len_train_male
    len_train_female = int(np.round(0.8 * len(indices_female)))
    len_test_female = len(indices_female) - len_train_female

    c_count_female = itemgetter(*indices_female)(c_count[category])
    c_count_male = itemgetter(*indices_male)(c_count[category])

    y_count_female = itemgetter(*indices_female)(y[category])
    y_count_male = itemgetter(*indices_male)(y[category])

    c_gender_female = itemgetter(*indices_female)(c_gender[category])
    c_gender_male = itemgetter(*indices_male)(c_gender[category])

    #creating the trainset
    c_count_train_female = np.array(c_count_female[:len_train_female])
    c_count_train_male = np.array(c_count_male[:len_train_male])

    c_count_train = np.concatenate([c_count_train_female, c_count_train_male])

    y_count_train_female = np.array(y_count_female[:len_train_female])
    y_count_train_male = np.array(y_count_male[:len_train_male])

    y_count_train = np.concatenate([y_count_train_female, y_count_train_male])

    c_count_train = np.reshape(c_count_train, (len(c_count_train), 1))
    y_count_train = np.reshape(y_count_train, (len(y_count_train), 1))

    c_gender_train_female = np.array(c_gender_female[:len_train_female])
    c_gender_train_male = np.array(c_gender_male[:len_train_male])

    c_gender_train = np.concatenate([c_gender_train_female, c_gender_train_male])

    x_train = np.concatenate([c_count_train, y_count_train], axis=1)

    #creating the test set
    c_count_test_female = np.array(c_count_female[len_train_female:])
    c_count_test_male = np.array(c_count_male[len_train_male:])

    c_count_test = np.concatenate([c_count_test_female, c_count_test_male])

    y_count_test_female = np.array(y_count_female[len_train_female:])
    y_count_test_male = np.array(y_count_male[len_train_male:])

    y_count_test = np.concatenate([y_count_test_female, y_count_test_male])

    c_count_test = np.reshape(c_count_test, (len(c_count_test), 1))
    y_count_test = np.reshape(y_count_test, (len(y_count_test), 1))

    c_gender_test_female = np.array(c_gender_female[len_train_female:])
    c_gender_test_male = np.array(c_gender_male[len_train_male:])

    c_gender_test = np.concatenate([c_gender_test_female, c_gender_test_male])

    x_test = np.concatenate([c_count_test, y_count_test], axis=1)

    # print(betas[category])
    beta_0 =betas[category][0]
    beta_1 =betas[category][1]
    beta_2 =betas[category][2]
    beta_3 =betas[category][3]

    # y_count_train-beta_0-beta_1*c_count_train = beta_2*x+beta_3*c_count_train*x
    y = y_count_train/c_count_train
    # print(y[:10])
    # print(y_count_train[:10])
    # print(c_count_train[:10])
    # print(type(y))

    y[y == inf] = 0
    y = np.nan_to_num(y)

    # print(y[:10])

     # x(beta_2+beta_3*c_count_train)
    x = (y-beta_0-beta_1*c_count_train)/((beta_2+beta_3*c_count_train))

    # x = (y_count_train-beta_0-beta_1*c_count_train)/((beta_2+beta_3*c_count_train))
    # print(type(x))
    # print(len(x)/2)
    print('male',np.mean(x[:int(len(x)/2)]))
    print('female',np.mean(x[int(len(x)/2):]))
    # print(x[0:10])

    m = GEKKO()
    x=m.Var()
    m.Equation(y_count_train[0][0]-beta_0-beta_1*c_count_train[0][0] == beta_2*x+beta_3*c_count_train[0][0]*x)
    m.solve(disp=False)
    print('x value',x.value)
    print('gender', c_gender_train[0])
    # equation =beta_0 + beta_1*c_count_train + beta_2 * unknown_yet + beta_3 * c_count_train * unknown_yet
    print('len trainset',len(c_gender_train))
    m = GEKKO()
    x=m.Var()
    m.Equation(y_count_train[10000][0]-beta_0-beta_1*c_count_train[10000][0] == beta_2*x+beta_3*c_count_train[10000][0]*x)
    m.solve(disp=False)
    print('x value',x.value)
    print('gender', c_gender_train[10000])


    clf = LogisticRegression(random_state=0).fit(x_train, c_gender_train)

    print(clf)
    prediction = clf.predict(x_test)

    acc = accuracy_score(c_gender_test, prediction)
    print('acc', acc)

    return 'hi'

adjacency_pairs_list = pd.read_pickle('/Users/hannahlim/Documents/GitHub/CDM/Alignment/stylistic_AMI/orig/t_m_prepped_apl.pkl')
c_count, c_gender, y, betas = get_dataset(adjacency_pairs_list)
print(betas)
CATEGORIES = ['articles', 'pronoun', 'prepositions', 'negations', 'tentative', 'certainty', 'discrepancy', 'exclusive', 'inclusive']
for w in CATEGORIES:
    joehoe = get_predictor(c_count, c_gender, y, betas, w)



