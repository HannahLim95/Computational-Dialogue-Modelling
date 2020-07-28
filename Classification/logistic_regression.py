from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle as pkl
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from operator import itemgetter

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
    print(betas[category])
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


#
# #TODO: what is gender =1, male or female
# indices_male = [i for i, x in enumerate(c_gender['inclusive']) if x == 1]
# indices_female = [i for i, x in enumerate(c_gender['inclusive']) if x == 0]
# len_train_male = int(np.round(0.8*len(indices_male)))
# len_test_male = len(indices_male) - len_train_male
#
# len_train_female = int(np.round(0.8*len(indices_female)))
# len_test_female = len(indices_female) - len_train_female
#
# c_count_female = itemgetter(*indices_female)(c_count['inclusive'])
# c_count_male = itemgetter(*indices_male)(c_count['inclusive'])
#
# y_count_female = itemgetter(*indices_female)(y['inclusive'])
# y_count_male = itemgetter(*indices_male)(y['inclusive'])
#
# c_gender_female = itemgetter(*indices_female)(c_gender['inclusive'])
# c_gender_male = itemgetter(*indices_male)(c_gender['inclusive'])
#
# #train set
# c_count_inclusive_train_female = np.array(c_count_female[:len_train_female])
# c_count_inclusive_train_male = np.array(c_count_male[:len_train_male])
# print('len before',len(c_count_inclusive_train_female))
# print('len bef',len(c_count_inclusive_train_male))
#
# c_count_inclusive_train = np.concatenate([c_count_inclusive_train_female, c_count_inclusive_train_male])
# print('len after',len(c_count_inclusive_train))
#
# y_count_inclusive_train_female = np.array(y_count_female[:len_train_female])
# y_count_inclusive_train_male = np.array(y_count_male[:len_train_male])
#
# y_count_inclusive_train = np.concatenate([y_count_inclusive_train_female, y_count_inclusive_train_male])
#
#
# # c_count_inclusive_train = np.array(c_count['inclusive'][:len_train])
# # y_count_inclusive_train = np.array(y['inclusive'][:len_train])
# print(np.shape(c_count_inclusive_train))
# c_count_inclusive_train=np.reshape(c_count_inclusive_train, (len(c_count_inclusive_train), 1))
# y_count_inclusive_train=np.reshape(y_count_inclusive_train, (len(y_count_inclusive_train), 1))
#
# # c_gender_train = c_gender['inclusive'][:len_train]
#
# c_gender_train_female = np.array(c_gender_female[:len_train_female])
# c_gender_train_male = np.array(c_gender_male[:len_train_male])
#
# c_gender_train = np.concatenate([c_gender_train_female, c_gender_train_male])
#
# x_inclusive_train = np.concatenate([c_count_inclusive_train, y_count_inclusive_train], axis =1)
#
#
# #test set
#
# c_count_inclusive_test_female = np.array(c_count_female[len_train_female:])
# c_count_inclusive_test_male = np.array(c_count_male[len_train_male:])
# print('len before',len(c_count_inclusive_test_female))
# print('len bef',len(c_count_inclusive_test_male))
#
# c_count_inclusive_test = np.concatenate([c_count_inclusive_test_female, c_count_inclusive_test_male])
# print('len after',len(c_count_inclusive_test))
#
# y_count_inclusive_test_female = np.array(y_count_female[len_train_female:])
# y_count_inclusive_test_male = np.array(y_count_male[len_train_male:])
#
# y_count_inclusive_test = np.concatenate([y_count_inclusive_test_female, y_count_inclusive_test_male])
#
# c_count_inclusive_test=np.reshape(c_count_inclusive_test, (len(c_count_inclusive_test), 1))
# y_count_inclusive_test=np.reshape(y_count_inclusive_test, (len(y_count_inclusive_test), 1))
#
# c_gender_test_female = np.array(c_gender_female[len_train_female:])
# c_gender_test_male = np.array(c_gender_male[len_train_male:])
#
# c_gender_test = np.concatenate([c_gender_test_female, c_gender_test_male])
#
# x_inclusive_test = np.concatenate([c_count_inclusive_test, y_count_inclusive_test], axis =1)
#
#
# # c_count_inclusive_test = np.array(c_count['inclusive'][len_train:])
# # y_count_inclusive_test = np.array(y['inclusive'][len_train:])
# #
# # c_count_inclusive_test=np.reshape(c_count_inclusive_test, (len_test, 1))
# # y_count_inclusive_test=np.reshape(y_count_inclusive_test, (len_test, 1))
# #
# # c_gender_test = c_gender['inclusive'][len_train:]
# #
# # x_inclusive_test = np.concatenate([c_count_inclusive_test, y_count_inclusive_test], axis =1)
# #
#
# # logistic regression function inclusive
#
# print(c_gender_train)
# clf = LogisticRegression(random_state=0).fit(x_inclusive_train, c_gender_train)
#
# print(clf)
#
# prediction = clf.predict(x_inclusive_test)
# # print(prediction)
#
# acc = accuracy_score(c_gender_test, prediction)
# print('acc',acc)
#
#
#
#







#now it is just a simple classifier that does not especially look at the alignment
# # TODO: classify using the alignment coefficients
#
# # this is a tuple containing lists of dictionary
# x = pd.read_pickle('/Users/hannahlim/Documents/GitHub/CDM/Corpora/ICSI/in_between_split.pkl')
#
# f = open("/Users/hannahlim/Documents/GitHub/CDM/Alignment/stylistic_ICSI/results_orig.txt", "r")
# # print(f.read())
#
# results = f.read()
# print(type(results))
# print(len(results))
# results = pd.read_pickle('/Users/hannahlim/Documents/GitHub/CDM/Alignment/stylistic_ICSI/results_orig.txt')
# print(type(results))

# input prime female and target male with the output binary
# x_m_f_b  = x[2]
# print('type', type(x_m_f_b))
# nr_m_f_b = len(x_m_f_b)
# y_m_f_b = [1]*nr_m_f_b


# input other than prime female and target male with the output binary
# x_other = x[0]+x[1]+x[3]
# nr_other = len(x_other)
# y_other = [0] * nr_other
#
# print(len(x_m_f_b))
# print(x_m_f_b[0])
#
# print(type(x_m_f_b[0]))
# print(x_m_f_b[0]['a'])
# print(x_m_f_b[0]['b'])
# print(type(x_m_f_b[0]['a']))


# fake_dict = x_m_f_b[0]
# print(fake_dict)
# a = list(fake_dict.keys())
# print('real keys', a)
# fake_dict.pop('mb')
# fake_dict.pop('fb')
# fake_dict.pop('n_mb')
# fake_dict.pop('n_fb')
#
# print('remove mb',fake_dict)
# a = list(fake_dict.keys())
# print(a)
# fake_dict = fake_dict.pop('fb')
# print(fake_dict)

#
# X, y = load_iris(return_X_y=True)
# clf = LogisticRegression(random_state=0).fit(X, y)
# prediction = clf.predict(X[:2, :])
# print(prediction)
# prediction_proba = clf.predict_proba(X[:2, :])
# print(prediction_proba)
# score = clf.score(X, y)
# print(score)