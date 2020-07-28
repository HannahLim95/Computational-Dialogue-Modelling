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

    x_train_count = np.concatenate([c_count_train, y_count_train], axis=1)

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

    x_test_count = np.concatenate([c_count_test, y_count_test], axis=1)

    beta_0 =betas[category][0]
    beta_1 =betas[category][1]
    beta_2 =betas[category][2]
    beta_3 =betas[category][3]

    #y as not binary
    y_train = y_count_train/c_count_train
    y_train[y_train == inf] = 0
    y_train = np.nan_to_num(y_train)

    # y as not binary
    y_test = y_count_test / c_count_test
    y_test[y_test == inf] = 0
    y_test = np.nan_to_num(y_test)

    # using the y
    x_train = (y_train-beta_0-beta_1*c_count_train)/((beta_2+beta_3*c_count_train))
    x_test = (y_test - beta_0 - beta_1 * c_count_test) / ((beta_2 + beta_3 * c_count_test))


    # OLD Y
    # x = (y_count_train-beta_0-beta_1*c_count_train)/((beta_2+beta_3*c_count_train))

    print('male train',np.mean(x_train[:int(len(x_train)/2)]))
    print('female train',np.mean(x_train[int(len(x_train)/2):]))
    print('male test', np.mean(x_test[:int(len(x_test) / 2)]))
    print('female test', np.mean(x_test[int(len(x_test) / 2):]))

    # x_train = np.concatenate([x_train, x_train_count], axis=1)
    # x_test = np.concatenate([x_test, x_test_count], axis=1)
    return x_train, x_train_count, x_test, x_test_count


adjacency_pairs_list = pd.read_pickle('/Users/hannahlim/Documents/GitHub/CDM/Alignment/stylistic_AMI/orig/t_m_prepped_apl.pkl')
c_count, c_gender, y, betas = get_dataset(adjacency_pairs_list)

CATEGORIES = ['articles', 'pronoun', 'prepositions', 'negations', 'tentative', 'certainty', 'discrepancy', 'exclusive', 'inclusive']
X=[]
X_test = []
for w in CATEGORIES:
    # print(type(get_predictor(c_count, c_gender, y, betas, w)))
    # x = list(get_predictor(c_count, c_gender, y, betas, w))
    x_train, x_train_count, x_test, x_test_count = get_predictor(c_count, c_gender, y, betas, w)
    print(x_train[:10])
    print(np.shape(x_train))
    x_train = list(x_train.T)
    x_test = list(x_test.T)
    x_train_count = list(x_train_count.T)
    x_test_count = list(x_test_count.T)
    # print(x_test[:10])
    print(x_train[:10])
    X.append(x_train[0])
    X.append(x_train_count[0])
    X_test.append(x_test[0])
    X_test.append(x_test_count[0])

X=np.asarray(X)
print('shape',np.shape(X))

# trainset
indices_male = [i for i, x in enumerate(c_gender['pronoun']) if x == 1]
indices_female = [i for i, x in enumerate(c_gender['pronoun']) if x == 0]

indices_male=indices_male[:len(indices_female)]

len_train_male = int(np.round(0.8 * len(indices_male)))
len_test_male = len(indices_male) - len_train_male
len_train_female = int(np.round(0.8 * len(indices_female)))
len_test_female = len(indices_female) - len_train_female

c_gender_female = itemgetter(*indices_female)(c_gender['pronoun'])
c_gender_male = itemgetter(*indices_male)(c_gender['pronoun'])

c_gender_train_female = np.array(c_gender_female[:len_train_female])
c_gender_train_male = np.array(c_gender_male[:len_train_male])

y_train = np.concatenate([c_gender_train_female, c_gender_train_male])
y_train=list(y_train)

# test set
# c_count_test_female = np.array(c_count_female[len_train_female:])
# c_count_test_male = np.array(c_count_male[len_train_male:])
#
# c_count_test = np.concatenate([c_count_test_female, c_count_test_male])
#
# y_count_test_female = np.array(y_count_female[len_train_female:])
# y_count_test_male = np.array(y_count_male[len_train_male:])
#
# y_count_test = np.concatenate([y_count_test_female, y_count_test_male])
#
# c_count_test = np.reshape(c_count_test, (len(c_count_test), 1))
# y_count_test = np.reshape(y_count_test, (len(y_count_test), 1))

c_gender_test_female = np.array(c_gender_female[len_train_female:])
c_gender_test_male = np.array(c_gender_male[len_train_male:])

y_test = np.concatenate([c_gender_test_female, c_gender_test_male])
y_test=list(y_test)
# x_test = np.concatenate([c_count_test, y_count_test], axis=1)
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 6), random_state=1, activation='relu')

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
X = np.array(X).T

clf.fit(X, y_train)
print(clf)


X_test = np.array(X_test).T
y_pred = clf.predict(X_test)
print(y_pred[:10])
print(type(y_pred))
print(y_test[:10])
print(type(y_test))

y_test = np.asarray(y_test)
print(type(y_test))
acc = accuracy_score(y_test, y_pred)
print('acc', acc)

print(np.sum(y_pred))
print(len(y_pred))
print(np.sum(y_test))