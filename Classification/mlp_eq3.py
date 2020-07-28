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
from sklearn.utils import resample

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


def get_dataset(adjacency_pairs_list):
    normalise = False
    c_count, c_gender, c_plen, y = create_predictors(adjacency_pairs_list, 3, normalise)

    #TODO compute these beta's based on the training set only
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
    return c_count, c_gender, c_plen, y, betas

#TODO preparing train and test set
adjacency_pairs_list = pd.read_pickle('/Users/hannahlim/Documents/GitHub/CDM/Alignment/stylistic_AMI/orig/t_m_prepped_apl.pkl')

c_count, c_gender, c_plen, target_count = create_predictors(adjacency_pairs_list, 3, False)

# creating dataframe for upsampling
gender_prime = c_gender['articles']
x = [c_plen, gender_prime]
x_columns = ['c_plen', 'gender_prime']
for w in CATEGORIES:
    c_count_w = c_count[w]
    x.append(c_count_w)
    x_columns.append('c_count_'+w)
    target_count_w = target_count[w]
    x.append(target_count_w)
    x_columns.append('y_count_'+w)
x=np.array(x).T
df = pd.DataFrame(x)
df.columns=x_columns

# Separate majority and minority classes
df_minority = df[df.gender_prime == 0]
df_majority = df[df.gender_prime == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,  # sample with replacement
                                 n_samples=len(df_majority),  # to match majority class
                                 random_state=123)  # reproducible results

# Combine majority class with upsampled minority class

print(len(df_majority))
print(len(df_minority_upsampled))

df_upsampled = pd.concat([df_majority, df_minority_upsampled])
# print(len(df_upsampled))
#random shuffle dataset
df = df_upsampled.sample(frac=1)

y=df['gender_prime']
df=df.drop(columns='gender_prime')
df=df.drop(columns='c_plen')
x=df
print(x)
len_train = int(np.round(0.8*len(y)))

#
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5,2), random_state=1, activation='relu')
#
X = np.array(x[:len_train])
print(np.shape(X))
y_train=np.array(y[:len_train])
print(np.shape(y))

#
clf.fit(X, y_train)
print(clf)
#
#
X_test = np.array(x[len_train:])
y_test = np.array(y[len_train:])
#
y_pred = clf.predict(X_test)
print(y_pred)
print(np.sum(y_pred))
print(y_test)

# y_test = np.asarray(gender_prime[len_train:])
acc = accuracy_score(y_test, y_pred)
print(acc)
# #
print(np.sum(y_pred))
print(len(y_pred))
print(np.sum(y_test)/len(y_test))

print(np.sum(y_train)/len(y_train))









#
#
#
# gender_prime = c_gender['articles']
# print(len(gender_prime))
#
# indices_male = [i for i, x in enumerate(gender_prime) if x == 1]
# indices_female = [i for i, x in enumerate(gender_prime) if x == 0]
#
# # to make the amount of males and females in the dataset even
# indices_male = indices_male[:len(indices_female)]
#
# len_train = int(np.round(0.8*len(gender_prime)))
# print(len_train)
#
# x = [c_plen[:len_train]]
# x_test = [c_plen[len_train:]]
# for w in CATEGORIES:
#     c_count_w = c_count[w]
#     x.append(c_count_w[:len_train])
#     x_test.append(c_count_w[len_train:])
#     target_count_w = target_count[w]
#     x.append(target_count_w[:len_train])
#     x_test.append(target_count_w[len_train:])
#
#
#
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 6), random_state=1, activation='relu')
#
# X = np.array(x).T
#
# clf.fit(X, gender_prime[:len_train])
#
#
# X_test = np.array(x_test).T
#
# y_pred = clf.predict(X_test)
# y_test = np.asarray(gender_prime[len_train:])
# acc = accuracy_score(y_test, y_pred)
# print(acc)
#
# print(np.sum(y_pred))
# print(len(y_pred))
# print(np.sum(y_test))
#


#
#
# print(len(c_count['articles']))
# print(len(c_plen))






















#
# # To create balanced datasets
# indices_male = [i for i, x in enumerate(c_gender['articles']) if x == 1]
# indices_female = [i for i, x in enumerate(c_gender['articles']) if x == 0]
# # print(len(indices_male))
# # print(len(indices_female))
#
# # to make the amount of males and females in the dataset even
# indices_male = indices_male[:len(indices_female)]
#
# len_train_male = int(np.round(0.8 * len(indices_male)))
# # len_test_male = len(indices_male) - len_train_male
# len_train_female = int(np.round(0.8 * len(indices_female)))
# # len_test_female = len(indices_female) - len_train_female
#
#
# c_count_train_set =[]
# y_count_train_set=[]
#
# test_set=[]
#
# for w in CATEGORIES:
#     c_count_female = itemgetter(*indices_female)(c_count[w])
#     c_count_male = itemgetter(*indices_male)(c_count[w])
#
#     y_count_female = itemgetter(*indices_female)(y[w])
#     y_count_male = itemgetter(*indices_male)(y[w])
#
#     c_gender_female = itemgetter(*indices_female)(c_gender[w])
#     c_gender_male = itemgetter(*indices_male)(c_gender[w])
#
#     # print(type(c_plen))
#     # print(len(c_plen))
#     # print(type(c_gender))
#     # print(len(c_gender[w]))
#
#     c_plen_female = itemgetter(*indices_female)(c_plen)
#     c_plen_male = itemgetter(*indices_male)(c_plen)
#
#     # trainset
#     #c_count train
#     c_count_train_female = np.array(c_count_female[:len_train_female])
#     c_count_train_male = np.array(c_count_male[:len_train_male])
#
#     c_count_train = np.concatenate([c_count_train_female, c_count_train_male])
#     c_count_train = np.reshape(c_count_train, (len(c_count_train), 1))
#     c_count_train_set.append(c_count_train)
#
#     #y_count train
#     y_count_train_female = np.array(y_count_female[:len_train_female])
#     y_count_train_male = np.array(y_count_male[:len_train_male])
#
#     y_count_train = np.concatenate([y_count_train_female, y_count_train_male])
#     y_count_train = np.reshape(y_count_train, (len(y_count_train), 1))
#     y_count_train_set.append(y_count_train)
#
#
#     c_gender_train_female = np.array(c_gender_female[:len_train_female])
#     c_gender_train_male = np.array(c_gender_male[:len_train_male])
#
#     c_gender_train = np.concatenate([c_gender_train_female, c_gender_train_male])
#
#     x_train_count = np.concatenate([c_count_train, y_count_train], axis=1)
#
# # TODO: fit model using equation 3 (on 80% of train data) - check if this is balanced
#
#
# # TODO: filling c_count prime, c_count target, c_plen to predict the c_gender
#
# # TODO:
#
#
#
#
#
#
#
# # x_test = np.concatenate([c_count_test, y_count_test], axis=1)
# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(2, 6), random_state=1, activation='relu')
