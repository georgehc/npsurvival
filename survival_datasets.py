#!/usr/bin/env python
import csv
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


def load_dataset(dataset, random_seed_offset=0):
    """
    Loads a survival analysis dataset (supported public datasets: pbc, gbsg2,
    recid).

    Parameters
    ----------
    dataset : string
        One of 'pbc', 'gbsg2', 'recid'.

    Returns
    -------
    X_train : 2D numpy array, shape = [n_samples, n_features]
        Training feature vectors.

    y_train : 2D numpy array, shape = [n_samples, 2]
        Survival labels (first column is for observed times, second column
        is for event indicators) for training data. The i-th row corresponds to
        the i-th row in `X_train`.

    X_test : 2D numpy array
        Test feature vectors. Same features as for training.

    y_test : 2D numpy array
        Test survival labels.

    feature_names : list
        List of strings specifying the names of the features (columns of
        `X_train` and `X_test`).
    """
    if dataset == 'pbc':
        data = sm.datasets.get_rdataset('pbc', 'survival').data
        data_no_nans = data.dropna()
        del data_no_nans['id']
        y = data_no_nans[['time', 'status']].values
        y[:, 1] = y[:, 1] == 2
        y = y.astype(np.float)
        del data_no_nans['time']
        del data_no_nans['status']
        X = data_no_nans.values
        X[:, 2] = (X[:, 2] == 'f')
        X = X.astype(np.float)
        feature_names = list(data_no_nans.columns)
        dataset_random_seed = 518952347

    elif dataset == 'gbsg2':
        if not os.path.isfile('data/gbsg2_X.txt') \
                or not os.path.isfile('data/gbsg2_y.txt') \
                or not os.path.isfile('data/gbsg2_feature_names.txt'):
            X = []
            y = []
            with open('data/gbsg2.csv', 'r') as f:
                header = True
                for row in csv.reader(f):
                    if header:
                        feature_names = row[1:-2]
                        header = False
                    elif row:
                        horTh = 1*(row[1].lower().strip() == 'yes')
                        age = float(row[2])
                        menostat = 1*(row[3].lower().strip() == 'post')
                        tsize = float(row[4])
                        tgrade = row[5].lower().strip()
                        if tgrade == 'i':
                            tgrade = 0.
                        elif tgrade == 'ii':
                            tgrade = 1.
                        elif tgrade == 'iii':
                            tgrade = 2.
                        else:
                            raise ValueError('Invalid "tgrade" value: %s'
                                             % tgrade)
                        pnodes = float(row[6])
                        progrec = float(row[7])
                        estrec = float(row[8])
                        time = float(row[9])
                        cens = float(row[10])
                        X.append((horTh, age, menostat, tsize, tgrade, pnodes,
                                  progrec, estrec))
                        y.append((time, cens))
            X = np.array(X, dtype=np.float)
            y = np.array(y, dtype=np.float)

            with open('data/gbsg2_feature_names.txt', 'w') as f:
                f.write("\n".join(feature_names))
            np.savetxt('data/gbsg2_X.txt', X)
            np.savetxt('data/gbsg2_y.txt', y)

        X = np.loadtxt('data/gbsg2_X.txt')
        y = np.loadtxt('data/gbsg2_y.txt')
        feature_names = [line.strip() for line
                         in open('data/gbsg2_feature_names.txt').readlines()]

        dataset_random_seed = 2090429699

    elif dataset == 'recid':
        if not os.path.isfile('data/recid_X.txt') \
                or not os.path.isfile('data/recid_y.txt') \
                or not os.path.isfile('data/recid_feature_names.txt'):
            X = []
            y = []
            with open('data/recid.csv', 'r') as f:
                header = True
                for row in csv.reader(f):
                    if header:
                        feature_names = row[1:-4]
                        header = False
                    elif len(row) == 19:
                        black = float(row[1])
                        alcohol = float(row[2])
                        drugs = float(row[3])
                        super_ = float(row[4])
                        married = float(row[5])
                        felon = float(row[6])
                        workprg = float(row[7])
                        property_ = float(row[8])
                        person = float(row[9])
                        priors = float(row[10])
                        educ = float(row[11])
                        rules = float(row[12])
                        age = float(row[13])
                        tserved = float(row[14])
                        time = float(row[16])
                        cens = 1. - float(row[17])
                        X.append((black, alcohol, drugs, super_, married,
                                  felon, workprg, property_, person, priors,
                                  educ, rules, age, tserved))
                        y.append((time, cens))
            X = np.array(X, dtype=np.float)
            y = np.array(y, dtype=np.float)

            with open('data/recid_feature_names.txt', 'w') as f:
                f.write("\n".join(feature_names))
            np.savetxt('data/recid_X.txt', X)
            np.savetxt('data/recid_y.txt', y)

        X = np.loadtxt('data/recid_X.txt')
        y = np.loadtxt('data/recid_y.txt')
        feature_names = [line.strip() for line
                         in open('data/recid_feature_names.txt').readlines()]

        dataset_random_seed = 3959156915

    else:
        raise NotImplementedError('Unsupported dataset: %s' % dataset)

    rng = np.random.RandomState(dataset_random_seed + random_seed_offset)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=rng)

    return X_train, y_train, X_test, y_test, feature_names
