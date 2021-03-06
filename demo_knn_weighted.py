#!/usr/bin/env python
import ast
import configparser
import csv
import itertools
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
import sys

import numpy as np
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold

from npsurvival_models import KNNWeightedSurvival
from survival_datasets import load_dataset
from util import compute_median_survival_time, compute_IPEC_scores, \
    kernel_box, kernel_triangle, kernel_epanechnikov, kernel_trunc_gauss1, \
    kernel_trunc_gauss2, kernel_trunc_gauss3


if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit()

survival_estimator_name = 'knn_weighted'  # for this file
config = configparser.ConfigParser()
config.read(sys.argv[1])
n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
cross_val_n_repeats = int(config['DEFAULT']['cross_val_n_repeats'])
IPEC_percentiles = ast.literal_eval(config['DEFAULT']['IPEC_percentiles'])
cindex_method = config['DEFAULT']['cindex_method'].strip()
metrics = ast.literal_eval(config['DEFAULT']['metrics'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
train_random_seeds = \
    {dataset: int(config['dataset: %s' % dataset]['train_random_seed_%s'
                                                  % survival_estimator_name])
     for dataset in datasets}
os.makedirs(output_dir, exist_ok=True)

kernel_desc_pairs = \
    [(kernel_box, 'k-NN'),
     (kernel_triangle, 'k-NN (triangle)'),
     (kernel_epanechnikov, 'k-NN (Epanechnikov)'),
     (kernel_trunc_gauss1, 'k-NN (truncated Gauss sigma=1)'),
     (kernel_trunc_gauss2, 'k-NN (truncated Gauss sigma=2)'),
     (kernel_trunc_gauss3, 'k-NN (truncated Gauss sigma=3)')]

output_table_filename = os.path.join(output_dir,
                                     '%s_experiments%d_cv(%d,%d)_%s_table.csv'
                                     % (survival_estimator_name,
                                        n_experiment_repeats,
                                        cross_val_n_folds,
                                        cross_val_n_repeats,
                                        cindex_method))
if os.path.isfile(output_table_filename):
    print('*** Filename already exists: %s' % output_table_filename)
    print('*** Skipping')
    sys.exit()
output_table_file = open(output_table_filename, 'w')
csv_writer = csv.writer(output_table_file)
csv_writer.writerow(['dataset', 'experiment_idx', 'method', 'cindex'] +
                    ['IPEC (%.2f)' % q for q in IPEC_percentiles])

for experiment_idx in range(n_experiment_repeats):
    for dataset in datasets:
        print('[Dataset: %s, experiment: %d]' % (dataset, experiment_idx))
        print()

        X_train, y_train, X_test, y_test, feature_names = \
            load_dataset(dataset, experiment_idx)
        rng = np.random.RandomState(train_random_seeds[dataset]
                                    + experiment_idx)

        # this next count is approximate; the number of training examples per
        # fold may differ slightly due to the total number of training examples
        # not being divisible by `cross_val_n_folds`
        n_train_per_fold = int(np.floor((cross_val_n_folds - 1) * len(X_train)
                                        / cross_val_n_folds))

        hyperparams = [2**pow_ for pow_
                       in range(2, 1 + int(np.ceil(np.log2(len(X_train)))))]

        num_IPEC_horizons = len(IPEC_percentiles)
        sorted_train_times = np.sort(y_train[:, 0])
        num_train_times = len(sorted_train_times)
        IPEC_horizons = [sorted_train_times[int(q * num_train_times)]
                         for q in IPEC_percentiles[:-1]]
        IPEC_horizons.append(sorted_train_times[-1])

        for (kernel_func, desc), metric \
                in itertools.product(kernel_desc_pairs, metrics):
            print('Training (%s, %s)...' % (desc, metric))
            kf = RepeatedKFold(n_splits=cross_val_n_folds,
                               n_repeats=cross_val_n_repeats,
                               random_state=rng)

            max_cindex = -np.inf
            arg_max_cindex = None

            min_IPEC_scores = [np.inf for idx in range(num_IPEC_horizons)]
            arg_min_IPEC_scores = [None for idx in range(num_IPEC_horizons)]
            early_stop_due_to_k_being_too_large = False

            for k in hyperparams:
                if k > n_train_per_fold:
                    k = n_train_per_fold
                    early_stop_due_to_k_being_too_large = True

                cindex_scores = []
                IPEC_scores = [[] for idx in range(num_IPEC_horizons)]

                for train_idx, val_idx in kf.split(X_train):
                    fold_X_train = X_train[train_idx]
                    fold_y_train = y_train[train_idx]
                    fold_X_val = X_train[val_idx]
                    fold_y_val = y_train[val_idx]

                    scaler = StandardScaler()
                    fold_X_train_standardized = \
                        scaler.fit_transform(fold_X_train)
                    fold_X_val_standardized = scaler.transform(fold_X_val)

                    surv_model = KNNWeightedSurvival(n_neighbors=k, n_jobs=-1,
                                                     metric=metric)
                    surv_model.fit(fold_X_train_standardized, fold_y_train)

                    sorted_fold_y_val = np.sort(np.unique(fold_y_val[:, 0]))
                    if sorted_fold_y_val[0] != 0:
                        mesh_points = np.concatenate(([0.], sorted_fold_y_val))
                    else:
                        mesh_points = sorted_fold_y_val
                    surv = \
                        surv_model.predict_surv(fold_X_val_standardized,
                                                mesh_points,
                                                presorted_times=True,
                                                kernel_function=kernel_func)

                    # ---------------------------------------------------------
                    # compute c-index
                    #
                    if cindex_method == 'cum_haz':
                        cum_haz = \
                            surv_model.predict_cum_haz(
                                fold_X_val_standardized,
                                sorted_fold_y_val,
                                presorted_times=True,
                                kernel_function=kernel_func)
                        cum_hazard_scores = cum_haz.sum(axis=1)
                        cindex = concordance_index(fold_y_val[:, 0],
                                                   -cum_hazard_scores,
                                                   fold_y_val[:, 1])
                    elif cindex_method == 'cum_haz_from_surv':
                        surv_thresholded = np.maximum(surv,
                                                      np.finfo(float).eps)
                        cum_haz = -np.log(surv_thresholded)
                        cum_hazard_scores = cum_haz.sum(axis=1)
                        cindex = concordance_index(fold_y_val[:, 0],
                                                   -cum_hazard_scores,
                                                   fold_y_val[:, 1])
                    elif cindex_method == 'median':
                        predicted_medians = \
                            np.array([compute_median_survival_time(mesh_points,
                                                                   surv_row)
                                      for surv_row in surv])
                        cindex = concordance_index(fold_y_val[:, 0],
                                                   predicted_medians,
                                                   fold_y_val[:, 1])
                    elif cindex_method == 'median_from_cum_haz':
                        cum_haz = \
                            surv_model.predict_cum_haz(
                                fold_X_val_standardized,
                                sorted_fold_y_val,
                                presorted_times=True,
                                kernel_function=kernel_func)
                        predicted_medians = \
                            np.array([compute_median_survival_time(mesh_points,
                                                                   surv_row)
                                      for surv_row in np.exp(-cum_haz)])
                        cindex = concordance_index(fold_y_val[:, 0],
                                                   predicted_medians,
                                                   fold_y_val[:, 1])
                    else:
                        raise NotImplementedError(
                            'Unsupported c-index method: %s' % cindex_method)
                    cindex_scores.append(cindex)

                    # ---------------------------------------------------------
                    # compute IPEC score using a few horizon times
                    #
                    fold_IPEC_scores = compute_IPEC_scores(fold_y_train,
                                                           fold_y_val,
                                                           mesh_points,
                                                           surv, IPEC_horizons)
                    for idx, IPEC_horizon in enumerate(IPEC_horizons):
                        IPEC_scores[idx].append(fold_IPEC_scores[IPEC_horizon])

                cross_val_cindex = np.mean(cindex_scores)
                cross_val_IPEC_scores = {}
                for idx, q in enumerate(IPEC_percentiles):
                    cross_val_IPEC_scores[q] = np.mean(IPEC_scores[idx])
                # print('k: %d, c-index: %5.4f, ' % (k, cross_val_cindex)
                #       + ', '.join(['IPEC (%.2f): %5.4f'
                #                 % (q, cross_val_IPEC_scores[q] / horizon)
                #                 for q, horizon in
                #                 zip(IPEC_percentiles,
                #                     IPEC_horizons)]))

                if cross_val_cindex > max_cindex:
                    max_cindex = cross_val_cindex
                    arg_max_cindex = k

                for idx, q in enumerate(IPEC_percentiles):
                    if cross_val_IPEC_scores[q] < min_IPEC_scores[idx]:
                        min_IPEC_scores[idx] = cross_val_IPEC_scores[q]
                        arg_min_IPEC_scores[idx] = k

                if early_stop_due_to_k_being_too_large:
                    break

            best_hyperparams = []
            best_hyperparams_set = set()
            hyperparameter_to_score = {}
            final_test_scores = {'cindex': ''}
            for q in IPEC_percentiles:
                final_test_scores[q] = ''

            if arg_max_cindex is not None:
                best_hyperparams.append(arg_max_cindex)
                best_hyperparams_set.add(arg_max_cindex)
                print('Best k for maximizing training c-index: %d '
                      % arg_max_cindex
                      + '-- achieves score %5.4f' % max_cindex)
                hyperparameter_to_score[arg_max_cindex] = ['cindex']

            for idx, (q, horizon) in enumerate(zip(IPEC_percentiles,
                                                   IPEC_horizons)):
                if arg_min_IPEC_scores[idx] is not None:
                    if arg_min_IPEC_scores[idx] not in best_hyperparams_set:
                        best_hyperparams.append(arg_min_IPEC_scores[idx])
                        best_hyperparams_set.add(arg_min_IPEC_scores[idx])
                    print('Best k for minimizing training IPEC (%.2f) ' % q +
                          'score: %d ' % arg_min_IPEC_scores[idx] +
                          '-- achieves score %5.4f'
                          % (min_IPEC_scores[idx] / horizon))
                    if arg_min_IPEC_scores[idx] \
                            not in hyperparameter_to_score:
                        hyperparameter_to_score[arg_min_IPEC_scores[idx]] \
                            = [q]
                    else:
                        hyperparameter_to_score[
                            arg_min_IPEC_scores[idx]].append(q)

            print()
            print('Testing (using the hyperparameters mentioned above)...')
            for k in best_hyperparams:
                scaler = StandardScaler()
                X_train_standardized = scaler.fit_transform(X_train)
                X_test_standardized = scaler.transform(X_test)

                surv_model = KNNWeightedSurvival(n_neighbors=k, n_jobs=-1,
                                                 metric=metric)
                surv_model.fit(X_train_standardized, y_train)

                sorted_y_test = np.sort(np.unique(y_test[:, 0]))
                if sorted_y_test[0] != 0:
                    mesh_points = np.concatenate(([0.], sorted_y_test))
                else:
                    mesh_points = sorted_y_test
                surv = \
                    surv_model.predict_surv(X_test_standardized, mesh_points,
                                            presorted_times=True,
                                            kernel_function=kernel_func)

                # -------------------------------------------------------------
                # compute c-index
                #
                if cindex_method == 'cum_haz':
                    cum_haz = \
                        surv_model.predict_cum_haz(X_test_standardized,
                                                   sorted_y_test,
                                                   presorted_times=True,
                                                   kernel_function=kernel_func)
                    cum_hazard_scores = cum_haz.sum(axis=1)
                    test_cindex = concordance_index(y_test[:, 0],
                                                    -cum_hazard_scores,
                                                    y_test[:, 1])
                elif cindex_method == 'cum_haz_from_surv':
                    surv_thresholded = np.maximum(surv,
                                                  np.finfo(float).eps)
                    cum_haz = -np.log(surv_thresholded)
                    cum_hazard_scores = cum_haz.sum(axis=1)
                    test_cindex = concordance_index(y_test[:, 0],
                                                    -cum_hazard_scores,
                                                    y_test[:, 1])
                elif cindex_method == 'median':
                    predicted_medians = \
                        np.array([compute_median_survival_time(mesh_points,
                                                               surv_row)
                                  for surv_row in surv])
                    test_cindex = concordance_index(y_test[:, 0],
                                                    predicted_medians,
                                                    y_test[:, 1])
                elif cindex_method == 'median_from_cum_haz':
                    cum_haz = \
                        surv_model.predict_cum_haz(X_test_standardized,
                                                   sorted_y_test,
                                                   presorted_times=True,
                                                   kernel_function=kernel_func)
                    predicted_medians = \
                        np.array([compute_median_survival_time(mesh_points,
                                                               surv_row)
                                  for surv_row in np.exp(-cum_haz)])
                    test_cindex = concordance_index(y_test[:, 0],
                                                    predicted_medians,
                                                    y_test[:, 1])
                else:
                    raise NotImplementedError('Unsupported c-index method: %s'
                                              % cindex_method)

                # -------------------------------------------------------------
                # compute IPEC score using a few horizon times
                #
                test_IPEC_scores = compute_IPEC_scores(y_train, y_test,
                                                       mesh_points, surv,
                                                       IPEC_horizons)
                print('k: %d, c-index: %5.4f, ' % (k, test_cindex)
                      + ', '.join(['IPEC (%.2f): %5.4f'
                                  % (q, test_IPEC_scores[horizon] / horizon)
                                  for q, horizon in
                                  zip(IPEC_percentiles,
                                      IPEC_horizons)]))

                if 'cindex' in hyperparameter_to_score[k]:
                    final_test_scores['cindex'] = test_cindex
                for q, horizon in zip(IPEC_percentiles, IPEC_horizons):
                    if q in hyperparameter_to_score[k]:
                        final_test_scores[q] = \
                            test_IPEC_scores[horizon] / horizon

            csv_writer.writerow([dataset, experiment_idx,
                                 '%s %s' % (desc, metric),
                                 final_test_scores['cindex']] +
                                [final_test_scores[q]
                                 for q in IPEC_percentiles])

            print()
        print()
