#!/usr/bin/env python
import ast
import configparser
import csv
import os
os.environ['QT_QPA_PLATFORM']='offscreen'
import sys

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.preprocessing import StandardScaler

from npsurvival_models import BasicSurvival
from survival_datasets import load_dataset
from util import compute_median_survival_time, compute_IPEC_scores


if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit()

survival_estimator_name = 'coxph'  # for this file
config = configparser.ConfigParser()
config.read(sys.argv[1])
n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
IPEC_percentiles = ast.literal_eval(config['DEFAULT']['IPEC_percentiles'])
cindex_method = config['DEFAULT']['cindex_method'].strip()
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']
os.makedirs(output_dir, exist_ok=True)

output_table_filename = os.path.join(output_dir,
                                     '%s_experiments%d_%s_table.csv'
                                     % (survival_estimator_name,
                                        n_experiment_repeats,
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

        num_IPEC_horizons = len(IPEC_percentiles)
        sorted_train_times = np.sort(y_train[:, 0])
        num_train_times = len(sorted_train_times)
        IPEC_horizons = [sorted_train_times[int(q * num_train_times)]
                         for q in IPEC_percentiles[:-1]]
        IPEC_horizons.append(sorted_train_times[-1])

        print('Testing...')
        scaler = StandardScaler()
        X_train_standardized = scaler.fit_transform(X_train)
        X_test_standardized = scaler.transform(X_test)

        sort_indices = np.argsort(y_train[:, 0])

        train_data_df = \
            pd.DataFrame(np.hstack((X_train_standardized, y_train)),
                         columns=feature_names + ['time', 'status'])

        surv_model = CoxPHFitter()
        surv_model.fit(train_data_df, duration_col='time', event_col='status',
                       show_progress=False, step_size=.1)

        sorted_y_test = np.sort(np.unique(y_test[:, 0]))
        if sorted_y_test[0] != 0:
            mesh_points = np.concatenate(([0.], sorted_y_test))
        else:
            mesh_points = sorted_y_test
        surv = \
            surv_model.predict_survival_function(X_test_standardized,
                                                 mesh_points)
        surv = surv.values.T

        # ---------------------------------------------------------------------
        # compute c-index
        #
        if cindex_method == 'cum_haz':
            cum_haz = \
                surv_model.predict_cumulative_hazard(X_test_standardized,
                                                     sorted_y_test)
            cum_haz = cum_haz.values.T
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
                surv_model.predict_cumulative_hazard(X_test_standardized,
                                                     sorted_y_test)
            cum_haz = cum_haz.values.T
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

        # ---------------------------------------------------------------------
        # compute IPEC score using a few horizon times
        #
        test_IPEC_scores = compute_IPEC_scores(y_train, y_test,
                                               mesh_points, surv,
                                               IPEC_horizons)
        print('c-index: %5.4f' % test_cindex
              + ', '
              + ', '.join(['IPEC (%.2f): %5.4f'
                          % (q, test_IPEC_scores[horizon] / horizon)
                          for q, horizon in
                          zip(IPEC_percentiles,
                              IPEC_horizons)]))

        csv_writer.writerow([dataset, experiment_idx, 'cox', test_cindex] +
                            [test_IPEC_scores[horizon] / horizon
                             for horizon in IPEC_horizons])

        print()
        print()
