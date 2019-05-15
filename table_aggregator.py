#!/usr/bin/env python
import ast
import configparser
import csv
import os
import sys

import numpy as np
import pandas as pd


if not (len(sys.argv) == 2 and os.path.isfile(sys.argv[1])):
    print('Usage: python "%s" [config file]' % sys.argv[0])
    sys.exit()

config = configparser.ConfigParser()
config.read(sys.argv[1])
n_experiment_repeats = int(config['DEFAULT']['n_experiment_repeats'])
cross_val_n_folds = int(config['DEFAULT']['cross_val_n_folds'])
cross_val_n_repeats = int(config['DEFAULT']['cross_val_n_repeats'])
IPEC_percentiles = ast.literal_eval(config['DEFAULT']['IPEC_percentiles'])
cindex_method = config['DEFAULT']['cindex_method'].strip()
survival_estimator_names = ast.literal_eval(config['DEFAULT']['methods'])
metrics = ast.literal_eval(config['DEFAULT']['metrics'])
datasets = ast.literal_eval(config['DEFAULT']['datasets'])
output_dir = config['DEFAULT']['output_dir']

csv_filenames = []
for survival_estimator_name in survival_estimator_names:
    if survival_estimator_name == 'coxph':
        table_filename = os.path.join(output_dir,
                                      '%s_experiments%d_%s_table.csv'
                                      % (survival_estimator_name,
                                         n_experiment_repeats,
                                         cindex_method))
    else:
        table_filename = os.path.join(output_dir,
                                      '%s_experiments%d_cv(%d,%d)_%s_table.csv'
                                      % (survival_estimator_name,
                                         n_experiment_repeats,
                                         cross_val_n_folds,
                                         cross_val_n_repeats,
                                         cindex_method))
    assert os.path.isfile(table_filename)
    csv_filenames.append(table_filename)

results = {}
methods = [row.strip() for row
           in open('survival_estimator_names.txt', 'r').readlines()]
method_name_fixes = {'kernel cosine': 'kernel (box) cosine',
                     'kernel l1': 'kernel (box) l1',
                     'kernel l2': 'kernel (box) l2'}
methods_seen = set()

for filename in csv_filenames:
    with open(filename, 'r') as f:
        csv_file = csv.reader(f)
        header = True
        for row in csv_file:
            if header:
                assert row[0] == 'dataset'
                header = False
                continue

            dataset = row[0]
            experiment_idx = int(row[1])
            method = row[2]
            if method in method_name_fixes:
                method = method_name_fixes[method]
            methods_seen.add(method)

            metrics = []
            try:
                cindex = float(row[3])
            except:
                cindex = np.nan
            metrics.append(cindex)

            for raw_IPEC_score in row[4:]:
                try:
                    IPEC_score = float(raw_IPEC_score)
                except:
                    IPEC_score = np.nan
                metrics.append(IPEC_score)

            key = (dataset, method)
            if key not in results:
                results[key] = [metrics]
            else:
                results[key].append(metrics)

for key in results:
    results[key] = np.array(results[key])

columns = ['cindex'] + ['IPEC (%.2f)' % q for q in IPEC_percentiles]
for dataset in datasets:
    dataset_table = []
    for method in methods:
        if method in methods_seen:
            key = (dataset, method)
            dataset_table.append(np.nanmean(results[key], axis=0))

    methods_capitalize_l1_l2 = []
    for method in methods:
        if method.endswith(' l1'):
            methods_capitalize_l1_l2.append(method[:-3] + ' L1')
        elif method.endswith(' l2'):
            methods_capitalize_l1_l2.append(method[:-3] + ' L2')
        else:
            methods_capitalize_l1_l2.append(method)

    df = pd.DataFrame(dataset_table, methods_capitalize_l1_l2, columns)
    print('[%s]' % dataset)
    print(df)
    print()
