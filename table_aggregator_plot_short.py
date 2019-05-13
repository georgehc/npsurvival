#!/usr/bin/env python
import ast
import configparser
import csv
import os
import sys

import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg')
# matplotlib.use('GTKCairo')
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('seaborn')


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
           in open('survival_estimator_names_short.txt', 'r').readlines()]
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

# matplotlib.rcParams.update({'font.size': 8})
figsize = (4.3, 4)
vert = False

columns = ['cindex'] + ['IPEC (%.2f)' % q for q in IPEC_percentiles]
for dataset in datasets:
    dataset_table = []
    all_cindex_scores = []
    all_IPEC_scores = []
    for method in methods:
        key = (dataset, method)
        dataset_table.append(np.nanmean(results[key], axis=0))

        cindex_scores = results[key][:, 0]  # c-index
        IPEC_scores = results[key][:, 4]  # IPEC (0.75)

        cindex_scores = cindex_scores[~np.isnan(cindex_scores)]
        IPEC_scores = IPEC_scores[~np.isnan(IPEC_scores)]
        all_cindex_scores.append(cindex_scores)
        all_IPEC_scores.append(IPEC_scores)

    plt.figure(figsize=figsize)
    if vert:
        # if dataset == 'pbc':
        #     plt.ylim([0.725, 0.89])
        plt.boxplot(all_cindex_scores, vert=vert)
        plt.xticks(range(1, len(methods) + 1), methods, rotation=90)
        plt.ylabel('c-index')
    else:
        # if dataset == 'pbc':
        #     plt.xlim([0.725, 0.89])
        plt.boxplot(all_cindex_scores[::-1], vert=vert)
        plt.xlabel('c-index')
        plt.yticks(range(1, len(methods) + 1), methods[::-1])
    plt.title('Dataset "%s" Concordance Indices' % dataset)
    plt.tight_layout()
    plt.savefig('fig_%s_cindex_%s.pdf' % (dataset, cindex_method))

    plt.figure(figsize=figsize)
    if vert:
        plt.boxplot(all_IPEC_scores, vert=vert)
        plt.xticks(range(1, len(methods) + 1), methods, rotation=90)
        plt.ylabel('IPEC / time horizon')
    else:
        plt.boxplot(all_IPEC_scores[::-1], vert=vert)
        plt.xlabel('IPEC / time horizon')
        plt.yticks(range(1, len(methods) + 1), methods[::-1])
    plt.title('Dataset "%s" IPEC Scores' % dataset)
    plt.tight_layout()
    plt.savefig('fig_%s_ipec.pdf' % dataset)

    # df = pd.DataFrame(dataset_table, methods, columns) 
    # print('[%s]' % dataset)
    # print(df)
    # print()

plt.show()
