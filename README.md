# Code for ICML 2019 paper "Nearest Neighbor and Kernel Survival Analysis: Nonasymptotic Error Bounds and Strong Consistency Rates"

Author: George H. Chen (georgechen [at symbol] cmu.edu)

Code requirements:

- Anaconda Python 3.6
- Additional packages: joblib, lifelines
- cython compilation is required:

```
python setup_random_survival_forest_cython.py build_ext --inplace
```

The main code implementing all the different nonparametric survival methods from the paper is in `npsurvival_models.py`. Cython helper code for random survival forests is in `random_survival_forest_cython.pyx`. There are two main utility files: `survival_datasets.py` deals with loading datasets (the "pbc" dataset is loaded from the statsmodels Python package; the "gbsg2" and "recid" datasets are loaded from the "data/"), and `util.py` has some helper calculation functions. Note: the "kidney" dataset is not public so I have removed it from this distribution. These Python files just mentioned should not be directly run. Instead the files that should be run are the `demo_*.py` files (e.g., `python demo_rsfann.py config_cum_haz.ini`); in particular, to generate all the experimental results for the "pbc", "gbsg2" and "recid" datasets (and save their results to csv files in the directory `output`), run `./demo.sh` (warning: this takes a while to run).

After running `demo.sh`, a simple way to display all the tabulated outputs is to run `python table_aggregator.py config_cum_haz.ini`. To produce the plots (excluding the "kidney" dataset) in the main part of the paper (i.e., not the extended results), run `python table_aggregator_plot_short.py config_cum_haz.ini`. To produce the plots in the appendix (the extended results, excluding the "kidney" dataset), run `python table_aggregator_plot.py config_cum_haz.ini`. Note that these display/plot scripts require the auxiliary text files `survival_estimator_names.txt` and `survival_estimator_names_short.txt`.

*Important*: If you do not want to re-run all the methods but still want to produce plots (excluding for the "kidney" dataset), I have included precomputed csv tables in the folder `precomputed`. Please move the csv files in this folder to be in the output directory (as specified in the configuration file used; by default if using the provided `config_cum_haz.ini` file, the output directory is `output`) and run the plotting code to regenenerate plots.
