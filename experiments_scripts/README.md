# Experiments

Experiments were run using these scripts. To be able to use them, you must install the following requirements:

    - Python 3.6
    - rpy2
    - ['SCCS' R library](https://cran.r-project.org/web/packages/SCCS/index.html)
    - scipy
    - numpy
    - pandas
    - Tick 0.5
    - sqlalchemy

and compile extensions-functions.c (see readme in the header of the .c file for instructions). This folder should also be added to your `PYTHONPATH` as some utility classes and functions need to be imported from the `utils` module.

Running the `.sh` scripts will create a database for each experiment, and then run the corresponding experiments. Experiment results are pushed to the databases each time a loop is completed. The schema of the databases is explicited in `utils/database_config.py`.

Boxplot presented in the papers are simply made by fetching MAE scores for each model / experiment from the databases, and plotting them using [seaborn](https://seaborn.pydata.org/).

`run_algo_comparison_feature_set_1.sh` corresponds to fig. 1 of the main paper
`run_algo_comparison_feature_set_2.sh` corresponds to fig. 2 of the main paper
fig. 3 was produced using "manual" runs of each model, as running them using python multiprocessing resulted in an unexplained overhead for R models.

`run_noisy_timestamps_experiment.sh` corresponds to fig. 4 of the supplementary material
`run_missing_data_experiment.sh` corresponds to fig. 5 of the supplementary material
`run_two_missing_features_experiment.sh` corresponds to fig. 6 of the supplementary material
`run_multiple_missing_features_experiment.sh` corresponds to fig. 7 of the supplementary material