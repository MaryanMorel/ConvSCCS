#~/bin/sh
N_JOBS=20
DB_NAME="multiple_missing_features_experiments.db"
python utils/database_config $DB_NAME
python utils/run_experiment.py experiments/multiple_missing_features_experiment.py $N_JOBS 1 2 $DB_NAME
python utils/run_experiment.py experiments/multiple_missing_features_experiment.py $N_JOBS 2 2 $DB_NAME
python utils/run_experiment.py experiments/multiple_missing_features_experiment.py $N_JOBS 2 4 $DB_NAME
python utils/run_experiment.py experiments/multiple_missing_features_experiment.py $N_JOBS 2 6 $DB_NAME
python utils/run_experiment.py experiments/multiple_missing_features_experiment.py $N_JOBS 2 8 $DB_NAME
python utils/run_experiment.py experiments/multiple_missing_features_experiment.py $N_JOBS 2 10 $DB_NAME
python utils/run_experiment.py experiments/multiple_missing_features_experiment.py $N_JOBS 2 12 $DB_NAME
python utils/run_experiment.py experiments/multiple_missing_features_experiment.py $N_JOBS 2 14 $DB_NAME
