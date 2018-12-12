#~/bin/sh
N_JOBS=20
DB_NAME="missing_features_experiments.db"
python utils/database_config $DB_NAME
python utils/run_experiment.py experiments/two_missing_features_experiment.py $N_JOBS 1.0 $DB_NAME
python utils/run_experiment.py experiments/two_missing_features_experiment.py $N_JOBS 1.2 $DB_NAME
python utils/run_experiment.py experiments/two_missing_features_experiment.py $N_JOBS 1.5 $DB_NAME
python utils/run_experiment.py experiments/two_missing_features_experiment.py $N_JOBS 2.0 $DB_NAME
python utils/run_experiment.py experiments/two_missing_features_experiment.py $N_JOBS 3.0 $DB_NAME
python utils/run_experiment.py experiments/two_missing_features_experiment.py $N_JOBS 4.0 $DB_NAME
python utils/run_experiment.py experiments/two_missing_features_experiment.py $N_JOBS 5.0 $DB_NAME
