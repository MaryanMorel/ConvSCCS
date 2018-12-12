#~/bin/sh
N_JOBS=20
DB_NAME="noisy_timestamps_experiments.db"
python utils/database_config $DB_NAME
python utils/run_experiment.py experiments/noisy_timestamps_experiment.py $N_JOBS 0 $DB_NAME
python utils/run_experiment.py experiments/noisy_timestamps_experiment.py $N_JOBS 10 $DB_NAME
python utils/run_experiment.py experiments/noisy_timestamps_experiment.py $N_JOBS 20 $DB_NAME
python utils/run_experiment.py experiments/noisy_timestamps_experiment.py $N_JOBS 40 $DB_NAME
python utils/run_experiment.py experiments/noisy_timestamps_experiment.py $N_JOBS 80 $DB_NAME
python utils/run_experiment.py experiments/noisy_timestamps_experiment.py $N_JOBS 160 $DB_NAME
