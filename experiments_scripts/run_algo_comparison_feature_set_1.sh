#~/bin/sh
N_JOBS=20
DB_NAME="algo_comp_set_1.db"
python utils/database_config $DB_NAME
python utils/run_experiment.py experiments/algo_comparison_feature_set_1.py $N_JOBS $DB_NAME
