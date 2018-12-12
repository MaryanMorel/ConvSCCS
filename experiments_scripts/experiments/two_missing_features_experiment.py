# coding: utf-8

import sys
import numpy as np


if __name__ == "__main__":
    try:
        experiment_id = 0
        seed = int(sys.argv[1])
        features_set = "set 2"
        magnitude = float(sys.argv[2])
        experiment_desc = "two_missing_features_%f_magnitude" % magnitude
        version = magnitude
        print(
            "running experiment {expid} with seed {seed}".format(
                expid=experiment_id, seed=seed
            )
        )
        db_name = sys.argv[3]
    except IndexError:
        print(
            """Invalid usage, you should use the following syntax:

        python missing_data_experiment.py seed missing_feature_magnitude db_name.db

        where both seed is an integer and missing_feature_magnitude is a float.
        In this experiment, missing_feature_magnitude controls the missing
        features RI order of magnitude.
        
        db_name should be the path/name to the database created with 
        utils/databaseconfig.py"""
        )

    # Simulation parameters
    n_features = 16
    lags = 49
    n_cases = 2000
    n_intervals = 750
    effects_str = """null_effect = [ce.constant_effect(1)] * 7
    constant_effect = ce.constant_effect(1.5)
    early_effect = ce.bell_shaped_effect(2, 20)
    intermediate_effect = ce.bell_shaped_effect(2, 30, 15, 15)
    late_effects = [ce.increasing_effect(2, curvature_type=1)[::-1],
                    ce.increasing_effect(2, curvature_type=1),
                    ce.increasing_effect(2, curvature_type=2),
                    ce.increasing_effect(2, curvature_type=4)]

    missing_effect_1 = ce.constant_effect(%f)
    missing_effect_2 = ce.bell_shaped_effect(%f, 30, 15, 15)
    sim_effects = [*null_effect, constant_effect, early_effect,
    intermediate_effect, *late_effects, missing_effect_1, 
    missing_effect_2]""" % (
        magnitude,
        magnitude,
    )
    hidden_features = [14, 15]
    time_drift = None
    time_drift_str = "time_drift = lambda t: np.log(8 * np.sin(.01 * t) + 9)"
    n_corr = n_features

    import json
    from pickle import dumps
    from datetime import datetime
    from time import time

    # Database
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from utils.database_config import (
        Base,
        Experiment,
        ExperimentResult,
        Simulation,
        ConvSCCSModel,
    )

    # Utility functions
    from utils.preprocessing import to_nonparasccs
    from utils.metrics import squared_error, absolute_error,\
        absolute_percentage_error

    # Simulation and model
    from tick.survival import SimuSCCS, ConvSCCS
    from tick.survival.simu_sccs import CustomEffects

    # init DB connection
    engine = create_engine("sqlite:///" + db_name)
    Base.metadata.bind = engine
    DBsession = sessionmaker(bind=engine)
    session = DBsession()

    # --- Simulate data
    # Setup sim
    n_cols = lags + 1
    ce = CustomEffects(n_cols)
    null_effect = [ce.constant_effect(1)] * 7
    constant_effect = ce.constant_effect(1.5)
    early_effect = ce.bell_shaped_effect(2, 20)
    intermediate_effect = ce.bell_shaped_effect(2, 30, 15, 15)
    late_effects = [
        ce.increasing_effect(2, curvature_type=1)[::-1],
        ce.increasing_effect(2, curvature_type=1),
        ce.increasing_effect(2, curvature_type=2),
        ce.increasing_effect(2, curvature_type=4),
    ]

    missing_effect_1 = ce.constant_effect(magnitude)
    missing_effect_2 = ce.bell_shaped_effect(magnitude, 30, 15, 15)
    sim_effects = [
        *null_effect,
        constant_effect,
        early_effect,
        intermediate_effect,
        *late_effects,
        missing_effect_1,
        missing_effect_2,
    ]
    n_features = len(sim_effects)
    coeffs = [np.log(s) for s in sim_effects]
    simu_n_lags = np.repeat(49, n_features).astype("uint64")

    n_missing_features = 2
    hidden_features = [n_features - (i + 1) for i in range(n_missing_features)]
    sim = SimuSCCS(
        int(n_cases),
        n_intervals,
        n_features,
        simu_n_lags,
        time_drift=time_drift,
        n_correlations=n_features,
        coeffs=coeffs,
        seed=seed,
        verbose=False,
        hidden_features=hidden_features,
    )

    features, censored_features, labels, censoring, coeffs = sim.simulate()
    [coeffs.pop(n_features - i - 1) for i in range(n_missing_features)]
    n_features = n_features - n_missing_features
    n_lags = np.repeat(49, n_features).astype("uint64")

    adjacency_matrix = sim.hawkes_exp_kernels.adjacency.tobytes()

    # Convert to DataFrame format
    df = to_nonparasccs(censored_features, labels, censoring, lags)
    df["indiv"] = df.index
    df = df.astype("int64")

    exposures_frequencies = df.drugid.value_counts()

    exp_log = Experiment(
        experiment_id=experiment_id,
        version=version,
        description=experiment_desc,
        features_set=features_set,
        effects=effects_str,
        time_drift=time_drift_str,
        n_features=n_features,
        n_intervals=n_intervals,
        n_cases=n_cases,
        sim_n_lags=n_lags,
        sim_n_corr=n_corr,
        sim_coeffs_obj=dumps(coeffs),
    )

    session.merge(exp_log)

    sim_log = Simulation(
        experiment_id=experiment_id,
        version=version,
        seed=seed,
        sim_adjacency_matrix=dumps(adjacency_matrix),
        features_frequency=dumps(exposures_frequencies),
    )

    session.merge(sim_log)
    session.commit()

    start = time()
    lrn = ConvSCCS(
        n_lags=n_lags, penalized_features=np.arange(n_features), verbose=False
    )
    C_tv_range = (1, 5)
    C_group_l1_range = (1, 5)
    fitted_coeffs, cv_track = lrn.fit_kfold_cv(
        censored_features,
        labels,
        censoring,
        C_tv_range=C_tv_range,
        C_group_l1_range=C_group_l1_range,
        confidence_intervals=False,
    )  # WARNING: no bootstrap in this simulation
    elapsed_time = time() - start

    model_id = "ConvSCCS"
    model_log = ConvSCCSModel(
        experiment_id=experiment_id,
        version=version,
        seed=seed,
        model_id=model_id,
        run_time=elapsed_time,
        model_params=str(cv_track.model_params),
        cv_track=dumps(cv_track),
    )
    session.add(model_log)

    # Send results to DB
    se_age = -1
    ae_age = -1
    ape_age = -1
    for drug_id in range(n_features):
        # True coefficients
        c = np.exp(coeffs[drug_id])
        # ConvSCCS estimate
        d_fit = np.exp(fitted_coeffs[drug_id])

        se_features = squared_error(c, d_fit)
        ae_features = absolute_error(c, d_fit)
        ape_features = absolute_percentage_error(c, d_fit)
        result = ExperimentResult(
            experiment_id=experiment_id,
            version=version,
            seed=seed,
            model_id=model_id,
            drug_id=drug_id,
            insert_date=datetime.now(),
            se_features=se_features,
            se_age=se_age,
            ae_features=ae_features,
            ae_age=ae_age,
            ape_features=ape_features,
            ape_age=ape_age,
        )
        session.add(result)

    session.commit()

    # --- Close DB connexion
    session.close()
