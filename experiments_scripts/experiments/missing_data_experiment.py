# coding: utf-8

import sys
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def remove_col(idx, col, row, data):
    feat_idx = np.where(col == idx)[0]
    if len(feat_idx) > 0:
        col = np.delete(col, feat_idx)
        row = np.delete(row, feat_idx)
        data = np.delete(data, feat_idx)
    return col, row, data


def compute_exposures(feat_mat, shape):
    blub = feat_mat.T.tolil()
    blab = [(i, np.min(r)) for i, r in enumerate(blub.rows) if len(r) > 0]
    rows = [b[1] for b in blab]
    cols = [b[0] for b in blab]
    data = np.ones_like(rows).astype("float64")
    exposures = csr_matrix((data, (rows, cols)), shape=shape)
    return exposures


def create_censored_exposures(input_feat, hospitalisation_feature_idx):
    coo_feat = input_feat.tocoo()
    final_shape = (coo_feat.shape[0], coo_feat.shape[1] - 1)

    # Filter hospitalizations
    candidate_hospitalizations = np.where(coo_feat.col ==
                                          hospitalisation_feature_idx)[0]
    hospitalizations = candidate_hospitalizations[
        np.random.random(candidate_hospitalizations.shape) > (1 - p)
    ]
    event_deletion_idx = np.array(
        list(set(candidate_hospitalizations).difference(set(hospitalizations)))
    ).astype("int")
    col = np.delete(coo_feat.col, event_deletion_idx)
    row = np.delete(coo_feat.row, event_deletion_idx)
    data = np.delete(coo_feat.data, event_deletion_idx)

    filtered_coo_feat = coo_matrix((data, (row, col)), shape=coo_feat.shape)

    censoring_starts = np.where(coo_feat.col == hospitalisation_feature_idx)[0]
    if len(censoring_starts) > 0:
        # Censor other features according to hospitalizations
        censored_values = []
        for i in censoring_starts:
            censored_values.append(
                np.where(
                    np.logical_and(np.greater_equal(row, i),
                    np.less(row, (i + D)))
                )[0]
            )
        censored_values = np.hstack(censored_values)

        # Mask (delete) hospitalization feature
        u_col, u_row, u_data = remove_col(hospitalisation_feature_idx, col,
                                          row, data)
        uncensored_events = csr_matrix((u_data, (u_row, u_col)),
                                       shape=final_shape)

        # apply censoring
        c_col = np.delete(col, censored_values)
        c_row = np.delete(row, censored_values)
        c_data = np.delete(data, censored_values)

        c_col, c_row, c_data = remove_col(
            hospitalisation_feature_idx, c_col, c_row, c_data
        )

        censored_events = csr_matrix((c_data, (c_row, c_col)),
                                     shape=final_shape)

        uncensored_exposures = compute_exposures(uncensored_events, 
                                                 final_shape)
        censored_exposures = compute_exposures(censored_events, final_shape)
    else:
        # Mask (delete) hospitalization feature
        u_col, u_row, u_data = remove_col(hospitalisation_feature_idx, col,
                                          row, data)
        uncensored_events = csr_matrix((u_data, (u_row, u_col)),
                                       shape=final_shape)
        uncensored_exposures = compute_exposures(uncensored_events,
                                                 final_shape)
        censored_exposures = compute_exposures(uncensored_events, final_shape)

    return uncensored_exposures, censored_exposures


if __name__ == "__main__":
    try:
        experiment_id = 1
        seed = int(sys.argv[1])
        p = 1.0
        D = int(sys.argv[2])
        features_set = "set 2"
        noise_level = 0
        experiment_desc = "missing_feature_p_%f_noise_level_%d" % (p, D)
        version = 4.0
        print(
            "running experiment {expid} with seed {seed}".format(
                expid=experiment_id, seed=seed
            )
        )
        db_name = sys.argv[3]
    except IndexError:
        print(
            """Invalid usage, you should use the following syntax:
        
        python missing_data_experiment.py seed noise_level db_name.db
        
        where both seed and noise level are integers. In this experiment,
        noise_level indicates the length of missing data periods (in days).
        
        db_name should be the path/name to the database created with 
        utils/databaseconfig.py"""
        )

    # Simulation parameters
    n_features = 14
    hospitalisation_feature_idx = n_features
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
    sim_effects = [*null_effect, constant_effect, early_effect, 
    intermediate_effect, *late_effects]"""
    time_drift = None
    time_drift_str = "None"
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
    from tick.survival import ConvSCCS
    from utils import SimuSCCS
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
    sim_effects = [
        *null_effect,
        constant_effect,
        early_effect,
        intermediate_effect,
        *late_effects,
    ]
    n_lags = np.repeat(0, n_features + 1).astype("uint64")
    sim = SimuSCCS(
        n_cases,
        n_intervals,
        n_features + 1,
        n_lags,
        exposure_type="multiple_exposures",
        n_correlations=n_corr,
    )

    X, Xc, y, c, coeffs = sim.simulate()

    adjacency_matrix = sim.hawkes_exp_kernels.adjacency.tobytes()

    new_X = [create_censored_exposures(x, hospitalisation_feature_idx) for x
             in X]

    uncensored_features = [x[0] for x in new_X]
    censored_features = [x[1] for x in new_X]
    coeffs = [np.log(s) for s in sim_effects]
    n_lags = np.repeat(49, n_features).astype("uint64")

    sim = SimuSCCS(
        n_cases,
        n_intervals,
        n_features,
        n_lags,
        exposure_type="single_exposure",
        n_correlations=n_corr,
        coeffs=coeffs,
    )

    features = [x[0] for x in new_X]
    censored_features = [x[1] for x in new_X]

    labels = sim.simulate_outcomes(uncensored_features)
    censoring = np.repeat(n_intervals, n_cases).astype("int")

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
