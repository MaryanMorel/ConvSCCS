# coding: utf-8

import sys
import json
from pickle import dumps
from datetime import datetime
from time import time

# Database
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from utils.database_config import *

# Simulations and ConvSCCS model
from utils import *
from tick.survival import ConvSCCS, SimuSCCS

# Model
from tick.inference import LearnerSCCS

# R models
from rpy2 import robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

pandas2ri.activate()
sccs = importr("SCCS")

###############################################################################
experiment_id = 1
experiment_desc = "Custom_%i_feat_%i_samples"
features_set = "set 2"
version = 4.0
###############################################################################

# Simulation parameters
seed = 42
lags = 49
n_cases = 4000
n_intervals = 750
effects_str = """null_effect = [ce.constant_effect(1)] * 7
constant_effect = ce.constant_effect(1.5)
early_effect = ce.bell_shaped_effect(2, 20)
intermediate_effect = ce.bell_shaped_effect(2, 30, 15, 15)
late_effects = [ce.increasing_effect(2, curvature_type=1)[::-1],
                ce.increasing_effect(2, curvature_type=1),
                ce.increasing_effect(2, curvature_type=2),
                ce.increasing_effect(2, curvature_type=4)]

sim_effects = [*null_effect, constant_effect, early_effect, intermediate_effect, *late_effects]"""
time_drift_str = "time_drift = lambda t: np.log(8 * np.sin(.01 * t) + 9)"
n_corr = 24

# Models parameters
agegrp = [125, 250, 375, 500, 625]  # bins for age groups
kn = 12  # Number of spline knots for R SCCS models


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
        print(
            "running experiment {expid} with seed {seed}".format(
                expid=experiment_id, seed=seed
            )
        )
    db_name = sys.argv[2]

    # init DB connection
    engine = create_engine("sqlite:///" + db_name)
    Base.metadata.bind = engine
    DBsession = sessionmaker(bind=engine)
    session = DBsession()
    pandas2ri.activate()

    # --- Simulate data
    # Setup sim
    n_cols = lags + 1
    effect = Effects(lags + 1)
    ce = CustomEffects(lags + 1)
    effects_compiled = compile(effects_str, "<string>", "exec")
    exec(effects_compiled)  # create sim_effects
    td_compiled = compile(time_drift_str, "<string>", "exec")
    exec(td_compiled)  # create time_drift
    n_features = len(sim_effects)
    sim_effects = np.hstack(sim_effects)
    coeffs = np.log(sim_effects)
    normalized_time_drift = np.exp(time_drift(np.arange(750)))
    normalized_time_drift /= normalized_time_drift.sum()

    sim = SimuSCCS(
        int(n_cases),
        n_intervals,
        n_features,
        n_lags,
        time_drift=time_drift,
        n_correlations=n_features,
        coeffs=coeffs,
        seed=seed,
        verbose=False,
    )

    features, censored_features, labels, censoring, coeffs = sim.simulate()

    adjacency_matrix = sim.hawkes_exp_kernels.adjacency.tobytes()

    # Convert to R format
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
        sim_n_lags=lags,
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

    # add age features
    agegrps = [0, 125, 250, 375, 500, 625, 750]
    n_agegrps = len(agegrps) - 1

    feat_agegrp = np.zeros((n_intervals, n_agegrps))
    for i in range(n_agegrps):
        feat_agegrp[agegrps[i]:agegrps[i + 1], i] = 1

    feat_agegrp = csr_matrix(feat_agegrp)
    features = [hstack([f, feat_agegrp]).tocsr() for f in features]
    censored_features = [
        hstack([f, feat_agegrp]).tocsr() for f in censored_features
    ]
    n_lags = np.hstack([n_lags, np.zeros(n_agegrps)])

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
        confidence_intervals=True
    )
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
    refitted_coeffs = cv_track.best_model["_coeffs"]
    normalized_age_effect = np.exp(refitted_coeffs[n_features * n_cols:])
    normalized_age_effect /= normalized_age_effect.sum()
    se_age = squared_error(normalized_time_drift, normalized_age_effect)
    ae_age = absolute_error(normalized_time_drift, normalized_age_effect)
    ape_age = absolute_percentage_error(normalized_time_drift,
                                        normalized_age_effect)
    for drug_id in range(n_features):
        # True coefficients
        c = np.exp(coeffs[drug_id * n_cols:(drug_id + 1) * n_cols])
        # ConvSCCS estimate
        d_refit = np.exp(refitted_coeffs[drug_id*n_cols:(drug_id+1)*n_cols])
        d_fit = np.exp(fitted_coeffs[drug_id*n_cols:(drug_id+1)*n_cols])

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

        se_features = squared_error(c, d_refit)
        ae_features = absolute_error(c, d_refit)
        ape_features = absolute_percentage_error(c, d_refit)
        result = ExperimentResult(
            experiment_id=experiment_id,
            version=version,
            seed=seed,
            model_id=model_id + "_refit",
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

    # # R models parameters
    agegrp_R = robjects.IntVector(agegrp)
    model_param = json.dumps({"kn": kn, "agegrp": agegrp})

    # SmoothSCCS (Ghebremichael 2016)
    model_id = "SmoothSCCS"
    for drug_id in range(n_features):
        # Learn
        df_single_drug = df[df.drugid == drug_id]
        start = time()
        results_R = sccs.smoothexposccs(
            indiv=df_single_drug.indiv,
            astart=df_single_drug.astart,
            aend=df_single_drug.aend,
            aevent=df_single_drug.aevent,
            adrug=df_single_drug.adrug,
            aedrug=df_single_drug.aedrug,
            agegrp=agegrp_R,
            kn=12,
            data=df_single_drug,
        )
        elapsed_time = time() - start
        # extract results
        results = extract_smoothexposccs_results(results_R)

        model_log = RModel(
            experiment_id=experiment_id,
            version=version,
            seed=seed,
            model_id=model_id,
            drug_id=drug_id,
            run_time=elapsed_time,
            model_params=model_param,
            r_model_track=dumps(results),
        )
        session.add(model_log)

        # Age effect
        normalized_age_effect = np.exp(np.repeat(np.array([0,
                                                           *results["coef"]]),
                                                 125))
        normalized_age_effect /= normalized_age_effect.sum()
        se_age = squared_error(normalized_time_drift, normalized_age_effect)
        ae_age = absolute_error(normalized_time_drift, normalized_age_effect)
        ape_age = absolute_percentage_error(
            normalized_time_drift, normalized_age_effect
        )
        # Features effect
        c = np.exp(coeffs[drug_id * n_cols:(drug_id + 1) * n_cols])
        d = results["exposure"]
        se_features = squared_error(c, d)
        ae_features = absolute_error(c, d)
        ape_features = absolute_percentage_error(c, d)

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

    # Non parametric SCCS (Ghebremichael 2017)
    model_id = "NonParaSCCS"
    for drug_id in range(n_features):
        # Learn
        df_single_drug = df[df.drugid == drug_id]
        start = time()
        results_R = sccs.nonparasccs(
            indiv=df_single_drug.indiv,
            astart=df_single_drug.astart,
            aend=df_single_drug.aend,
            aevent=df_single_drug.aevent,
            adrug=df_single_drug.adrug,
            aedrug=df_single_drug.aedrug,
            kn1=kn,
            kn2=kn,
            data=df_single_drug,
        )
        elapsed_time = time() - start

        # extract results
        results = extract_smoothexposccs_results(results_R)

        model_log = RModel(
            experiment_id=experiment_id,
            version=version,
            seed=seed,
            model_id=model_id,
            drug_id=drug_id,
            run_time=elapsed_time,
            model_params=model_param,
            r_model_track=dumps(results),
        )
        session.add(model_log)

        # Age effect
        normalized_age_effect = np.exp(results["age"])
        normalized_age_effect /= normalized_age_effect.sum()
        se_age = squared_error(normalized_time_drift, normalized_age_effect)
        ae_age = absolute_error(normalized_time_drift, normalized_age_effect)
        ape_age = absolute_percentage_error(
            normalized_time_drift, normalized_age_effect
        )

        # Features effect
        c = np.exp(coeffs[drug_id * n_cols:(drug_id + 1) * n_cols])
        d = results["exposure"]
        se_features = squared_error(c, d)
        ae_features = absolute_error(c, d)
        ape_features = absolute_percentage_error(c, d)

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
    pandas2ri.deactivate()
