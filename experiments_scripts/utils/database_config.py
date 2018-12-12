from sqlalchemy import Column, DateTime, Integer, String, Float, LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()


class Experiment(Base):
    __tablename__ = "experiments"
    # Primary keys
    experiment_id = Column(Integer, primary_key=True)
    version = Column(Integer, primary_key=True)
    # Content
    description = Column(String(500), nullable=True)
    features_set = Column(String(50), nullable=False)
    effects = Column(String(1000), nullable=False)
    time_drift = Column(String(1000), nullable=False)
    n_features = Column(Integer, nullable=False)
    n_intervals = Column(Integer, nullable=False)
    n_cases = Column(Integer, nullable=False)
    sim_n_lags = Column(Integer, nullable=False)
    sim_n_corr = Column(Integer, nullable=False)
    sim_coeffs_obj = Column(LargeBinary, nullable=False)


class Simulation(Base):
    __tablename__ = "simulations"
    # Primary keys
    experiment_id = Column(Integer, primary_key=True)
    version = Column(Integer, primary_key=True)
    seed = Column(Integer, primary_key=True)
    # Content
    sim_adjacency_matrix = Column(LargeBinary, nullable=False)
    features_frequency = Column(LargeBinary, nullable=False)


class ConvSCCSModel(Base):
    __tablename__ = "convsccs_models"
    # Primary keys
    experiment_id = Column(Integer, primary_key=True)
    version = Column(Integer, primary_key=True)
    seed = Column(Integer, primary_key=True)
    model_id = Column(String(100), primary_key=True)
    # Content
    run_time = Column(Float, nullable=False)
    model_params = Column(String(1000), nullable=False)
    cv_track = Column(LargeBinary, nullable=True)


class RModel(Base):
    __tablename__ = "r_models"
    # Primary keys
    experiment_id = Column(Integer, primary_key=True)
    version = Column(Integer, primary_key=True)
    seed = Column(Integer, primary_key=True)
    model_id = Column(String(100), primary_key=True)
    drug_id = Column(Integer, primary_key=True)
    # Content
    run_time = Column(Float, nullable=False)
    model_params = Column(String(1000), nullable=False)
    r_model_track = Column(LargeBinary, nullable=True)


class ExperimentResult(Base):
    __tablename__ = "experiment_results"
    # Primary Keys
    experiment_id = Column(Integer, primary_key=True)
    version = Column(Integer, primary_key=True)
    seed = Column(Integer, primary_key=True)
    model_id = Column(String(100), primary_key=True)
    drug_id = Column(Integer, primary_key=True)
    # Content
    insert_date = Column(DateTime, nullable=False)
    se_features = Column(Float, nullable=False)
    se_age = Column(Float, nullable=False)
    ae_features = Column(Float, nullable=False)
    ae_age = Column(Float, nullable=False)
    ape_features = Column(Float, nullable=False)
    ape_age = Column(Float, nullable=False)


if __name__ == '__main__':
    import sys

    try:
        db_name = sys.argv[1]
    except IndexError:
        print(
            """Invalid usage, you should use the following syntax:
        
        python database_config.py [db_name.db]
        
        where db_name should be the path/name to the database you want to 
        create."""
        )

    engine = create_engine('sqlite:///' + db_name)
    Base.metadata.create_all(engine)
