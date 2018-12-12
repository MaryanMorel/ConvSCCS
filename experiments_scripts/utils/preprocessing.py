import numpy as np
import pandas as pd


def listdict2dictlist(LD):
    return dict(zip(LD[0], zip(*[d.values() for d in LD])))


def dictlist2listdict(DL):
    return [dict(zip(DL, t)) for t in zip(*DL.values())]


def check_feat_names(feat_names, n_features):
    if feat_names:
        if len(feat_names) != n_features:
            raise ValueError("`feat_names` sould have lenght %i" % n_features)
    else:
        feat_names = ["feature %i" % i for i in range(n_features)]
    feat_names = [n[:12] for n in feat_names]
    return feat_names


def to_nonparasccs(X, Y, C, n_lags):
    res = []
    n_samples = len(Y)
    for i in range(n_samples):
        x = X[i].tocoo()
        y = Y[i]
        c = C[i]  # (open side of the interval)
        nnz_y = np.nonzero(y)
        res.append(
            pd.DataFrame(
                {
                    "astart": 0,
                    "aend": c - 1,
                    "adrug": x.row,
                    "aedrug": np.minimum.reduce(
                        [np.repeat(c, len(x.row)), x.row + n_lags + 1]
                    )
                    - 1,
                    "aevent": nnz_y[0][0],
                    "drugid": x.col,
                },
                index=pd.Index(data=np.full(len(x.row), i), name="indiv"),
            )
        )
    return pd.concat(res)


def extract_smoothexposccs_results(results):
    names = list(results.names)
    return {n: np.array(results.rx2(n)).ravel() for n in names}
