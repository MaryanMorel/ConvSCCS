import numpy as np
from scipy.stats import beta, norm, truncnorm, wald


class Effects:
    def __init__(self, n_intervals):
        self.n_intervals = n_intervals

    def yonas_1(self):
        a, b = 5, 20
        x_sim = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b),
                            self.n_intervals)
        y = np.exp(beta.pdf(x_sim, a, b) / 4) 
        y -= y.min() - 1 
        return y

    def yonas_2(self):
        y = np.repeat(4, self.n_intervals)
        return y

    def yonas_3(self):
        a, b = 0, 3
        x_sim = np.linspace(truncnorm.ppf(0.01, a, b),
                            truncnorm.ppf(0.99, a, b), self.n_intervals)
        x = np.arange(self.n_intervals)
        y = np.exp(truncnorm.pdf(x_sim, a, b) * 1.35)
        return y

    def yonas_4(self):
        a, b = 1, 3
        x_sim = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b),
                            self.n_intervals)
        y = np.exp(beta.pdf(x_sim, a, b) / 2.15) 
        return y
    
    @staticmethod
    def negative_effect(positive_effect):
        return np.exp(-np.log(positive_effect))
