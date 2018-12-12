import numpy as np
from scipy.stats import beta, norm, truncnorm, wald


class Effects:
    def __init__(self, n_intervals):
        self.n_intervals = n_intervals

    def rapid(self):
        a, b = 5, 100
        x_sim = np.linspace(wald.ppf(0.01), wald.ppf(0.99), self.n_intervals)
        y = np.exp(wald.pdf(x_sim) / 2)
        y[2:] = y[:-2]
        y[:2] = 1

        return y

    def early(self):
        a, b = 5, 30
        x_sim = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b),
                            self.n_intervals)
        y = np.exp(beta.pdf(x_sim, a, b) / 4) / 10 + 1
        y -= y[0] - 1
        y[y < 1] = 1
        return y

    def intermediate(self):
        a, b = 4, 4.5
        x_sim = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b),
                            self.n_intervals)
        y = np.exp(beta.pdf(x_sim, a, b)) / 30 + 1
        y -= y[0] - 1 
        y[25:] = 1
        return y

    def late(self):
        a, b = 2, 100
        x_sim = np.linspace(truncnorm.ppf(0.01, a, b),
                            truncnorm.ppf(0.99, a, b), self.n_intervals)
        x = np.arange(50)
        y = np.exp(truncnorm.pdf(x_sim, a, b))
        y = y[::-1] / 15 + 1
        y -= y[0] - 1 
        return y

    def delayed(self):
        a, b = 1000, 1000
        x_sim = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b),
                            self.n_intervals)
        y = np.exp(beta.pdf(x_sim, a, b)) / 7e15 + 1
        y -= y[0] - 1 
        return y

    def null(self):
        y = np.ones(self.n_intervals)
        return y

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

    
class Effects2:
    def __init__(self, n_intervals, n_effective_intervals=50):
        self.n_intervals = n_intervals
        self.n_effective_intervals = min(n_intervals, n_effective_intervals)
        
    def rapid(self):
        a, b = 5, 100
        y = np.linspace(wald.ppf(0.01), wald.ppf(0.99), self.n_effective_intervals)
        y = np.exp(wald.pdf(y) / 1.6)
        y[2:] = y[:-2]
        y[:2] = 1
        effect = np.ones(self.n_intervals)
        effect[0:self.n_effective_intervals] = y
        return effect

    def early(self):
        a, b = 5, 30
        y = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b),
                        self.n_effective_intervals)
        y = np.exp(beta.pdf(y, a, b) / 3) / 10 + 1
        y -= y[0] - 1
        y[y < 1] = 1
        effect = np.ones(self.n_intervals)
        delay = int(self.n_effective_intervals / 5)
        max_time = min(self.n_intervals, self.n_effective_intervals+delay)
        effect[0+delay:max_time] = y
        return effect

    def intermediate(self):
        a, b = 4, 4.5
        x_sim = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b),
                            self.n_effective_intervals*2)
        y = np.exp(beta.pdf(x_sim[0:self.n_effective_intervals], a, b) / 3.3) + 1
        y -= y[0] - 1 
        effect = np.ones(self.n_intervals)
        delay = int(self.n_effective_intervals * 2)
        max_time = min(self.n_intervals, self.n_effective_intervals+delay)
        effect[0+delay:max_time] = y
        return effect

    def late(self):
        a, b = 2, 3
        y = np.linspace(truncnorm.ppf(0.01, a, b),
                        truncnorm.ppf(0.99, a, b), self.n_intervals)
        y = np.exp(truncnorm.pdf(y, a, b))
        y = y[::-1] / 13 + 1
        y -= y[0] - 1 
        return y

    def delayed(self):
        a, b = 2, 2
        y = np.linspace(beta.ppf(0.01, a, b), beta.ppf(0.99, a, b),
                        self.n_effective_intervals)
        y = np.exp(beta.pdf(y, a, b)) / 4 + 1
        y -= y[0] - 1
        y[y < 1] = 1
        effect = np.ones(self.n_intervals)
        delay = int(self.n_effective_intervals * 4)
        max_time = min(self.n_intervals, self.n_effective_intervals+delay)
        effect[0+delay:max_time] = y
        return effect

    def null(self):
        y = np.ones(self.n_intervals)
        return y

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


class CustomEffects:
    def __init__(self, n_intervals):
        self.n_intervals = n_intervals
        self._curves_type_dict = {
            1: (5, 1),
            2: (2, 2),
            3: (.5, .5),
            4: (2, 5),
            5: (1, 3)
        }

    def constant_effect(self, amplitude, cut=0):
        risk_curve = np.ones(self.n_intervals) * amplitude
        if cut:
            risk_curve[cut:] = 1
        return risk_curve

    def bell_shaped_effect(self, amplitude, width, lag=0, cut=0):
        self._check_params(lag, width, amplitude, cut)
        if width % 2 == 0:
            width += 1
        effect = norm(0, width / 5).pdf(np.arange(width) - int(width / 2))
        return self._create_risk_curve(effect, amplitude, cut, width, lag)

    def increasing_effect(self, amplitude, width=None, lag=0, cut=0,
                          curvature_type=1):
        if width is None:
            width = self.n_intervals
        self._check_params(lag, width, amplitude, cut)
        if curvature_type not in np.arange(5) + 1:
            raise ValueError('curvature type should be in {1, 2, 3, 4, 5}')
        a, b = self._curves_type_dict[curvature_type]
        effect = beta(a, b).cdf(np.arange(width) / width)
        return self._create_risk_curve(effect, amplitude, cut, width, lag)

    def _check_params(self, lag, width, amplitude, cut):
        if cut is not None and cut >= width:
            raise ValueError('cut should be < width')
        if lag > self.n_intervals:
            raise ValueError('n_intervals should be > lag')
        if amplitude <= 0:
            raise ValueError('amplitude should be > 0')

    def _create_risk_curve(self, effect, amplitude, cut, width, lag):
        if cut:
            effect = effect[:int(width - cut)]
        end_effect = int(lag + width - cut)
        if end_effect > self.n_intervals:
            end_effect = self.n_intervals
        effect = effect[:end_effect - lag]

        M = effect.max()
        m = effect.min()
        effect = (effect - m) / (M - m)
        effect *= (amplitude - 1)
        risk_curve = np.ones((self.n_intervals))
        risk_curve[lag:end_effect] += effect
        return risk_curve

    @staticmethod
    def negative_effect(positive_effect):
        return np.exp(-np.log(positive_effect))
