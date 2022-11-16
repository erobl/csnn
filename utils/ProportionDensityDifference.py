from sklearn.neighbors import KernelDensity
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

class ProportionDensityDifference:
    def __init__(self, bandwidth=400, kernel='gaussian', sample_size=5000, disable_tqdm=False, n_jobs=10):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.sample_size = sample_size
        self.disable_tqdm = disable_tqdm
        self.kdeneg = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kdepos = []
        self.alpha = []
        self.n_jobs = n_jobs

    def fit(self, x_train, y_train, alpha, concatenate=True):
        self.alpha = alpha[y_train == 1]
        allneg_sample = []
        for xt in x_train[y_train == 0]:
            neg_sample = xt[np.random.permutation(xt.shape[0])[:self.sample_size],:]
            allneg_sample.append(neg_sample)
        allneg_sample_concat = np.concatenate(allneg_sample)
        print("ans", allneg_sample_concat.shape)
        self.kdeneg.fit(allneg_sample_concat)

        self.kdepos = []
        allpos_sample = []
        for xt in tqdm(x_train[y_train == 1], disable=self.disable_tqdm):
            kp = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)

            perm = np.random.permutation(xt.shape[0])[:2*self.sample_size]
            perm_train = perm[:self.sample_size]
            perm_sample = perm[self.sample_size:]
            ap_train = xt[perm_train,:]
            ap_sample = xt[perm_sample,:]

            kp.fit(ap_train)
            self.kdepos.append(kp)

            allpos_sample.append(ap_sample)

        allpos_sample = np.stack(allpos_sample)
        allneg_sample = np.stack(allneg_sample)

        return allpos_sample, allneg_sample

    def score_samples(self, X, return_scores=False):
        def score(kde, x):
            return kde.score_samples(x)

        score_neg = Parallel(n_jobs=self.n_jobs)(delayed(self.kdeneg.score_samples)(x) for x in tqdm(X, disable=self.disable_tqdm))
        score_pos = Parallel(n_jobs=self.n_jobs)(delayed(score)(kp, x) for kp, x in tqdm(zip(self.kdepos, X), disable=self.disable_tqdm))

        diff = []
        for alpha, sp, sn in zip(self.alpha, score_pos, score_neg):
            diff.append(1 - np.exp(np.log(1 - alpha) + sn - sp))

        if return_scores:
            return diff, (score_pos, score_neg)
        return diff
