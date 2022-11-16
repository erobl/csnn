from sklearn.neighbors import KernelDensity
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

class DensityDifference:
    def __init__(self, alpha, bandwidth=400, kernel='gaussian', sample_size=10000, disable_tqdm=False, n_jobs=10):
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.sample_size = sample_size
        self.disable_tqdm = disable_tqdm
        self.kdeneg = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.kdepos = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)
        self.alpha = alpha
        self.n_jobs=n_jobs

    def fit(self, x_train, y_train, concatenate=True):
        allneg_sample = []
        for xt in x_train[y_train == 0]:
            neg_sample = xt[np.random.permutation(xt.shape[0])[:self.sample_size],:]
            allneg_sample.append(neg_sample)
        allneg_sample_concat = np.concatenate(allneg_sample)
        self.kdeneg.fit(allneg_sample_concat)

        allpos_sample = []
        allpos_train = []
        for xt in tqdm(x_train[y_train == 1], disable=self.disable_tqdm):
            kp = KernelDensity(kernel=self.kernel, bandwidth=self.bandwidth)

            perm = np.random.permutation(xt.shape[0])[:2*self.sample_size]
            perm_train = perm[:self.sample_size]
            perm_sample = perm[self.sample_size:]
            ap_train = xt[perm_train,:]
            ap_sample = xt[perm_sample,:]

            allpos_train.append(ap_train)
            allpos_sample.append(ap_sample)

        allpos_train_concat = np.concatenate(allpos_train)
        self.kdepos.fit(allpos_train_concat)
        if concatenate:
            allpos_sample = np.concatenate(allpos_sample)
            allneg_sample = allneg_sample_concat
        else:
            allpos_sample = np.stack(allpos_sample)
            allneg_sample = np.stack(allneg_sample)

        return allpos_sample, allneg_sample

    def score_samples(self, X, return_scores=False):
        X_split = np.array_split(X, self.n_jobs)
        # score_neg = self.kdeneg.score_samples(X)
        score_negs = Parallel(n_jobs=self.n_jobs)(delayed(self.kdeneg.score_samples)(x) for x in tqdm(X_split, disable=self.disable_tqdm))
        score_neg = np.concatenate(score_negs)
        # score_pos = self.kdepos.score_samples(X)
        score_poss = Parallel(n_jobs=self.n_jobs)(delayed(self.kdepos.score_samples)(x) for x in tqdm(X_split, disable=self.disable_tqdm))
        score_pos = np.concatenate(score_poss)

        diff =  1 - np.exp(np.log(1 - self.alpha) + score_neg - score_pos)

        if return_scores:
            return diff, (score_pos, score_neg)
        return diff
