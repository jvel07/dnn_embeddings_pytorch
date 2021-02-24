"""
Created by José Vicente Egas López
on 2021. 02. 19. 16 36

"""

import utils
import numpy as np
import cyvlfeat as vlf


def load_flevel_feats(path):
    list_feature_files = utils.get_files_abspaths(path=path, file_type='.npy')
    print("Loading features from {}: {} files".format(path, len(list_feature_files)))
    feat_list = []
    for file in list_feature_files:
        feat = np.load(file)
        feat_list.append(feat)
    return feat_list


def do_gmm(features, num_gaussian):
    print("Training {}-GMM for fishers...".format(num_gaussian))
    means, covs, priors, LL, posteriors = vlf.gmm.gmm(features, n_clusters=num_gaussian, n_repetitions=2, verbose=0)
    return means, covs, priors


def extract_fishers(features, means, covs, priors):
    # Extracting fisher vecs
    print("Extracting FV encodings...")
    fish = vlf.fisher.fisher(features.transpose(), means.transpose(), covs.transpose(), priors, improved=True)
    return fish
