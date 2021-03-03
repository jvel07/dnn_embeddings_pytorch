"""
Created by José Vicente Egas López
on 2021. 02. 15. 10 49

Class intended to instantiate SVM or SVR algorithms to perform classification or regression.
"""
from sklearn.svm import NuSVR


def train_svm_gpu(X, Y, X_eval, c, kernel, gamma):
    from thundersvm import SVC as thunder
    svc = thunder(kernel=kernel, C=c, probability=True, gamma=gamma, class_weight='balanced', max_iter=100000, gpu_id=0)
    svc.fit(X, Y)
    y_prob = svc.predict_proba(X_eval)
    return y_prob


def train_svr_gpu(X, Y, X_eval, c, kernel='linear', nu=0.5):
    from thundersvm import NuSVR as thunder
    svc = thunder(kernel=kernel, C=c, max_iter=100000, gpu_id=0, nu=nu, gamma='auto')
    svc.fit(X, Y)
    y_prob = svc.predict(X_eval)
    return y_prob


def train_svr_cpu(X, Y, X_eval, c, kernel='linear', nu=0.5):
    svc = NuSVR(kernel=kernel, C=c, max_iter=100000, nu=nu, gamma='auto')
    svc.fit(X, Y)
    y_prob = svc.predict(X_eval)
    return y_prob
