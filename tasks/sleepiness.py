"""
Created by José Vicente Egas López
on 2021. 02. 15. 11 03

Estimation of the level of sleepiness of subjects using their speech.
Corpus used: DUSSELDORF Sleepiness Corpus.
Please, refer to the paper: ...
"""
import os

import numpy as np
from scipy import stats
from sklearn import preprocessing

import ml.svm_helper as svm_helper
from utils import load_data_full

list_c = [1e-6, 1e-5, 1e-4, 1e-3, 0.01, 0.1]

dev_preds_dic = {}
feat = 'spec'
x_train, x_dev, x_test, y_train, y_dev, y_test, file_n = load_data_full(
                                                            data_path='../data/sleepiness/spectrogram/xvecs_3',
                                                            layer_name='fc1')

x_combined = np.concatenate((x_train, x_dev))
y_combined = np.concatenate((y_train, y_dev))

std_flag = True
if std_flag:
    std_scaler = preprocessing.StandardScaler()
    x_train = std_scaler.fit_transform(x_train)
    x_dev = std_scaler.transform(x_dev)

    x_combined = std_scaler.fit_transform(x_combined)
    x_test = std_scaler.transform(x_test)

spear_scores = []
for c in list_c:
    preds = svm_helper.train_svr_gpu(x_train, y_train.ravel(), X_eval=x_dev, c=c)

    preds_orig = preds
    # preds = sh.linear_trans_preds_dev(y_train=y_train, preds_dev=preds)
    coef, p_std = stats.spearmanr(y_dev, preds)

    spear_scores.append(coef)
    print("with", c, "- spe:", coef)

    # util.results_to_csv(file_name='exp_results/results_{}_{}_srand_spec.csv'.format(task, feat_type[0]),
    #                     list_columns=['Exp. Details', 'Gaussians', 'Deltas', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
    #                     list_values=[os.path.basename(file_n), ga, feat_type[2], c, coef,
    #                                  std_flag, 'DEV', srand])

# Train SVM model on the whole training data with optimum complexity and get predictions on test data
optimum_complexity = list_c[np.argmax(spear_scores)]
print('\nOptimum complexity: {0:.6f}'.format(optimum_complexity))

y_pred = svm_helper.train_svr_gpu(x_combined, y_combined.ravel(), X_eval=x_test, c=optimum_complexity, nu=0.5)
# y_pred = sh.linear_trans_preds_test(y_train=y_train, preds_dev=preds_orig, preds_test=y_pred)
coef_test, p_2 = stats.spearmanr(y_test, y_pred)
# coef_test2 = np.corrcoef(y_test, y_pred)

print(os.path.basename(file_n), "\nTest results with", optimum_complexity, "- spe:", coef_test)
print(20 * '-')
# util.results_to_csv(file_name='exp_results/results_{}_{}_srand_spec.csv'.format(task, feat_type[0]),
#                     list_columns=['Exp. Details', 'Gaussians', 'Deltas', 'C', 'SPE', 'STD', 'SET', 'SRAND'],
#                     list_values=[os.path.basename(file_n), ga, feat_type[2], optimum_complexity, coef_test,
#                                  std_flag, 'TEST', srand])
# a = confusion_matrix(y_test, np.around(y_pred), labels=np.unique(y_train))
# plot_confusion_matrix_2(a, np.unique(y_train), 'conf.png', cmap='Oranges', title="Spearman CC .365")
