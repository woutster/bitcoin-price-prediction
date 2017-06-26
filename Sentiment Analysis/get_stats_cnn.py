import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support as score

labels_true = np.array(pd.read_csv('testdata.manual.2009.06.14.csv', sep=',', encoding='latin1', header=None))
labels_new = np.array(np.delete(labels_true[:, 0], np.where((labels_true[:, 0] == 2))), dtype=int)
labels_predicted_1 = np.array(pd.read_csv('prediction_1.csv', sep=',', encoding='latin1', header=None)[1], dtype=int)
labels_predicted_1[labels_predicted_1 == 1] = 4
labels_predicted_2 = np.array(pd.read_csv('prediction_2.csv', sep=',', encoding='latin1', header=None)[1], dtype=int)
labels_predicted_2[labels_predicted_2 == 1] = 4

precisionNo, recallNo, fscoreNo, supportNo = score(labels_new, labels_predicted_2)
print('labels: {}'.format(['Negative', 'Positive']))
print('precision: {}'.format(precisionNo))
print('recall: {}'.format(recallNo))
print('fscore: {}'.format(fscoreNo))
print('support: {}'.format(supportNo))
print('')
precisionYes, recallYes, fscoreYes, supportYes = score(labels_new, labels_predicted_1)
print('labels: {}'.format(['Negative', 'Positive']))
print('precision: {}'.format(precisionYes))
print('recall: {}'.format(recallYes))
print('fscore: {}'.format(fscoreYes))
print('support: {}'.format(supportYes))