import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import lightgbm as lgb
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import Lasso, LassoCV
from sklearn.linear_model import ElasticNet,ElasticNetCV
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
import datetime
import warnings
warnings.filterwarnings("ignore")

#########################################################################
# data preparation:
#dataset = pd.read_csv('C:/Users/Shijie Wang/Desktop/python_learning/Dataset/overfitting/overfitting.csv')
#dataset = dataset.drop(['Target_Leaderboard', 'Target_Evaluate'], axis = 1)
#dataset.set_index(['case_id'], inplace = True)

# split train and test: train is 150 * 201, ratio is 0.0075.
#random_seed = 718
#train = dataset[dataset['train'] == 1]
#test = dataset[dataset['train'] == 0]
#train = train.drop('train', axis = 1)
#test = test.drop(['train'], axis = 1)

# write to csv file:
#train.to_csv('C:/Users/Shijie Wang/Desktop/python_learning/Dataset/overfitting/train.csv')
#test.to_csv('C:/Users/Shijie Wang/Desktop/python_learning/Dataset/overfitting/test.csv')
#########################################################################



###################                read train test data                   ###################
#train = pd.read_csv('C:/Users/DELL/Desktop/python_learning/Dataset/overfitting/train.csv')
#test = pd.read_csv('C:/Users/DELL/Desktop/python_learning/Dataset/overfitting/test.csv')

train = pd.read_csv('C:/Users/Shijie Wang/Desktop/python_learning/Dataset/overfitting/train.csv')
test = pd.read_csv('C:/Users/Shijie Wang/Desktop/python_learning/Dataset/overfitting/test.csv')
train = train.iloc[0:150, :]

###################                Feature Engineering Analysis:                  ###################
train_data = train.drop(['Target_Practice', 'case_id'], axis = 1)
train_label = train['Target_Practice'].ravel()
test_data = test.drop(['Target_Practice', 'case_id'], axis = 1)
test_label = test['Target_Practice'].ravel()
feature = train_data.columns

###################                print some information of data                  ###################
# print('The row of data frame is %d'%(train.shape[0]), 'the column of data frame is %d'%(train.shape[1]), sep = '\n')
print('The dimension of data frame is : {}'.format(list(train.shape)),
      'The missing feature ration of each column is : {}'.format([(train.shape[0] - train.iloc[:,x].count())/train.shape[0] for x in range(0,train.shape[1])]),
      'The traget(binary classification outcome) distribution is : {}'.format(dict(train['Target_Practice'].value_counts())),
      sep = '\n')

###################                Data Distributions of First 28 columns:                  ###################
print('Data Distributions of First 28 columns')
plt.figure(figsize = (28,24))
for i, col in enumerate(list(train.columns[2:30])):
    # the first two columns is useless
    plt.subplot(7, 4, i + 1)
    plt.hist(train[col], alpha = 0.8)
    plt.title(col)

###################                overview of label distribution:                  ###################
train['Target_Practice'].value_counts()


###################                Correlation / VIF:                  ###################
vif = pd.DataFrame(np.linalg.inv(train_data.corr().values).diagonal(), index = train_data.columns)
vif_df = vif.sort_values(by = 0, ascending = False)[:30].reset_index()
vif_df.columns = ['id', 'vif']


###################                bar plot of VIF Top 30:                  ###################
sns.set_style('whitegrid')
sns.set(font_scale = 2)
plt.figure(figsize = (28,24))
plt.title('VIF Value top 30', y = 1, size = 14)
graph = sns.barplot(y = 'id', x = 'vif', data = vif_df, orient = 'h', order = vif_df['id'])#sns_plot = sns_plot.get_figure()
graph.savefig("C:/Users/Shijie Wang/Desktop/STAT 718 code/Team Project/accuracy.png")

sns.set(font_scale = 2)
plt.figure(figsize = (28,24))
graph = sns.scatterplot(x = "Method", y = "Value", hue = "Model: Accuracy DataFrame", style = 'Model: Accuracy DataFrame', alpha = 1, s = 100, data = Accuray_df_plot.sort_values("Value", ascending = False), markers = ['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
graph.axhline(y = 0.7, color = 'r', alpha = 0.4, linestyle = '-')

###################                Basic Moldeling :                  ###################
###################                without feature engineering                  ###################
###################                K fold basic model :                 ###################
random_seed = 718
folds = 10
kf = KFold(n_splits = folds, shuffle = False, random_state = random_seed)
#################################################################################################################################################
###########################################                 Model Train function                     ###########################################
#################################################################################################################################################
def train_model(x, X_test, y, params, folds = folds, model_type = None, clf = None, with_feature_importance = False, feature = feature):
    '''
    input :

    param x : train data
    param X_test : test data for the final output
    param y : x_train data's label
    params : the parameter set for the model
    param model_type : different code for lgb and sklearn model
    param clf : model input

    return :
    oof_train : use basic model to predict the valid data; valid data from x_train as the second stacking model's input/max_features for learning
    predictions : use basic model to predict the test data; use second layer to predict this to get the final output
    feature_importances_df : contains feature importance; for feature selection or so

    stacking model:
    oof_train as feature of second layers
    y_train as the label and train the second layer with (oof_train, y_train)
    final output: predict the (predictions) with trained second layer
    '''
    # Initialization of list :
    oof_train = np.zeros(len(x))
    oof_test_kf = np.empty((folds, len(X_test)))
    predictions = np.zeros(len(X_test))
    accuracys = []
    scores = []
    feature_importance = np.empty((x.shape[1],folds))
    time = datetime.datetime.now()

    if with_feature_importance == True  :

        print(str(clf) + ' : has started ')
        for fold, (train_index, valid_index) in enumerate(kf.split(x)):
            x_train, x_valid = x.iloc[train_index], x.iloc[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            # print(' fold number is {}, x_train number is: {}, y_train number is : {}'.format(fold, len(x_train), len(y_train)))
            # model fit
            model = clf(**params)
            model.fit(x_train, y_train)
            y_pred_valid = model.predict(x_valid)                                # model's prediction on valid set which divided in 10
            y_pred = model.predict(X_test)                                       # model's predictions on test set
            score = roc_auc_score(y_valid, y_pred_valid)                         # model's auc roc score/ cv
            accuracy = accuracy_score(y_valid, y_pred_valid)
            print(' the currnt fold number : {}, auc_roc_score : {:.3f}'.format(fold, score))

            # in the fold loop :
            oof_train[valid_index] = y_pred_valid                                 # all valid index sum up tp complete row/columns
            oof_test_kf[fold, :] = y_pred                                         # in every fold's predictions on test set
            scores.append(score)
            accuracys.append(accuracy)
            feature_importance[:, fold] = model.feature_importances_              # feature importances, rows = fold numbers

        predictions[:] = oof_test_kf.mean(axis = 0)                               # mean of each columns' y_pred
        # return feature importance data frame
        feature_importance_df = pd.DataFrame({'feature': feature, 'importance' : feature_importance.mean(axis = 1)})
        cv_mean_score = np.mean(scores)
        cv_mean_accuracy = np.mean(accuracys)
        print(' CV mean auc_roc_score : {:.4f},  auc_roc_score std : {:.4f}'.format(cv_mean_score , np.std(scores)))
        print(' Time : {}'.format(datetime.datetime.now() - time))
        return oof_train.reshape(-1,1), predictions.reshape(-1,1), feature_importance_df, cv_mean_score, cv_mean_accuracy

    # model without feature importances
    else :

        print(str(clf) + ' : has started ')
        for fold, (train_index, valid_index) in enumerate(kf.split(x)):
            x_train, x_valid = x.iloc[train_index], x.iloc[valid_index]
            y_train, y_valid = y[train_index], y[valid_index]
            # model fit
            model = clf(**params)
            model.fit(x_train, y_train)
            y_pred_valid = model.predict(x_valid)                                # model's prediction on valid set which divided in 10
            y_pred = model.predict(X_test)                                       # model's predictions on test set
            score = roc_auc_score(y_valid, y_pred_valid)                         # model's auc roc score/ cv
            accuracy = accuracy_score(y_valid, y_pred_valid)
            print(' the currnt fold number : {}, auc_roc_score : {:.3f}'.format(fold, score))

            # in the fold loop :
            oof_train[valid_index] = y_pred_valid                                 # all valid index sum up tp complete row/columns
            oof_test_kf[fold, :] = y_pred                                         # in every fold's predictions on test set
            scores.append(score)
            accuracys.append(accuracy)

        predictions[:] = oof_test_kf.mean(axis = 0)                               # mean of each columns' y_pred
        cv_mean_score = np.mean(scores)
        cv_mean_accuracy = np.mean(accuracys)
        print(' CV mean auc_roc_score : {:.4f},  auc_roc_score std : {:.4f}'.format(cv_mean_score, np.std(scores)))
        print(' Time : {}'.format(datetime.datetime.now() - time))
        return oof_train.reshape(-1,1), predictions.reshape(-1,1), cv_mean_score, cv_mean_accuracy

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier


###################                params set :                  ###################
seed = random_seed
rf_params = {'n_estimators': 1500,  'max_depth': 6, 'min_samples_leaf': 2, 'max_features' : 'sqrt', 'verbose': 0, 'random_state': seed}
et_params = {'n_estimators': 1500, 'max_features': 0.5, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0, 'random_state': seed}
ada_params = { 'n_estimators': 1500, 'learning_rate' : 0.01, 'random_state': seed}
gbdt_params = {'n_estimators': 1500, 'max_features': 0.2, 'max_depth': 5, 'min_samples_leaf': 2,'verbose': 0, 'random_state': seed}
xgb_params = {'n_estimators': 1500, 'learning_rate' : 0.01, 'max_depth': 7, 'subsample' : 0.1, 'reg_alpha' : 0.1, 'reg_lambda' : 0.1, 'random_state': seed}
lgb_params = {'boosting_type': 'gbdt', 'num_leaves': 50, 'reg_alpha': 1.0, 'reg_lambda' : 1.0, 'max_depth': -1, 'n_estimators': 1500,
             'objective': 'binary', 'subsample' : 0.7, 'colsample_bytree': 0.7, 'learning_rate': 0.01, 'min_child_weight': 50, 'random_state': seed}
lr_params = {'class_weight': 'balanced', 'penalty': 'l1', 'C': 0.1, 'solver': 'liblinear', 'random_state': seed}
svm_params = {'probability' : True, 'gamma': 'auto', 'C': 0.01, 'kernel': 'linear', 'random_state': seed}
sgd_params = {'max_iter': 1000, 'tol': 0.001, 'loss': 'hinge', 'penalty': 'l2', 'random_state': seed}
qda_params = {'priors': None, 'reg_param': 0.0, }
lda_params = {'solver':'svd', 'store_covariance': True}
knn_params = {'n_neighbors': 3}
dt_params = {'max_features': 0.5, 'max_depth': 8, 'min_samples_leaf': 2, 'random_state': seed}
gp_params = {'kernel': 1.0 * RBF(2.0)}
mlp_params = {'hidden_layer_sizes': 4, 'learning_rate' : 'adaptive', 'random_state': seed, 'early_stopping': True}

###################                Basic modeling :                  ###################
########################################################################################
rf_oof_train, rf_oof_predictions, rf_feature_importance, rf_cv_score, rf_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = rf_params, model_type = 'sklearn', clf = RandomForestClassifier, with_feature_importance = True, feature = feature)
et_oof_train, et_oof_predictions, et_feature_importance, et_cv_score, et_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = et_params, model_type = 'sklearn', clf = ExtraTreesClassifier, with_feature_importance = True, feature = feature)
ada_oof_train, ada_oof_predictions, ada_feature_importance, ada_cv_score, ada_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = ada_params, model_type = 'sklearn', clf = AdaBoostClassifier, with_feature_importance = True, feature = feature)
xgb_oof_train, xgb_oof_predictions, xgb_feature_importance, xgb_cv_score, xgb_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = xgb_params, model_type = 'sklearn', clf = XGBClassifier, with_feature_importance = True, feature = feature)
lgb_oof_train, lgb_oof_predictions, lgb_feature_importance, lgb_cv_score, lgb_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = lgb_params, model_type = 'sklearn', clf = lgb.LGBMClassifier, with_feature_importance = True, feature = feature)
gbdt_oof_train, gbdt_oof_predictions, gbdt_feature_importance, gbdt_cv_score, gbdt_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = gbdt_params, model_type = 'sklearn', clf = GradientBoostingClassifier, with_feature_importance = True, feature = feature)
lr_oof_train, lr_oof_predictions, lr_cv_score, lr_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = lr_params, model_type = 'sklearn', clf = LogisticRegression, with_feature_importance = False)
svm_oof_train, svm_oof_predictions, svm_cv_score, svm_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = svm_params, model_type = 'sklearn', clf = SVC, with_feature_importance = False)
sgd_oof_train, sgd_oof_predictions, sgd_cv_score, sgd_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = sgd_params, model_type = 'sklearn', clf = SGDClassifier, with_feature_importance = False)
qda_oof_train, qda_oof_predictions, qda_cv_score, qda_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = qda_params, model_type = 'sklearn', clf = QuadraticDiscriminantAnalysis, with_feature_importance = False)
lda_oof_train, lda_oof_predictions, lda_cv_score, lda_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = lda_params, model_type = 'sklearn', clf = LinearDiscriminantAnalysis, with_feature_importance = False)
knn_oof_train, knn_oof_predictions, knn_cv_score, knn_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = knn_params, model_type = 'sklearn', clf = KNeighborsClassifier, with_feature_importance = False)
dt_oof_train, dt_oof_predictions, dt_cv_score, dt_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = dt_params, model_type = 'sklearn', clf = DecisionTreeClassifier, with_feature_importance = False)
gp_oof_train, gp_oof_predictions, gp_cv_score, gp_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = gp_params, model_type = 'sklearn', clf = GaussianProcessClassifier, with_feature_importance = False)
mlp_oof_train, mlp_oof_predictions, mlp_cv_score, mlp_cv_accuracy = train_model(x = train_data, X_test = test_data, y = train_label, params = mlp_params, model_type = 'sklearn', clf = MLPClassifier, with_feature_importance = False)


###################                score output save:                  ###################
model_list = ['Random_Forest', 'Extra_Tree', 'Adaboost', 'Xgboost', 'Lightgbm', 'GBDT', 'Logistic_Reg', 'SVM', 'SGD_clf', 'QDA', 'LDA', 'KNN', 'Decision Tree', 'Gaussian Process', 'Mlp']
auc_score_without_eda = [rf_cv_score, et_cv_score, ada_cv_score, xgb_cv_score, lgb_cv_score, gbdt_cv_score, lr_cv_score, svm_cv_score, sgd_cv_score, qda_cv_score, lda_cv_score, knn_cv_score, dt_cv_score, gp_cv_score, mlp_cv_score]
accuracy_score_without_eda = [rf_cv_accuracy, et_cv_accuracy,ada_cv_accuracy, xgb_cv_accuracy, lgb_cv_accuracy, gbdt_cv_accuracy, lr_cv_accuracy, svm_cv_accuracy, sgd_cv_accuracy, qda_cv_accuracy, lda_cv_accuracy, knn_cv_accuracy, dt_cv_accuracy, gp_cv_accuracy, mlp_cv_accuracy]
auc_score = pd.DataFrame({'model':model_list, 'auc_without_EDA':auc_score_without_eda, 'accuracy_without_EDA':accuracy_score_without_eda})
print('The auc_roc and accuracy score in different models are : \n {},'.format(auc_score))


###################                Feature Engineering :                  ###################
###################               1. Feature Importance                   ###################
###################          Top var output for tree model :                  ###################
model_fi = pd.DataFrame({'feature': rf_feature_importance['feature'],
                         'rf_feature_importance': rf_feature_importance['importance'],
                         'et_feature_importance': et_feature_importance['importance'],
                         'ada_feature_importance' : ada_feature_importance['importance'],
                         'xgb_feature_importance' : xgb_feature_importance['importance'],
                         #'lgb_feature_importance' : lgb_feature_importance['importance'],
                         'gbdt_feature_importance' : gbdt_feature_importance['importance']})
model_fi['feature_importance_mean'] = model_fi.iloc[:,1:5].mean(axis = 1)
var_df =  model_fi.sort_values(by = 'feature_importance_mean', ascending = False)[['feature', 'feature_importance_mean']].reset_index().drop('index', axis = 1)
top_var = var_df['feature'][:50]

###################                plot top 50 features                  ###################
sns.set_style('darkgrid')
sns.set(font_scale = 2)
plt.figure(figsize = (28,24))
plt.title('Top 50 Features According to Feature Importance', y = 1, size = 24)
sns_plot = sns.barplot(y = 'feature', x = 'feature_importance_mean', data = var_df, orient = 'h', order = top_var)
sns_plot = sns_plot.get_figure()
sns_plot.savefig("C:/Users/Shijie Wang/Desktop/STAT 718 code/Team Project/fi.png")


###################               plot top 1 feature's distribution                 ###################
sns.set_style('whitegrid')
plt.figure(figsize = (14,8))
plt.title('Top 1 Feature According to Feature Importance', y = 1, size = 14)
sns.distplot(train['var_29'], kde = True, rug = False)

###################                feature engineering :                  ###################
selected_feature = train_data[top_var]
x_test_50 = test_data[top_var]
print('The Selected Feature dimension is : {} '.format(selected_feature.shape))

###################                Basic modeling with Feature Selected:                  ###################
#############################################################################################################
rf_oof_train_50, rf_oof_predictions_50, rf_feature_importance_50, rf_cv_score, rf_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = rf_params, model_type = 'sklearn', clf = RandomForestClassifier, with_feature_importance = True, feature = top_var)
et_oof_train_50, et_oof_predictions_50, et_feature_importance_50, et_cv_score, et_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = et_params, model_type = 'sklearn', clf = ExtraTreesClassifier, with_feature_importance = True, feature = top_var)
ada_oof_train_50, ada_oof_predictions_50, ada_feature_importance_50, ada_cv_score, ada_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = ada_params, model_type = 'sklearn', clf = AdaBoostClassifier, with_feature_importance = True, feature = top_var)
xgb_oof_train_50, xgb_oof_predictions_50, xgb_feature_importance_50, xgb_cv_score, xgb_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = xgb_params, model_type = 'sklearn', clf = XGBClassifier, with_feature_importance = True, feature = top_var)
lgb_oof_train_50, lgb_oof_predictions_50, lgb_feature_importance_50, lgb_cv_score, lgb_cv_accuracy= train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = lgb_params, model_type = 'sklearn', clf = lgb.LGBMClassifier, with_feature_importance = True, feature = top_var)
gbdt_oof_train_50, gbdt_oof_predictions_50, gbdt_feature_importance, gbdt_cv_score, gbdt_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = gbdt_params, model_type = 'sklearn', clf = GradientBoostingClassifier, with_feature_importance = True, feature = top_var)
lr_oof_train_50, lr_oof_predictions_50, lr_cv_score, lr_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = lr_params, model_type = 'sklearn', clf = LogisticRegression, with_feature_importance = False)
svm_oof_train_50, svm_oof_predictions_50, svm_cv_score, svm_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = svm_params, model_type = 'sklearn', clf = SVC, with_feature_importance = False)
sgd_oof_train_50, sgd_oof_predictions_50, sgd_cv_score, sgd_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = sgd_params, model_type = 'sklearn', clf = SGDClassifier, with_feature_importance = False)
qda_oof_train_50, qda_oof_predictions_50, qda_cv_score, qda_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = qda_params, model_type = 'sklearn', clf = QuadraticDiscriminantAnalysis, with_feature_importance = False)
lda_oof_train_50, lda_oof_predictions_50, lda_cv_score, lda_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = lda_params, model_type = 'sklearn', clf = LinearDiscriminantAnalysis, with_feature_importance = False)
knn_oof_train_50, knn_oof_predictions_50, knn_cv_score, knn_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = knn_params, model_type = 'sklearn', clf = KNeighborsClassifier, with_feature_importance = False)
dt_oof_train_50, dt_oof_predictions_50, dt_cv_score, dt_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = dt_params, model_type = 'sklearn', clf = DecisionTreeClassifier, with_feature_importance = False)
gp_oof_train_50, gp_oof_predictions_50, gp_cv_score, gp_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = gp_params, model_type = 'sklearn', clf = GaussianProcessClassifier, with_feature_importance = False)
mlp_oof_train_50, mlp_oof_predictions_50, mlp_cv_score, dt_cv_accuracy = train_model(x = selected_feature, X_test = x_test_50, y = train_label, params = mlp_params, model_type = 'sklearn', clf = MLPClassifier, with_feature_importance = False)

###################                Basic Moldeling :                  ###################
# Score comparsion :
auc_score_with_50 = [rf_cv_score, et_cv_score, ada_cv_score, xgb_cv_score, lgb_cv_score, gbdt_cv_score, lr_cv_score, svm_cv_score, sgd_cv_score, qda_cv_score, lda_cv_score, knn_cv_score, dt_cv_score, gp_cv_score, mlp_cv_score]
accuracy_score_50 = [rf_cv_accuracy, et_cv_accuracy,ada_cv_accuracy, xgb_cv_accuracy, lgb_cv_accuracy, gbdt_cv_accuracy, lr_cv_accuracy, svm_cv_accuracy, sgd_cv_accuracy, qda_cv_accuracy, lda_cv_accuracy, knn_cv_accuracy, dt_cv_accuracy, gp_cv_accuracy, mlp_cv_accuracy]
auc_score['auc_with_50_vars'] = auc_score_with_50
auc_score['accuracy_with_50_vars'] = accuracy_score_50
auc_score


#################################################################################################################################################
###################                Boosting Method :                  ###################
###################                My Own Adaboost Function                  ###################
###################                Adaboost Classifier:                  ###################
def my_adaboost_clf(Y_train, X_train, X_test, M = None, weak_clf = None):

    '''
    input :
    params : the parameter set for the model
    param M: number of iterations
    param weak_clf : target weak classifier to be improved
    '''

    n_train, n_test = len(X_train), len(X_test)
    # Initialize weights
    w = np.ones(n_train) / n_train
    pred_train, pred_test = [np.zeros(n_train), np.zeros(n_test)]

    for i in range(M):
        # Fit a classifier with the specific weights
        weak_clf.fit(X_train, Y_train, sample_weight = w)
        pred_train_i = weak_clf.predict(X_train)
        pred_test_i = weak_clf.predict(X_test)

        # Indicator function
        miss = [int(x) for x in (pred_train_i != Y_train)]
        print("weak_clf_%02d train acc: %.4f" % (i + 1, 1 - sum(miss) / n_train))

        # Error updating
        err_m = np.dot(w, miss)

        # Alpha updating
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))

        # New weights
        miss2 = [x if x == 1 else -1 for x in miss]  # -1 * y_i * G(x_i): 1 / -1
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        w = w / sum(w)

        # Add to prediction
        pred_train_i = [1 if x == 1 else -1 for x in pred_train_i]
        pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
        pred_train = pred_train + np.multiply(alpha_m, pred_train_i)
        pred_test = pred_test + np.multiply(alpha_m, pred_test_i)

    pred_train = (pred_train > 0) * 1
    pred_test = (pred_test > 0) * 1

    print("My AdaBoost clf train accuracy: %.4f" % (sum(pred_train == Y_train) / n_train))
    return pred_train, pred_test
#################################################################################################################################################


###################                Basic Moldeling :                  ###################
###################                Set Parameter :                  ###################
random_seed = 718
folds = 10
kf = KFold(n_splits = folds, shuffle = True, random_state = random_seed)
#################################################################################################################################################
###########################################                 Model Train function                     ###########################################
#################################################################################################################################################
def model_train(x, X_test, y, folds = folds, clf = None, iter_res = 0):
    '''
    input :

    param x : train data
    param X_test : test data for the final output
    param y : x_train data's label
    param clf : model input

    '''
    # Initialization of list :
    oof_train = np.zeros(len(x))
    oof_test_kf = np.empty((folds, len(X_test)))
    predictions = np.zeros(len(X_test))
    accuracys = []
    scores = []
    time = datetime.datetime.now()

    #print(str(clf) + ' : has started ')
    for fold, (train_index, valid_index) in enumerate(kf.split(x)):
        x_train, x_valid = x.iloc[train_index], x.iloc[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
        # print(' fold number is {}, x_train number is: {}, y_train number is : {}'.format(fold, len(x_train), len(y_train)))
        # model fit
        model = clf
        model.fit(x_train, y_train)
        y_pred_valid = model.predict(x_valid)                                # model's prediction on valid set which divided in 10
        y_pred = model.predict(X_test)                                       # model's predictions on test set
        y_pred_train = model.predict(x_train)
        score = roc_auc_score(y_valid, y_pred_valid)                         # model's auc roc score/ cv
        accuracy = accuracy_score(y_valid, y_pred_valid)
        train_accuracy = accuracy_score(y_pred_train, y_train)

        if iter_res == 1:
            print(' the currnt fold number : {}, oof accuracy : {:.3f}, train accuracy: {:.3f}'.format(fold, accuracy, train_accuracy))

        # in the fold loop :
        oof_train[valid_index] = y_pred_valid                                 # all valid index sum up tp complete row/columns
        oof_test_kf[fold, :] = y_pred                                         # in every fold's predictions on test set
        scores.append(score)
        accuracys.append(accuracy)

    cv_mean_score = np.mean(scores)
    cv_mean_accuracy = np.mean(accuracys)
    print(' CV mean accuracy : {:.4f},  CV mean Auc-Roc : {:.4f}'.format(cv_mean_accuracy , cv_mean_score))
    print(' Time : {}'.format(datetime.datetime.now() - time))

    return cv_mean_score, cv_mean_accuracy


##############################################################################################
###################                2. Developing models                    ###################
###################                By Cross Validation                    ####################
##############################################################################################

###################                Read in mcp/scad/top_var result from R                    ###################
###################                MCP Variable output from R                    ###################
mcp_var = pd.read_csv("C:\\Users\\Shijie Wang\\Desktop\\STAT 718 code\\mcp.csv")
mcp_var = mcp_var["x"]
###################                SCAD Variable output from R                    ###################
scad_var = pd.read_csv("C:\\Users\\Shijie Wang\\Desktop\\STAT 718 code\\scad.csv")
scad_var = scad_var["x"]
###################                SCAD Variable output from R                    ###################
lasso_var = pd.read_csv("C:\\Users\\Shijie Wang\\Desktop\\STAT 718 code\\lasso.csv")
lasso_var = lasso_var["x"]
###################                Feature Importance Previously(tree model)                    ###################
top_var = var_df['feature'][:50]
###################                Feature Importance Previously(30)                    ###################
top_var_30 = var_df['feature'][:30]


###################                Look at the previous result                ###################
###################                Boosting method might overfitting          ###################
col_names = ['Model: Accuracy DataFrame', 'Auc without variable Selection', 'Accuray without variable Selection', 'Auc with Feature Importance', 'Accuray with Feature Importance']
auc_score.columns = col_names
Accuray_df = auc_score.loc[: , ('Model: Accuracy DataFrame', 'Accuray without variable Selection', 'Accuray with Feature Importance')]
Accuray_df.columns = ['Model: Accuracy DataFrame', 'No Variable Selection', 'Feature Importance 50 vars']
Accuray_df

###################                Train Model with Lasso variable                ###################
rf, et, ada, xgb, lgb, gbdt, lr, svm, sgd, qda, lda, knn, dt, gp, mlp  = RandomForestClassifier(**rf_params), ExtraTreesClassifier(**et_params), AdaBoostClassifier(**ada_params), XGBClassifier(**xgb_params), lgb.LGBMClassifier(**lgb_params), GradientBoostingClassifier(**gbdt_params), LogisticRegression(**lr_params), SVC(**svm_params), SGDClassifier(**sgd_params), QuadraticDiscriminantAnalysis(**qda_params), LinearDiscriminantAnalysis(**lda_params), KNeighborsClassifier(**knn_params), DecisionTreeClassifier(**dt_params), GaussianProcessClassifier(**gp_params), MLPClassifier(**mlp_params)
# set the data set
train_data_select = train_data[top_var_30]
test_data_select = test_data[top_var_30]
# get the accuracy
accuracy_fi_50 = []
for classifier in [rf, et, ada, xgb, lgb, gbdt, lr, svm, sgd, qda, lda, knn, dt, gp, mlp]:
    #print('The current classifier is {:4f}'.format(str(classifier)))
    auc, accuracy = model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = classifier, iter_res = 0)
    accuracy_fi_50.append(accuracy)
# get the accuracy score
Accuray_df['Feature Importance 30 vars'] = accuracy_fi_50
Accuray_df

###################                Train Model with MCP variable                ###################
# set the data set
train_data_select = train_data[mcp_var]
test_data_select = test_data[mcp_var]
# get the accuracy
accuracy_mcp = []
for classifier in [rf, et, ada, xgb, lgb, gbdt, lr, svm, sgd, qda, lda, knn, dt, gp, mlp]:
    #print('The current classifier is {:4f}'.format(str(classifier)))
    auc, accuracy = model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = classifier, iter_res = 0)
    accuracy_mcp.append(accuracy)
# get the accuracy score
Accuray_df['MCP Variable Selection'] = accuracy_mcp
Accuray_df

###################                Train Model with Feature Importance 30               ###################

# set the data set
train_data_select = train_data[lasso_var]
test_data_select = test_data[lasso_var]
# get the accuracy
accuracy_lasso = []
for classifier in [rf, et, ada, xgb, lgb, gbdt, lr, svm, sgd, qda, lda, knn, dt, gp, mlp]:
    #print('The current classifier is {:4f}'.format(str(classifier)))
    auc, accuracy = model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = classifier, iter_res = 0)
    accuracy_lasso.append(accuracy)
# get the accuracy score
Accuray_df['Lasso Variable Selection'] = accuracy_lasso
Accuray_df

###################                Get the figure plot               ###################
Accuray_df_plot = pd.melt(Accuray_df, "Model: Accuracy DataFrame", var_name = "Method", value_name = "Value")
sns.set_style('darkgrid')
sns.set(font_scale = 2)
plt.figure(figsize = (28,24))
graph = sns.scatterplot(x = "Method", y = "Value", hue = "Model: Accuracy DataFrame", style = 'Model: Accuracy DataFrame', alpha = 1, s = 100, data = Accuray_df_plot.sort_values("Value", ascending = False), markers = ['o','o','o','o','o','o','o','o','o','o','o','o','o','o','o'])
graph.axhline(y = 0.7, color = 'r', alpha = 0.4, linestyle = '-')
#sns_plot = sns.lineplot(x = "Method", y = "Value", data = Accuray_df_plot[Accuray_df_plot["Value"] > 0.7].sort_values("Value", ascending = False), ax = graph)
#sns_plot = sns_plot.get_figure()
#sns_plot.savefig("C:/Users/Shijie Wang/Desktop/STAT 718 code/Team Project/accuracy.png")

###################                2.1 Examing model's with Boosting method                ###################
###################                use adabosst clf with base_estimator:                   ###################
ada_lr = AdaBoostClassifier(base_estimator = LogisticRegression(**lr_params), n_estimators = 1500, learning_rate = 0.01, random_state = seed)
ada_rf = AdaBoostClassifier(base_estimator = RandomForestClassifier(**rf_params), n_estimators = 1500, learning_rate = 0.01, random_state = seed)
ada_sgd = AdaBoostClassifier(base_estimator = SGDClassifier(**sgd_params), algorithm='SAMME', n_estimators = 1500, learning_rate = 0.01, random_state = seed)
ada_svc = AdaBoostClassifier(base_estimator = SVC(**svm_params), n_estimators = 100, learning_rate = 0.01, random_state = seed)
ada_et = AdaBoostClassifier(base_estimator = ExtraTreesClassifier(**et_params), n_estimators = 100, learning_rate = 0.01, random_state = seed)
ada_dt = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(**dt_params), n_estimators = 100, learning_rate = 0.01, random_state = seed)
ada_gbdt = AdaBoostClassifier(base_estimator = GradientBoostingClassifier(**gbdt_params), n_estimators = 100, learning_rate = 0.01, random_state = seed)
ada_lgb = AdaBoostClassifier(base_estimator = lgb.LGBMClassifier(**lgb_params), n_estimators = 100, learning_rate = 0.01, random_state = seed)

###################                Comparing Boosting method with no boosting method                ###################
###################                Result of *Featuure Importance* Variable Selection                 ###################
###################                set the train/test data                    ###################
train_data_select = train_data[top_var]
test_data_select = test_data[top_var]
###################                Model Result is shown below                 ###################
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_lr, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_sgd, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_svc, iter_res = 0) # good boosting: adaboost + SVM
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_et, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_dt, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_gbdt, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_lgb, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_rf, iter_res = 0)

###################                Result of *Featuure Importance 30 variables* Variable Selection                 ###################
###################                set the train/test data                    ###################
train_data_select = train_data[top_var_30]
test_data_select = test_data[top_var_30]
###################                Model Result is shown below                 ###################
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_lr, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_sgd, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_svc, iter_res = 0) # good boosting: adaboost + SVM
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_et, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_dt, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_gbdt, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_lgb, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_rf, iter_res = 0)

###################                Result of *Lasso Variable* Selection                 ###################
###################                set the train/test data                    ###################
train_data_select = train_data[lasso_var]
test_data_select = test_data[lasso_var]
###################                Model Result is shown below                 ###################
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_lr, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_sgd, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_svc, iter_res = 0) # good boosting: adaboost + SVM
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_et, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_dt, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_gbdt, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_lgb, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_rf, iter_res = 0)

###################                Result of *mcp_var Variable* Selection                 ###################
###################                set the train/test data                    ###################
train_data_select = train_data[mcp_var]
test_data_select = test_data[mcp_var]
###################                Model Result is shown below                 ###################
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_lr, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_sgd, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_svc, iter_res = 0) # good boosting: adaboost + SVM
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_et, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_dt, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_gbdt, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_lgb, iter_res = 0)
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = ada_rf, iter_res = 0)


#################################################################################################################################################
###########################################                2 - layers Stacking Method                 ###########################################
#####################################           1. base model: random forest + extra tree + GBDT + XGboost (n X 4)      ########################
#####################################           2. output: layer: LDA input: n X 4, train: y --- Ouput                  ########################
#################################################################################################################################################

def get_predictions(model, X):
	if hasattr(model, 'predict_proba'):
		pred = model.predict_proba(X)
	else:
		pred = model.predict(X)

	if len(pred.shape) == 1:  # for 1-d ouputs
			pred = pred[:,None]

	return pred

class StackedGeneralizer(object):

	def __init__(self, base_models = None, blending_model = None, n_folds = 5, verbose = True):
		self.base_models = base_models
		self.blending_model = blending_model
		self.n_folds = n_folds
		self.verbose = verbose
		self.base_models_cv = None

	def fit_base_models(self, X, y):
		if self.verbose:
			print('Fitting Base Models...')

		kf = list(KFold(y.shape[0], self.n_folds))

		self.base_models_cv = {}

		for i, model in enumerate(self.base_models):

			model_name = "model %02d: %s" % (i+1, model.__repr__())
			if self.verbose:
				print('Fitting %s' % model_name)

			# run stratified CV for each model
			self.base_models_cv[model_name] = []
			for j, (train_idx, test_idx) in enumerate(kf):
				if self.verbose:
					print('Fold %d' % (j + 1))

				X_train = X[train_idx]
				y_train = y[train_idx]

				model.fit(X_train, y_train)

				# add trained model to list of CV'd models
				self.base_models_cv[model_name].append(copy(model))

	def transform_base_models(self, X):
		# predict via model averaging
		predictions = []
		for key in sorted(self.base_models_cv.keys()):
			cv_predictions = None
			n_models = len(self.base_models_cv[key])
			for i, model in enumerate(self.base_models_cv[key]):
				model_predictions = get_predictions(model, X)

				if cv_predictions is None:
					cv_predictions = np.zeros((n_models, X.shape[0], model_predictions.shape[1]))

				cv_predictions[i,:,:] = model_predictions

			# perform model averaging and add to features
			predictions.append(cv_predictions.mean(0))

		# concat all features
		predictions = np.hstack(predictions)
		return predictions

	def fit_transform_base_models(self, X, y):
		self.fit_base_models(X, y)
		return self.transform_base_models(X)

	def fit_blending_model(self, X_blend, y):
		if self.verbose:
			model_name = "%s" % self.blending_model.__repr__()
			print('Fitting Blending Model:\n%s' % model_name)

		kf = list(KFold(y.shape[0], self.n_folds))
		# run  CV
		self.blending_model_cv = []

		for j, (train_idx, test_idx) in enumerate(kf):
			if self.verbose:
				print('Fold %d' % j)

			X_train = X_blend[train_idx]
			y_train = y[train_idx]

			model = copy(self.blending_model)

			model.fit(X_train, y_train)

			# add trained model to list of CV'd models
			self.blending_model_cv.append(model)

	def transform_blending_model(self, X_blend):

		# make predictions from averaged models
		predictions = []
		n_models = len(self.blending_model_cv)
		for i, model in enumerate(self.blending_model_cv):
			cv_predictions = None
			model_predictions = get_predictions(model, X_blend)

			if cv_predictions is None:
				cv_predictions = np.zeros((n_models, X_blend.shape[0], model_predictions.shape[1]))

			cv_predictions[i,:,:] = model_predictions

		# perform model averaging to get predictions
		predictions = cv_predictions.mean(0)
		return predictions

	def predict(self, X):
		# perform model averaging to get predictions
		X_blend = self.transform_base_models(X)
		predictions = self.transform_blending_model(X_blend)

		return predictions

	def fit(self, X, y):
		X_blend = self.fit_transform_base_models(X, y)
		self.fit_blending_model(X_blend, y)

	def evaluate(self, y, y_pred):
		print classification_report(y, y_pred)
		print 'Confusion Matrix:'
		print confusion_matrix(y, y_pred)
		return accuracy_score(y, y_pred)


#################################################################################################################################################
###########################################                2 - layers Stacking Method                 ###########################################
#####################################           1. base model: random forest + extra tree + GBDT + XGboost (n X 4)      ########################
#####################################           2. output: layer: LDA/xgb input: n X 4, train: y --- Ouput                  ########################
########################                          from sklearn.ensemble import StackingClassifier                       ########################
#################################################################################################################################################

###################                Base: random forest + extra tree + gradient boost tree                ###################
###################                2-layer: Gaussian Process Classifier                ###################
###################                Variable Selection: Feeture Importance                ###################
('rf', RandomForestClassifier(**rf_params)),
('sgd', SGDClassifier(**sgd_params)),
('gbdt', GradientBoostingClassifier(**gbdt_params)),
('lda', LinearDiscriminantAnalysis(**lda_params)),
('xgb', XGBClassifier(**xgb_params))
('gp', GaussianProcessClassifier(**gp_params))
# set the params 1
estimators_1 = [('rf', RandomForestClassifier(**rf_params)), ('sgd', SGDClassifier(**sgd_params)), ('gbdt', GradientBoostingClassifier(**gbdt_params)), ('lda', LinearDiscriminantAnalysis(**lda_params)), ('xgb', XGBClassifier(**xgb_params))]
clf_stack_1 = StackingClassifier(estimators = estimators_1, final_estimator = GaussianProcessClassifier(**gp_params), cv = 10, stack_method = 'auto')

# set the params 2
estimators_2 = [('gp', GaussianProcessClassifier(**gp_params)), ('sgd', SGDClassifier(**sgd_params)), ('gbdt', GradientBoostingClassifier(**gbdt_params)), ('lda', LinearDiscriminantAnalysis(**lda_params)), ('xgb', XGBClassifier(**xgb_params))]
clf_stack_2 = StackingClassifier(estimators = estimators_2, final_estimator = RandomForestClassifier(**rf_params), cv = 10, stack_method = 'auto')

# set the params 3
estimators_3 = [('rf', RandomForestClassifier(**rf_params)), ('gp', GaussianProcessClassifier(**gp_params)), ('gbdt', GradientBoostingClassifier(**gbdt_params)), ('lda', LinearDiscriminantAnalysis(**lda_params)), ('xgb', XGBClassifier(**xgb_params))]
clf_stack_3 = StackingClassifier(estimators = estimators_3, final_estimator = SGDClassifier(**sgd_params), cv = 10, stack_method = 'auto')

# set the params 4
estimators_4 = [('rf', RandomForestClassifier(**rf_params)), ('sgd', SGDClassifier(**sgd_params)), ('gp', GaussianProcessClassifier(**gp_params)), ('lda', LinearDiscriminantAnalysis(**lda_params)), ('xgb', XGBClassifier(**xgb_params))]
clf_stack_4 = StackingClassifier(estimators = estimators_4, final_estimator = GradientBoostingClassifier(**gbdt_params), cv = 10, stack_method = 'auto')

# set the params 5
estimators_5 = [('rf', RandomForestClassifier(**rf_params)), ('sgd', SGDClassifier(**sgd_params)), ('gbdt', GradientBoostingClassifier(**gbdt_params)), ('gp', GaussianProcessClassifier(**gp_params)), ('xgb', XGBClassifier(**xgb_params))]
clf_stack_5 = StackingClassifier(estimators = estimators_5, final_estimator = LinearDiscriminantAnalysis(**lda_params), cv = 10, stack_method = 'auto')

# set the params 6
estimators_6 = [('rf', RandomForestClassifier(**rf_params)), ('sgd', SGDClassifier(**sgd_params)), ('gbdt', GradientBoostingClassifier(**gbdt_params)), ('lda', LinearDiscriminantAnalysis(**lda_params)), ('gp', GaussianProcessClassifier(**gp_params))]
clf_stack_6 = StackingClassifier(estimators = estimators_6, final_estimator = XGBClassifier(**xgb_params), cv = 10, stack_method = 'auto')

# run the model:
for i, clf_i in enumerate([clf_stack_1, clf_stack_2, clf_stack_3, clf_stack_4, clf_stack_5, clf_stack_6]):
    print("The current clf number is: clf {}".format(i + 1))
    for j, vs in enumerate([top_var, top_var_30, mcp_var, lasso_var]):
        print('The Current Variable selection method is : variable selection method {}'.format(j + 1))
        train_data_select = train_data[vs]
        test_data_select = test_data[vs]
        model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = clf_i, iter_res = 0)

########################                          Final Model                      ########################
from sklearn.ensemble import BaggingClassifier, VotingClassifier
base = [('stack', clf_stack_5), ('boost', ada_svc), ('lda', lda)]
vote_clf = VotingClassifier(estimators = base, voting = 'hard')
########################                    Final Model on train data                     ########################
train_data_select = train_data[top_var]
test_data_select = test_data[top_var]
model_train(x = train_data_select, X_test = test_data_select, y = train_label, clf = bag_clf, iter_res = 1)


########################                          Final Accuray                     ########################
train_data_select = train_data[top_var]
test_data_select = test_data[top_var]
predictions = vote_clf.fit(train_data_select, train_label).predict(test_data_select)
accuracy_score(test_label, predictions)

#############################################################################################################
train = pd.read_csv('C:/Users/Shijie Wang/Desktop/python_learning/Dataset/overfitting/train.csv')
test = pd.read_csv('C:/Users/Shijie Wang/Desktop/python_learning/Dataset/overfitting/test.csv')
train = train.iloc[0:150,]
###################                Feature Engineering Analysis:                  ###################
train_data = train.drop(['Target_Practice', 'case_id'], axis = 1)
train_label = train['Target_Practice'].ravel()
test_data = test.drop(['Target_Practice', 'case_id'], axis = 1)
test_label = test['Target_Practice'].ravel()
feature = train_data.columns
