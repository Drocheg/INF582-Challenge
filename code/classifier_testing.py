from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV

def classifier_testing():
    return 

def tune_rf(train_x, train_y, seed):
    # finds best parameters for random forest through grid search. Only outputs the params
    # Result:
    # {'max_features': 'auto', 'n_estimators': 45, 'max_depth': 25, 'min_samples_leaf': 2}

    rfc = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=seed) 

    params = {
        'n_estimators': [9, 18, 27, 36, 45],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [5, 10, 15, 20, 25],
        'min_samples_leaf': [1, 2, 4]}
    CV_rfc = GridSearchCV(estimator=rfc, scoring='f1', param_grid=params, cv=10)
    CV_rfc.fit(train_x, train_y)
    print CV_rfc.best_params_

def tune_SVC(train_x, train_y):
    # https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    svc = svm.SVC(kernel='rbf')
    params = {
        'C': [2**(-3), 2**(-1), 2**0, 2**2, 2**4, 2**6],
        'gamma': [2**(-7), 2**(-5), 2**(-3), 2**(-1), 2**1]}
    # {'C': 16, 'gamma': 0.125}
    CV_svc = GridSearchCV(estimator=svc, param_grid=params, cv=6, n_jobs=-1, verbose=100)
    CV_svc.fit(train_x, train_y)
    print CV_svc.best_params_

def tune_lgbm(train_x, train_y):
    lgb = LGBMClassifier()
    # https://www.analyticsvidhya.com/blog/2017/06/which-algorithm-takes-the-crown-light-gbm-vs-xgboost/
    # https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
    params = {
    'num_leaves': [10, 31, 127],
    'min_data_in_leaf': [4,8,16,32,64],
    'max_depth': [4,8,16,32,64],
    'reg_alpha': [0, 0.1, 0.5]}
    # {'num_leaves': 127, 'reg_alpha': 0.5, 'max_depth': 8, 'min_data_in_leaf': 16}
    CV_lgb = GridSearchCV(estimator=lgb, param_grid=params, cv=6, n_jobs=-1, verbose=100)
    CV_lgb.fit(train_x, train_y)
    print CV_lgb.best_params_