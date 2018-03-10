from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from sklearn import svm
from sklearn.grid_search import GridSearchCV

def classifier_testing():
    return 

def tune_rf(train_x, train_y, seed):
    # finds best parameters for random forest through grid search. Only outputs the params
    rfc = RandomForestClassifier(n_jobs=-1, oob_score=True, random_state=seed) 

    params = {
        'n_estimators': [9, 18, 27, 36, 45],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [5, 10, 15, 20, 25],
        'min_samples_leaf': [1, 2, 4]}
    # {'max_features': 'auto', 'n_estimators': 45, 'max_depth': 25, 'min_samples_leaf': 2}
    CV_rfc = GridSearchCV(estimator=rfc, scoring='f1', param_grid=params, cv=10)
    CV_rfc.fit(train_x, train_y)
    print CV_rfc.best_params_

def tune_SVC(train_x, train_y):
    svc = svm.SVC(kernel='rbf')
    # https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
    params = {
        'C': [2**(-5), 2**(-3), 2**(-1), 2**0, 2**2, 2**4, 2**6],
        'gamma': [2**(-7), 2**(-5), 2**(-3), 2**(-1), 2**1, 2**3]}
    
    CV_svc = GridSearchCV(estimator=svc, param_grid=params, cv=10)
    CV_svc.fit(train_x, train_y)
    print CV_svc.best_params_