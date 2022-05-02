import sys
sys.path.append('../')
from bench.data import get_openml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit



for dataset_id in [42395, 1590, 41138]:  # 1471, 1502, 40922, 43551, 1461]:
    #try:
        X, y, transformer = get_openml(dataset_id)
        print('transofrmer', transformer)
        estimators = [
            ('RF', RandomForestClassifier(), {'n_estimators':[10, 20, 50, 100], 'max_depth':[None, 8, 32]}),
            ('GBDT', GradientBoostingClassifier(), {'n_estimators':[10, 20, 50, 100], 'max_depth':[3, 8, 16]}),
            ('MLP', MLPClassifier(max_iter=5000), {'hidden_layer_sizes':[(100,), (32, 32)], 'solver':['adam', 'sgd'], 'alpha':[1e-4, 1e-3]}),
        ]

        # print(X, y)

        # Realistic setting: in our experiments, we will take test = 20% and maxmum labeling is 10%
        # Therefore we take 10% of strain and 20% for test. Also, some datasets have imbalanced labels,
        # so we take stratified shuffle split.

        sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        for i, (ind_learn, ind_test) in enumerate(sss.split(X, y)):
            isss = StratifiedShuffleSplit(n_splits=1, train_size=0.125, random_state=i)
            X_learn_raw, X_test_raw = X[ind_learn], X[ind_test]
            y_learn, y_test = y[ind_learn], y[ind_test]
            
            best_model = None
            best_roc = None
            best_params = None
        
            for ind_train, _ in isss.split(X_learn_raw, y_learn):
                X_train_raw = X_learn_raw[ind_train]
                y_train = y_learn[ind_train]

                X_train = transformer.fit_transform(X_train_raw)
                X_test = transformer.transform(X_test_raw)
                
                for name, estimator, param_grid in estimators:
                    model = GridSearchCV(estimator, param_grid, scoring='roc_auc_ovr')
                    model.fit(X_train, y_train)
                    roc = model.score(X_test, y_test)
                    if best_model is None or roc > best_roc:
                        best_model = name
                        best_params = model.best_params_
                        best_roc = roc
            
            print('Dataset {} Iter {} Model {} Params {} ROC {}'.format(dataset_id, i, best_model, best_params, best_roc))

    #except Exception as e:
    #    print("{} error {}".format(dataset_id, e))
