search_cv = RandomizedSearchCV(
    RandomForestClassifier(random_state=0),
    param_distributions=param_dist,
    n_iter=20,
    verbose=1,
    n_jobs=2,
    random_state=0,
)

search_cv.fit(X_train, y_train)

search_cv.best_params_

search_cv.score(X_test, y_test)

halfing_cv = HalvingRandomSearchCV(
    RandomForestClassifier(random_state=0), 
    param_distributions=param_dist,
    verbose=1,
    n_jobs=2,
    random_state=0
)

halfing_cv.fit(X_train, y_train)

halfing_cv.best_params_

halfing_cv.score(X_test, y_test)
