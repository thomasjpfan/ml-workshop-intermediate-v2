
search_cv = RandomizedSearchCV(
    RandomForestClassifier(random_state=0),
    param_distributions=param_dist,
    n_iter=20,
    random_state=0,
    n_jobs=2,
)

search_cv.fit(X_train, y_train)

search_cv.best_params_

search_cv.score(X_test, y_test)

halving_cv = HalvingRandomSearchCV(
    RandomForestClassifier(random_state=0),
    param_distributions=param_dist,
    verbose=1,
    n_jobs=2,
)

halving_cv.fit(X_train, y_train)

halving_cv.best_params_

halving_cv.score(X_test, y_test)
