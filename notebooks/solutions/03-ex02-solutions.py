column_transformer.fit_transform(X_train).isna().sum()

hist = Pipeline([
    ("prep", column_transformer),
    ("hist", HistGradientBoostingClassifier(random_state=0))
])

hist.fit(X_train, y_train)

hist.score(X_test, y_test)

hist.get_params()

param_grid = {
    "hist__l2_regularization": [0, 0.1, 1, 10],
    "hist__max_bins": [32, 64, 128, 255],
}

halving_search = HalvingGridSearchCV(
    hist,
    param_grid,
    verbose=1,
    n_jobs=2,
)

halving_search.fit(X_train, y_train)

halving_search.best_params_

halving_search.score(X_test, y_test)
