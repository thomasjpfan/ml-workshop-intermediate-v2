
X_transformed = column_transformer.fit_transform(X_train)

X_transformed.head()

X_transformed.isna().sum()

pipe = Pipeline([
    ("prep", column_transformer),
    ("hist", HistGradientBoostingClassifier(random_state=0)),
])

pipe.get_params()

param_grid = {
    "hist__l2_regularization": [0.01, 0.1, 1, 10],
    "hist__max_bins": [32, 64, 128, 255],
}

halving_search_cv = HalvingGridSearchCV(
    pipe,
    param_grid=param_grid,
    verbose=1,
    n_jobs=-1
)

halving_search_cv.fit(X_train, y_train)

halving_search_cv.best_params_

halving_search_cv.score(X_test, y_test)
