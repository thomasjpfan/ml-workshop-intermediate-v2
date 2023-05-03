pipe = make_pipeline(
    StandardScaler(),
    KNNImputer(add_indicator=True),
    LogisticRegression()
)
pipe.set_output(transform="pandas")

pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)

log_reg = pipe[-1]

coef_series = pd.Series(
    log_reg.coef_.ravel(), index=log_reg.feature_names_in_
)

coef_series.sort_values().plot(kind="barh")
