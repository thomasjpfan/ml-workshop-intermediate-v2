log_reg = make_pipeline(
    StandardScaler(),
    KNNImputer(add_indicator=True),
    LogisticRegression()
)
log_reg.set_output(transform="pandas")
log_reg.fit(X_train, y_train)

coef = log_reg[-1].coef_.ravel()
feature_names = log_reg[-1].feature_names_in_

coef_series = pd.Series(coef, index=feature_names)

coef_series.sort_values().plot(kind="barh");
