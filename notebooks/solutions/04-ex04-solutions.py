
hist = HistGradientBoostingRegressor(random_state=0)
hist.fit(X_train, y_train)

hist.score(X_test, y_test)

hist_perm_results = permutation_importance(
    hist, X_test, y_test, n_repeats=5, random_state=0
)

plot_permutation_importance(hist_perm_results, feature_names);
