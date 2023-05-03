
hist_perm_results = permutation_importance(
    hist, X_test, y_test, n_repeats=5, random_state=0, scoring="neg_mean_absolute_error"
)

plot_permutation_importance(hist_perm_results, feature_names, top_k=10);

importances = hist_perm_results["importances_mean"]
sorted_idx = np.argsort(importances)

top_4_features = feature_names[sorted_idx[-4:]]

PartialDependenceDisplay.from_estimator(hist, X_test, features=top_4_features, n_cols=2);
