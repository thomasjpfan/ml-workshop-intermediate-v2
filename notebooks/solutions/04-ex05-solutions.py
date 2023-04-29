hist_perm_importance = permutation_importance(hist, X_test, y_test, n_repeats=5, random_state=0, n_jobs=2, scoring="neg_mean_absolute_error")

plot_permutation_importance(hist_perm_importance, feature_names, top_k=10);

top_importances_idx = np.argsort(hist_perm_importance["importances_mean"])[::-1]

top_4_features = feature_names[top_importances_idx[:4]]

top_4_features

PartialDependenceDisplay.from_estimator(
    hist,
    X_test,
    top_4_features,
    n_cols=2
);
