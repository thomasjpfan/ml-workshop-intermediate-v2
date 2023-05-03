X_train, X_test, y_train, y_test = train_test_split( 
    X, y, stratify=y, random_state=0
)

preprocessor = ColumnTransformer([
    ("numerical", SimpleImputer(), numerical_features),
    ("categorical", OrdinalEncoder(encoded_missing_value=-1), categorical_features)
], verbose_feature_names_out=False)

pipe = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))

pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)

rf = pipe[-1]

rf.feature_names_in_

rf.feature_importances_

importances_series = pd.Series(rf.feature_importances_, index=rf.feature_names_in_).sort_values()

importances_series.plot(kind="barh")
