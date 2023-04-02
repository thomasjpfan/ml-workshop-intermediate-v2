X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, stratify=y
)

ct = ColumnTransformer([
    ("numerical", SimpleImputer(), numerical_features),
    ("categorical", OrdinalEncoder(encoded_missing_value=-1), categorical_features)
], verbose_feature_names_out=False)

ct.fit_transform(X_train)

pipe = Pipeline([
    ("preprocessor", ct),
    ("rf", RandomForestClassifier(random_state=0))
])

pipe.fit(X_train, y_train)

pipe.score(X_test, y_test)
