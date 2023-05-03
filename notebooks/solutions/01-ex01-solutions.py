X.shape

X.isna().sum()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0
)

imputer = SimpleImputer(add_indicator=True)
imputer.set_output(transform="pandas")

X_train_imputed = imputer.fit_transform(X_train)

X_train_imputed.shape

X_train_imputed.head()
