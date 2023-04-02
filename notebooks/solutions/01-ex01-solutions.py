
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=0, stratify=y
)

X_train.isna().sum()

imputer = SimpleImputer(add_indicator=True)
imputer.set_output(transform="pandas")

imputer.fit_transform(X_train)
