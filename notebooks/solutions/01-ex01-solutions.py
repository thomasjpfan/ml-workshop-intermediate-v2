X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)

X_train.isna().sum()

X_train.shape

imputer = SimpleImputer(add_indicator=True)
imputer.set_output(transform="pandas")

X_trans = imputer.fit_transform(X_train)

X_trans.shape

X_trans.head()
