y

X.head()

X.isna().sum()

X.dtypes

categorical_features = X.select_dtypes(include="category").columns

numerical_features = X.select_dtypes(include="number").columns
