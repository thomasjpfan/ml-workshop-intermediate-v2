fig, ax = plt.subplots(figsize=(6, 4))
RocCurveDisplay.from_estimator(log_reg, X_test, y_test, ax=ax, name="Logistic Regression")
RocCurveDisplay.from_estimator(rf, X_test, y_test, ax=ax, name="Random Forest")

dummy = DummyClassifier()
dummy.fit(X_train, y_train)

dummy.predict(X_test)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
RocCurveDisplay.from_estimator(dummy, X_test, y_test, ax=ax1)
PrecisionRecallDisplay.from_estimator(dummy, X_test, y_test, ax=ax2);
