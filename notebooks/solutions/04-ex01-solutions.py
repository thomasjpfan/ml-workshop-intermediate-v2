
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)

y_predict_rf = rf.predict(X_test)

print(classification_report(y_test, y_predict_rf))
