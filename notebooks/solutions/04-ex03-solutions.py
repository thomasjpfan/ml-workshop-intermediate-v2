y_proba_rf = rf.predict_proba(X_test)

roc_auc_score(y_test, y_proba_rf[:, 1])

svc = SVC(random_state=0)
svc.fit(X_train, y_train)

y_decision_svc = svc.decision_function(X_test)

average_precision_score(y_test, y_decision_svc)
