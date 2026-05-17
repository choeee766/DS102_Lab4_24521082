from sklearn.tree import DecisionTreeClassifier as SklearnTree
from sklearn.ensemble import RandomForestClassifier as SklearnForest
from sklearn.metrics import f1_score

def run_assignment_3(X_train, X_test, y_train, y_test):
    """Hàm huấn luyện và đánh giá mô hình bằng thư viện Scikit-Learn."""
    
    # Decision Tree thư viện
    dt_clf = SklearnTree(max_depth=8, min_samples_split=5, random_state=42)
    dt_clf.fit(X_train, y_train)
    dt_pred = dt_clf.predict(X_test)
    dt_f1 = f1_score(y_test, dt_pred, average='weighted')
    print(f"Sklearn Decision Tree F1 Score: {dt_f1:.4f}")
    
    # Random Forest thư viện
    rf_clf = SklearnForest(n_estimators=5, max_depth=8, min_samples_split=5, random_state=42)
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_test)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    print(f"Sklearn Random Forest F1 Score: {rf_f1:.4f}\n")
    
    return dt_f1, rf_f1