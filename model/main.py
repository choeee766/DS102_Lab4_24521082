from data_preprocessing import preprocess_data
from decision_tree import run_assignment_1
from random_forest import run_assignment_2
from sklearn_models import run_assignment_3

def main():
    
    red_file = "winequality-red.csv"
    white_file = "winequality-white.csv"
    
    X_train, X_test, y_train, y_test = preprocess_data(
        red_filepath=red_file, 
        white_filepath=white_file
    )
    
    f1_dt_numpy = run_assignment_1(X_train, X_test, y_train, y_test)
    f1_rf_numpy = run_assignment_2(X_train, X_test, y_train, y_test)
    f1_dt_sklearn, f1_rf_sklearn = run_assignment_3(X_train, X_test, y_train, y_test)
    
    print("BẢNG TỔNG KẾT KẾT QUẢ F1-SCORE (WEIGHTED)")
    print(f"Bài 1: Decision Tree (NumPy) : {f1_dt_numpy:.4f}")
    print(f"Bài 2: Random Forest (NumPy) : {f1_rf_numpy:.4f}")
    print(f"Bài 3: Decision Tree (Sklearn): {f1_dt_sklearn:.4f}")
    print(f"Bài 3: Random Forest (Sklearn): {f1_rf_sklearn:.4f}")

if __name__ == "__main__":
    main()