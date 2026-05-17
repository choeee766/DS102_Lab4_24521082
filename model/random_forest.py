import numpy as np
from sklearn.metrics import f1_score
from decision_tree import DTClassifier
class RFClassifier:
    def __init__(self, n_trees=5, max_depth=8, min_samples_split=5, n_features=None):
        '''Hàm khởi tạo thiết lập các siêu tham số:
        n_trees: Số lượng cây quyết định cấu thành mô hình ensemble
        max_depth: Độ sâu tối đa cho phép của mỗi cây quyết định con nhằm kiểm soát cấu trúc và hạn chế overfitting
        min_samples_split: Số lượng mẫu tối thiểu yêu cầu tại một nút để tiếp tục thực hiện phân tách nhánh
        n_features: Số lượng features tối đa được chọn ngẫu nhiên để đánh giá điểm phân chia tối ưu tại mỗi nút của cây con
        self.trees: Danh sách lưu trữ các thực thể cây quyết định sau khi hoàn thành tiến trình huấn luyện'''
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        n_samples, n_feats = X.shape
        
        if self.n_features is None:
            self.n_features = int(np.sqrt(n_feats))

        for _ in range(self.n_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]
            
            # Khởi tạo và huấn luyện cây con
            tree = DTClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_features=self.n_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees]).T
        
        predictions = []
        for preds in tree_preds:
            preds = np.array(preds, dtype=int)
            predictions.append(np.argmax(np.bincount(preds)))
        return np.array(predictions)

def run_assignment_2(X_train, X_test, y_train, y_test):
    """Hàm wrapper để gọi chạy Bài 2."""
    clf = RFClassifier(n_trees=15, max_depth=12, min_samples_split=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Kết quả Bài 2 - F1 Score: {f1:.4f}\n")
    return f1