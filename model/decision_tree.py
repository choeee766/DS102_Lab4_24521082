import numpy as np
from sklearn.metrics import f1_score

def entropy(y):
    y = np.array(y, dtype=int)
    if len(y) == 0: return 0
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        ''' Khởi tạo một nút trong cây quyết định'''
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        '''Kiểm tra xem nút hiện tại có phải là nút lá hay không'''
        return self.value is not None

class DTClassifier:
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None):
        '''Hàm khởi tạo thiết lập các siêu tham số cho mô hình:

        min_samples_split: Số lượng mẫu tối thiểu cần thiết tại một nút để tiếp tục phân nhánh

        max_depth: Chiều sâu tối đa cho phép của cây để tránh overfitting

        n_features: Số lượng đặc trưng ngẫu nhiên được cân nhắc khi tìm điểm chia'''
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        '''Hàm đệ quy xây dựng cây quyết định theo phương pháp từ trên xuống'''
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self.best_split(X, y, feat_idxs)

        if best_feat is None:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = self.split(X[:, best_feat], best_thresh)
        left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def best_split(self, X, y, feat_idxs):
        '''Tìm kiếm đặc trưng và ngưỡng phân chia tối ưu nhất từ danh sách các đặc trưng cho trước'''
        best_gain = -1
        split_idx, split_threshold = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thr in thresholds:
                gain = self.information_gain(y, X_column, thr)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr
        return split_idx, split_threshold

    def information_gain(self, y, X_column, threshold):
        '''Tính toán chỉ số information gain nếu phân chia tập dữ liệu theo một ngưỡng xác định'''
        parent_entropy = entropy(y)
        left_idxs, right_idxs = self.split(X_column, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0: return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def split(self, X_column, split_thresh):
        '''Phân chia các indices của dữ liệu thành hai nhóm dựa trên ngưỡng so sánh'''
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def most_common_label(self, y):
        '''Xác định nhãn xuất hiện nhiều nhất trong tập nhãn y hiện tại'''
        if len(y) == 0: return 0
        y = np.array(y, dtype=int)
        return np.argmax(np.bincount(y))

    def predict(self, X):
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        '''Hàm đệ quy duyệt cây cho một mẫu dữ liệu đơn lẻ x'''
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)

def run_assignment_1(X_train, X_test, y_train, y_test):
    """Hàm wrapper để gọi chạy bài 1"""
    clf = DTClassifier(max_depth=12, min_samples_split=5)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Kết quả bài 1 - F1 Score: {f1:.4f}\n")
    return f1