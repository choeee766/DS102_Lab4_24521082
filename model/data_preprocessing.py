import pandas as pd
import numpy as np
from tqdm import tqdm

class StandardScaler:
    
    def __init__(self):
        self.mean = None
        self.std = None

    def fit_transform(self, X):
        X_arr = np.array(X, dtype=float)
        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0)
        
        self.std[self.std == 0] = 1.0
        return (X_arr - self.mean) / self.std

    def transform(self, X):
        X_arr = np.array(X, dtype=float)
        return (X_arr - self.mean) / self.std


def custom_train_test_split(X, y, test_size=0.2, random_state=42):
    
    np.random.seed(random_state)
    
    df_combined = X.copy()
    df_combined['_target_y'] = y.values if hasattr(y, 'values') else y
    
    train_list = []
    test_list = []
    
    for label in np.unique(df_combined['_target_y']):
        df_class = df_combined[df_combined['_target_y'] == label]
        
        shuffled_indices = df_class.index.to_numpy().copy()
        np.random.shuffle(shuffled_indices)
        df_class_shuffled = df_class.loc[shuffled_indices]
        
        n_test = int(len(df_class_shuffled) * test_size)
        
        test_list.append(df_class_shuffled.iloc[:n_test])
        train_list.append(df_class_shuffled.iloc[n_test:])
        
    df_train = pd.concat(train_list).sample(frac=1, random_state=random_state)
    df_test = pd.concat(test_list).sample(frac=1, random_state=random_state)
    
    X_train = df_train.drop(columns='_target_y')
    y_train = df_train['_target_y']
    X_test = df_test.drop(columns='_target_y')
    y_test = df_test['_target_y']
    
    return X_train, X_test, y_train, y_test


def preprocess_data(red_filepath="winequality-red.csv", 
                    white_filepath="winequality-white.csv", 
                    test_size=0.2, 
                    random_state=42):
 
    pbar = tqdm(total=5, desc="Tiến trình xử lý dữ liệu", unit="bước")

    df_red = pd.read_csv(red_filepath, sep=';')
    df_red['is_red'] = 1  # 1 là đỏ
    
    df_white = pd.read_csv(white_filepath, sep=';')
    df_white['is_red'] = 0  # 0 là trắng
    pbar.update(1)

    # Gộp và Làm sạch 
    df = pd.concat([df_red, df_white], ignore_index=True)
    df = df.dropna()
    df = df.drop_duplicates(ignore_index=True)
    pbar.update(1)

    # Gom nhóm nhãn 
    df['quality'] = df['quality'].map({
        3: 0, 4: 0, 5: 0,  # Thấp
        6: 1,              # Trung bình
        7: 2, 8: 2, 9: 2   # Cao
    }).fillna(1).astype(int) 
    
    X = df.drop(columns='quality')
    y = df['quality']
    pbar.update(1)
    df.to_csv("winequality-combined.csv", sep=";", index=False)

    # Chia tập train / test 
    X_train, X_test, y_train, y_test = custom_train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    pbar.update(1)

    scaler = StandardScaler()
    num_cols = [col for col in X.columns if col != 'is_red']
    
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    X_train_np = X_train.values
    X_test_np = X_test.values
    y_train_np = y_train.values
    y_test_np = y_test.values
    
    pbar.update(1)
    pbar.close()
    
    print(f"Kích thước Train: {X_train_np.shape}, Test: {X_test_np.shape}")
    
    return X_train_np, X_test_np, y_train_np, y_test_np