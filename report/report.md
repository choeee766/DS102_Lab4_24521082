

---

# Báo cáo Lab04: Triển khai Thuật toán Decision Tree và Random Forest từ đầu bằng NumPy

---

## 1. Giới thiệu

Trong lĩnh vực học máy, Decision Tree (Cây quyết định) và Random Forest (Rừng ngẫu nhiên) là hai thuật toán phân loại kinh điển nhờ cấu trúc logic dễ hiểu và hiệu quả cao. Thông thường, ta có thể dễ dàng gọi các mô hình này thông qua các thư viện có sẵn như scikit-learn. Tuy nhiên, để thực sự làm chủ thuật toán và hiểu rõ cách máy tính đưa ra quyết định ở bên dưới, bài thực hành này tập trung vào việc tự hiện thực lại toàn bộ cấu trúc logic của hai mô hình trên hoàn toàn bằng thư viện nền tảng NumPy.

Tập dữ liệu được lựa chọn để thử nghiệm là Wine Quality (Chất lượng rượu vang). Đây là một bài toán phân loại đa lớp thực tế nhưng gặp phải thách thức lớn về sự mất cân bằng giữa các nhóm chất lượng. Do đó, bài báo cáo này không chỉ trình bày cách xây dựng cấu trúc cây đệ quy hay kỹ thuật bốc mẫu ngẫu nhiên (Bagging) từ con số 0, mà còn tập trung vào quy trình tiền xử lý dữ liệu nghiêm ngặt và cách đánh giá mô hình một cách khách quan nhất bằng chỉ số F1-Score (Weighted). Sau cùng, kết quả thực nghiệm giữa hai bản code sẽ được so sánh để kiểm chứng độ chính xác.

**Mục tiêu chính:**

* Hiểu sâu cơ chế toán học, logic phân tách dữ liệu đệ quy dựa trên lý thuyết thông tin.
* Nắm vững kỹ thuật **Ensemble Learning** (Học kết hợp) thông qua cấu trúc Bagging của Random Forest.
* So sánh trực quan hiệu suất hiệu năng (độ đo F1-Score) giữa mô hình tự xây dựng và thư viện chuẩn `scikit-learn`.

---

## 2. Outline


```text
\Lab04
├─ data/
│  ├─ winequality-red.csv         # File dữ liệu rượu vang đỏ gốc
│  └─ winequality-white.csv       # File dữ liệu rượu vang trắng gốc
│  └─ winequality-combined.csv    

├─ model/
│  ├─ data_preprocessing.py       # Tiền xử lý dữ liệu
│  ├─ decision_tree.py            # Thuật toán Decision Tree bằng numpy
│  ├─ random_forest.py            # Thuật toán Random Forest bằng numpy
│  ├─ sklearn_models.py           # Gọi mô hình từ thư viện  Scikit-Learn 
│  └─ main.py                     
└─ report/
   └─ report.md                   

```



---

## 3. Các mô hình triển khai

### 3.1. Decision Tree 

Mô hình `DTClassifier` được xây dựng dựa trên tiêu chí **Entropy** để đo lường độ hỗn loạn thông tin và tìm ra ngưỡng chia tối ưu.

**Quy trình thực hiện:**

1. **Duyệt Feature:** Xem xét tập hợp các đặc trưng ngẫu nhiên được cấu hình để tìm khả năng phân loại tốt nhất.
2. **Thử ngưỡng (Threshold):** Trích xuất các giá trị độc nhất (`np.unique`) của từng đặc trưng số để làm các điểm cắt tiềm năng.
3. **Tính toán Information Gain:** Áp dụng công thức Entropy và độ lợi thông tin tại nút cha so với các nút con:

$$
H(y) = - \sum p_i \log_2(p_i)
$$

$$
IG = H(\text{parent}) - \left[
\frac{n_{\text{left}}}{n} H(\text{left})
+
\frac{n_{\text{right}}}{n} H(\text{right})
\right]
$$


4. **Tối ưu hóa:** Chọn cặp (Feature, Threshold) mang lại chỉ số **Information Gain (IG)** lớn nhất để tiến hành cắt nhánh.
5. **Đệ quy (grow_tree):** Lặp lại tiến trình phân tách cho các nút con cho đến khi thỏa mãn điều kiện dừng: đạt độ sâu tối đa (`max_depth`), số mẫu nhỏ hơn ngưỡng `min_samples_split`, hoặc nút hoàn toàn thuần nhất.

### 3.2. Random Forest 

Triển khai cấu trúc `RFClassifier` theo phương pháp **Bagging** để tối ưu hóa phương sai và hạn chế hiện tượng Overfitting.

**Kỹ thuật mấu chốt:**

* **Bootstrapping:** Với mỗi thực thể cây (`n_trees`), hệ thống tiến hành lấy mẫu ngẫu nhiên lặp lại có hoàn lại (`replace=True`) từ tập huấn luyện gốc với kích thước mẫu bằng $100\%$ tập nền.
- **Feature Subsampling:** Tại mỗi nút phân tách của cây con, số lượng đặc trưng đưa vào cân nhắc được giới hạn ngẫu nhiên theo tỷ lệ căn bậc hai `√n_features`, giúp tăng tính đa dạng sinh học giữa các cây.
* **Majority Voting:** Khi dự đoán tập dữ liệu mới, tất cả các cây con đồng loạt thực thi hành động `predict`, nhãn tối ưu sau cùng được quyết định bằng cơ chế bỏ phiếu đa số thông qua hàm `np.argmax(np.bincount())`.

---

## 4. Quy trình Tiền xử lý dữ liệu (Preprocessing)

Quy trình trong `data_preprocessing.py` được thiết kế chặt chẽ và nhất quán để chuẩn bị dữ liệu đầu vào sạch cho toàn bộ bài kiểm tra:

* **Tích hợp & Làm sạch:** Kết hợp hai tập dữ liệu gốc, bổ sung thuộc tính phân loại `is_red` ($1$ cho rượu đỏ, $0$ cho rượu trắng), loại bỏ toàn bộ các dòng chứa giá trị khuyết thiếu (`dropna`) cũng như các bản ghi trùng lặp (`drop_duplicates`).
* **Label Mapping (Gom nhóm nhãn):** Quy hoạch cấu trúc nhãn `quality` từ 7 bậc gốc về 3 nhóm phân lớp chiến lược:
* **Thấp (0):** Điểm chất lượng 3, 4, 5.
* **Trung bình (1):** Điểm chất lượng 6.
* **Cao (2):** Điểm chất lượng 7, 8, 9.


* **Stratified Train-Test Split:** Triển khai hàm `custom_train_test_split` chia tập dữ liệu theo tỷ lệ $80:20$. Hàm duyệt qua từng nhóm nhãn để bốc mẫu ngẫu nhiên, đảm bảo tỷ lệ phân phối các lớp đồng đều trên cả tập Train và Test nhằm tránh hiện tượng lệch phân lớp.
* **Standardization:** Xây dựng lớp `StandardScaler` để tính toán trung bình (`mean`) và độ lệch chuẩn (`std`) trên tập huấn luyện, thực hiện chuẩn hóa dữ liệu số về phân phối chuẩn nhằm nâng cao độ ổn định cho mô hình.
* **Tiến trình giám sát:** Tích hợp thanh trạng thái `tqdm` trực quan để theo dõi thời gian thực của 5 bước tiền xử lý cốt lõi.

---

## 5. Đánh giá & Kết quả

Kết quả thu được sau khi thực thi luồng kiểm tra tổng hợp từ `main.py` trên tập kiểm thử (Test set) sử dụng độ đo **F1-Score (Weighted)** để tối ưu khả năng đánh giá mất cân bằng:

| Chỉ số hình thái | DT NumPy (Bài 1) | RF NumPy (Bài 2) | DT Sklearn (Bài 3) | RF Sklearn (Bài 3) |
| --- | --- | --- | --- | --- |
| **Siêu tham số** | `max_depth=12` | `n_trees=15`, `depth=12` | `max_depth=8` | `n_estimators=5`, `depth=8` |
| **F1-Score (Weighted)** | **0.5482** | **0.5960** | **0.5447** | **0.5786** |


**Nhận xét:**

* **Độ chính xác thuật toán:** Mô hình `DT NumPy` tự viết đạt điểm số F1 tương đương, thậm chí nhỉnh hơn một chút so với phiên bản thư viện `DT Sklearn` nhờ cấu hình độ sâu linh hoạt hơn (`max_depth=12` so với `8`). Điều này chứng minh logic tính toán Entropy và đệ quy xây dựng cây hoàn toàn chính xác.
* **Sức mạnh của Ensemble Learning:** Đúng như lý thuyết, mô hình Random Forest ở cả hai phiên bản đều cho thấy sự vượt trội rõ rệt về hiệu suất (~4-5%) so với các cây quyết định đơn lẻ. Cơ chế Bagging đã chứng minh được vai trò giảm thiểu phương sai (Variance) hiệu quả trên tập dữ liệu Wine Quality.
* **Tính ổn định của cấu trúc Scratch:** Phiên bản `RF NumPy` tự viết (với 15 cây và độ sâu 12) cho kết quả F1-Score rất tốt (**0.5960**), vượt qua cấu hình cơ bản của Sklearn trong bài (5 cây, độ sâu 8), chứng tỏ mô hình tự phát triển có độ hoàn thiện cao và hoạt động ổn định trên các bài toán phân nhiều lớp.

