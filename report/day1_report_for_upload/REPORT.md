# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy: 06/11/2026

**Runtime Colab:** CPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=… ; `class_name`=… ; `rank`=1 ; `score`=… ; `taxonomy_name`=…
    class_id = 468; class_name = cab; rank = 1; score = 0.510915; taxonomy_name = ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    Nói về: Cả bức ảnh. Lý do: Đây là bài toán Phân loại ảnh (Image Classification). Tác vụ này dự đoán nhãn tổng thể đại diện cho toàn bộ khung hình, chứ không đưa ra tọa độ (bounding box) hay phân vùng từng vật thể thể riêng biệt như tác vụ Object Detection hay Instance Segmentation.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   - Danh sách tên lớp đi kèm với taxonomy/checkpoint của tập dữ liệu huấn luyện (ở đây là bộ chuẩn ImageNet-1K).
   - Mô hình chỉ phân loại dựa trên tập nhãn cố định đã được định nghĩa từ trước khi huấn luyện (468 tương ứng với "cab"), không tự nghĩ ra tên lớp mới. 
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    Đảm bảo tính chính xác và đồng bộ:
    - class_id giúp tránh nhầm lẫn do lỗi gõ chữ hoặc sự khác biệt về ngôn ngữ.
    - class_name giúp con người đọc và hiểu trực quan ngay lập tức.
    - taxonomy_name xác định không gian nhãn (context/schema). Cùng một tên lớp hoặc ID có thể mang ý nghĩa khác nhau ở các taxonomy khác nhau (ví dụ: COCO vs          ImageNet). Ghi đủ cả 3 giúp kết quả truy xuất nguyên bản và không bị sai lệch khi tích hợp hệ thống.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Guideline cần quy định rõ các tiêu chí ưu tiên để người gán nhãn đưa ra quyết định nhất quán:
     - Chủ đề chính: Vật thể chiếm vị trí trung tâm hoặc chiếm diện tích lớn nhất trong bức ảnh.
     - Quy tắc ưu tiên: Quy định rõ đối tượng nào được ưu tiên nhãn (ví dụ: Con người > Phương tiện > Cảnh nền).
     - Bối cảnh tổng thể: Chọn nhãn đại diện cho toàn bộ khung cảnh thay vì một chi tiết phụ (ví dụ: nếu ảnh chụp tắc đường có nhiều xe thì chọn nhãn chung là "traffic / street" thay vì nhãn xe lẻ).
6. Vì sao không được lấy `score` của model làm ground truth?
    - Model score chỉ là xác suất thống kê: score thể hiện độ tin cậy nội tại của mô hình dựa trên dữ liệu đã học, không phản ánh sự thật tuyệt đối.
    - Nguy cơ lỗi: Mô hình có thể dự đoán sai với độ tin cậy rất cao (Overconfidence) hoặc bị nhiễu do góc chụp/ánh sáng.
    - Bản chất của Ground Truth: Ground Truth là thực tế khách quan (thường do con người thẩm định và xác nhận). Lấy đầu ra của mô hình làm Ground Truth sẽ dẫn đến hiện tượng trôi lệch dữ liệu và tự củng cố lỗi sai của mô hình.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=… ; `score`=… ; `bbox_xyxy`=[…, …, …, …] ; `bbox_width`=… ; `bbox_height`=…
    `class_name`= person ; `score`= 0.912625 ; `bbox_xyxy`= [385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    Vùng bao quanh vật thể (person) bắt đầu từ điểm góc trên-trái tại tọa độ pixel (385.33, 69.24) và kéo dài đến điểm góc dưới-phải tại tọa độ pixel (498.92, 348.92). Khung này có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    - Ngưỡng 0.20: Có 17 prediction.
    - Ngưỡng 0.60: Có 6 prediction.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    Khi hạ ngưỡng (từ 0.60 xuống 0.20), số lượng hộp phát hiện tăng mạnh dẫn đến nhiều Dương tính giả (False Positives) và hộp nhiễu/trùng lặp. Người reviewer phải làm thêm:
     - Lọc và xoá: Xóa bỏ các hộp dự đoán sai (như nhận diện nhầm chi tiết nền thành vật thể).
     - Khử trùng lặp: Xóa các hộp bị vẽ đè lên cùng một vật thể.
     - Chỉnh sửa viền (Refine): Điều chỉnh lại các góc x, y của hộp cho ôm sát vật thể thực tế hơn.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    Quy tắc Bounding Box ôm sát (Tight Boundary Rule): Tất cả các khung bao (Bounding Box) phải được vẽ ôm sát tối đa vào các mép ngoài cùng của vật thể. Không được bỏ sót các phần thuộc vật thể (trừ phần bị che hoàn toàn) và không để thừa phần không khí/nền xung quanh quá 2-3 pixel.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    - Vật thể chỉ ra trên ảnh kitchen: Chiếc lò nướng bên trái (oven) bị cắt mất cạnh bên trái do rìa ảnh, hoặc các chiếc bát (bowl) bị các dụng cụ bếp khác che khuất một phần.
    - Hướng xử lý:
     - Viết thành quy tắc trong Guideline: Xử lý bằng quy tắc chuẩn, không cần escalation.
     - Giải thích: Hiện tượng vật thể bị cắt bởi mép ảnh hoặc bị che khuất một phần là các trường hợp cực kỳ phổ biến trong thị giác máy tính. Guideline tiêu chuẩn luôn có sẵn quy tắc cố định (ví dụ: "Nếu vật thể bị cắt ở mép ảnh, vẽ khung bao trùm toàn bộ phần hiển thị trong ảnh"). Việc đệ trình lên Lab Coach chỉ dành cho các trường hợp không thể nhận diện được bản chất vật thể hoặc mâu thuẫn tiêu chuẩn gán nhãn chưa từng được quy định.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=… ; `class_name`=… ; `score`=… ; số điểm polygon=`polygon_point_count`=… ; 3–5 điểm đầu của `polygon_xy`=…
     `instance_id`= kitchen-008 ; `class_name`= spoon ; `score`= 0.487577 ; số điểm polygon=`polygon_point_count`= 46 ; 3–5 điểm đầu của `polygon_xy`= [[544.0, 67.0], [544.0, 80.0], [545.0, 81.0], [545.0, 91.0], [544.0, 92.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    So với Bounding Box (chỉ cho khung chữ nhật bao quanh toàn bộ vùng ngoài), Polygon cung cấp:
    - Đường biên thực tế (Exact Contour): Biểu diễn chính xác hình dạng hình học thực sự của vật thể (đường cong, góc khuyết, độ nghiêng).
    - Loại bỏ vùng nền thừa (Background Exclusion): Phân biệt chính xác từng pixel thuộc vật thể với pixel thuộc về nền xung quanh hoặc thuộc vật thể khác nằm đè lên.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    - Mục đích: instance_id được dùng để phân biệt và định danh duy nhất cho từng đối tượng riêng biệt trong phạm vi tập dữ liệu output của bài lab.
    - Cách hiểu: 
     - Không phải Class ID: instance_id không đại diện cho phân loại/chủng loại vật thể (như spoon hay person), mà các vật thể cùng lớp vẫn có instance_id khác nhau.
     - Không phải Tracking ID: instance_id chỉ có giá trị cục bộ trên một bức ảnh tĩnh duy nhất. Nó không dùng để theo dõi sự di chuyển của vật thể qua chuỗi các frames liên tiếp trong video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    Quy tắc vẽ Mask sát biên (Exact Outline Rule): Mọi đường biên đa giác (polygon) phải chấm các đỉnh bám sát theo đúng viền mép ranh giới của vật thể. Không vẽ chờm ra ngoài vùng nền quá 1 pixel và không cắt lẹm vào phần thịt bên trong của vật thể.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    - Vật thể/Vị trí trên ảnh kitchen: Các chiếc bát (bowl) chồng đè lên bề mặt bàn (dining table), hoặc khu vực người (person) che khuất một phần chiếc lò/kệ phía sau, hoặc chiếc thìa/dụng cụ bếp (spoon 0.49) treo bị biên mờ sát với dụng cụ bên cạnh.
    - Dùng Guideline hay Escalation: Sử dụng Guideline. 
     Giải thích: Các trường hợp vật thể đè/che khuất lẫn nhau hoặc chạm nhau trên cùng một bề mặt là biểu hiện bình thường và phổ biến trong thực tế. Guideline gán nhãn luôn có sẵn quy tắc xử lý tiêu chuẩn (ví dụ: "Khi vật thể A che một phần vật thể B, vẽ mask tách biệt cho phần nhìn thấy được của từng vật thể và gán cho chúng các instance_id khác nhau"). Escalation chỉ cần thiết khi hai vật thể dính liền đến mức mắt thường không thể phân biệt ranh giới hoặc phân loại lớp.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Nhãn đơn (single class label) cho toàn bộ bức ảnh (ví dụ: class_id = 468, class_name = cab).| Khó chọn 1 nhãn duy nhất khi ảnh có nhiều chủ đề (ví dụ: ảnh traffic có cả xe buýt, xe con, người và cảnh đường phố). | Xem toàn bộ ảnh, xác định đối tượng/chủ đề chính theo guideline và gán đúng 1 nhãn duy nhất đại diện cho ảnh.| Kiểm tra xem nhãn được chọn có tuân thủ đúng quy tắc ưu tiên trong guideline không (ví dụ: ưu tiên vật thể trung tâm hay bối cảnh). |
| Phát hiện vật thể | Tọa độ khung chữ nhật ôm sát từng vật thể kèm nhãn lớp: [x_min, y_min, x_max, y_max] theo pixel (ví dụ: bbox_xyxy của person). |Dương tính giả khi nhận diện nhầm nền/đồ vật thành người (person score 0.61 ở góc dưới trái bếp), các vật thể nhỏ/bị che khuất (như chiếc bát bowl bị đè).|Vẽ khung chữ nhật (bounding box) ôm sát viền ngoài của từng đối tượng và chọn đúng nhãn lớp tương ứng.|Kiểm tra độ ôm sát của khung (không quá rộng/hẹp), đảm bảo không bỏ sót vật thể thực tế và xóa bỏ các hộp nhận diện sai (nhiễu).|
| Instance segmentation |Tập hợp các điểm đa giác bám sát đường biên thực tế cho từng instance: polygon_xy kèm instance_id riêng biệt (ví dụ: kitchen-008 cho spoon).|Biên mask của mô hình còn gập gềnh (jagged edges), nhận diện sai vùng lớn (dining table phủ đè nhiều vật), vùng giao giữa các vật dính liền/mờ.|Chấm các điểm đa giác (polygon) bám sát đường viền chính xác của từng đối tượng, tách riêng các vật thể chạm/chồng lên nhau.|Kiểm tra độ mịn và độ chính xác của đường biên mask, đảm bảo không chấm chờm ra nền và các instance được tách biệt đúng quy chuẩn.|

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    Quy tắc bảo vệ dữ liệu: Chỉ sử dụng các tập dữ liệu công khai được chỉ định (như COCO 2017) cho mục đích học tập/lab và tuyệt đối không tải lên hay chia sẻ dữ liệu riêng tư, nhạy cảm cá nhân ra bên ngoài.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Xử lý khi gặp dữ liệu sai phạm vi (Out-of-scope): Bạn sẽ ngay lập tức dừng xử lý dữ liệu và báo cáo cho Lab Coach (hoặc Data Lead / Trưởng dự án) để nhận hướng dẫn giải quyết.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
