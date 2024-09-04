# ** Temporal Recommendation **

## ** Giới thiệu **

## ** Cài đặt **

## ** Sử dụng **

### ** Cấu trúc thư mục trong trên server bên DAC **
```bash
temporal_recommendation
├── aug_data							# Lưu các file data augmentation
|	└── tv360/
|		└── *.txt
├── data/
|	└── tv360/
|		├── imap.json 					# Lưu dữ liệu map id item dưới dạng số tuần tự
|		├── umap.json					# Lưu dữ liệu map user id dưới dạng số tuần tự
|		├── train.txt					# Lưu dữ liệu train đã được tiền xử lý
|		├── test.txt					# Lưu dữ liệu test đã được tiền xử lý
|		├── valid.txt					# Lưu dữ liệu valid đã được tiền xử lý
|		├── train_reverse.txt				# Lưu dữ liệu train đã được xử lý qua module đào tạo trước đảo ngược (Reversely Pre-training)
|		├── test_reverse.txt				# Lưu dữ liệu test đã được xử lý qua module đào tạo trước đảo ngược (Reversely Pre-training)
|		└── valid_reverse.txt				# Lưu dữ liệu valid đã được xử lý qua module đào tạo trước đảo ngược (Reversely Pre-training)
├── log/
|	└── *.log						# Log các file chạy
├── pipeline/
|	├── 
|
|
|
|


```
