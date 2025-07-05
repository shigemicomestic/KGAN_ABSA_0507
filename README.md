### Env
* python 3.8.0
* model_weight/temp

### Tạo môi trường ảo
* python -m venv env
* env\Scripts\activate (cmd)
* source env/Scripts/activate (Git Bash)

### Cài đặt thư viện
* python -m pip install --upgrade pip
* pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html  (with CPU)
* pip install -r requirements.txt --no-deps
* python -m spacy download en_core_web_lg --no-deps

### Thoát khỏi môi trường ảo (khi không dùng nữa)
* deactivate

## Training options

- **ds_name**: tên của tập dữ liệu mục tiêu, ['14semeval_laptop','14semeval_rest','15semeval_rest','16semeval_rest','Twitter'], default='14semeval_rest'
- **bs**: kích thước lô để sử dụng trong quá trình đào tạo, [32, 64], default=64
- **learning_rate**: tốc độ học để sử dụng, [0,001, 0,0005], default=0,001
- **dropout_rate**: tỷ lệ bỏ qua đối với các đặc điểm tình cảm, [0,1, 0,3, 0,5], default=0,05
- **n_epoch**: số kỷ nguyên để sử dụng, default=20
- **model**: tên của mô hình, default='KGNN'
- **dim_w**: chiều của nhúng từ, default=300
- **dim_k**: chiều của nhúng đồ thị, [200,400], mặc định=200
- **is_test**: đào tạo hoặc kiểm tra mô hình, [0, 1], mặc định=1
- **is_bert**: dựa trên GloVe hoặc dựa trên BERT, [0, 1], mặc định=0


## Running

#### training based on GloVe: 

* python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.001 -dropout_rate 0.5 -n_epoch 20 -model KGNN -dim_w 300 -dim_k 400 -kge analogy  -gcn 0  -is_test 0 -is_bert 0
* python -m main_total -ds_name 14semeval_rest -bs 64 -learning_rate 0.001 -dropout_rate 0.5 -n_epoch 20 -model KGNN -dim_w 300 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 0

#### training based on BERT: 

* python -m main_total -ds_name 14semeval_laptop -bs 32 -learning_rate 0.00003 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 400 -kge analogy -gcn 0  -is_test 0 -is_bert 1
* python -m main_total -ds_name 14semeval_rest -bs 64 -learning_rate 0.00003 -n_epoch 20 -model KGNN -dim_w 768 -dim_k 200 -kge distmult -gcn 0 -is_test 0 -is_bert 1

The detailed training scripts can be found in "./scripts".

## Evaluation

To have a quick look, we saved the best model weight trained on the evaluated datasets in the "./model_weight/best_model_weight". You can easily load them and test the performance. You can evaluate the model weight with:

python -m main_total -ds_name 14semeval_laptop -bs 32 -model KGNN -dim_w 300 -dim_k 200 -is_test 1
python predict.py --model_path ./model_weight/best_model_weight/KGNN_14semeval_laptop_78.91_75.21.pth --model KGNN --ds_name 14semeval_laptop --dim_w 300 --dim_k 400 --is_bert 0  --kge analogy


- python -m main_total -ds_name 14semeval_rest -bs 64 -model KGNN -dim_w 300 -dim_k 200 -is_test 1 ## Training options


python predict.py -ds_name 14semeval_laptop -model KGNN
pip install spacy
python -m spacy download en_core_web_lg
pip install googletrans==3.1.0a0

boot time is super fast , around anywhere from 35 seconds to 1 minute.

    screen - although some people might complain about low res which i think is ridiculous .