import os
import trained_models

data_dir_dic = {
    'bm25_history': lambda ds_type, idx: f"data/history/{ds_type}_{idx}.json",
}

pretrain_map = {
    'bert-wwm-ext': "c3pretrain/chinese-bert-wwm-ext",
    'ernie_c3': "c3pretrain/ernie_c3",
}
# pretrain_model = pretrain_map['roberta_wwm_large_ext']
pretrain_model = pretrain_map['bert-wwm-ext']
ir_type = 'bm25_history'
gpu = "0,1"
dev_gpu = "2"
max_seq_len = 512
lr = '3e-5'
epoch = 10
p_num = 3
batch_size = 4
grad_acc = 1

output = 'output/'
data_dir = "data"
model_type = "dcmn"
comment = f"bs_{batch_size}_acc_{grad_acc}"

models = trained_models.ernie_c3_history_bs16

def train_one(idx):
    command = f'''
        MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={gpu} python ./run_dcmn_geo_4.py \
        --model_type {model_type} \
        --model_name_or_path "{pretrain_model}" \
        --do_train \
        --do_lower_case \
        --data_dir {data_dir}\
        --train_file "{data_dir_dic[ir_type]('train', idx)}" \
        --dev_file "{data_dir_dic[ir_type]('dev', idx)}" \
        --max_seq_length {max_seq_len} \
        --p_num {p_num} \
        --batch_size {batch_size} \
        --learning_rate {lr} \
        --gradient_accumulation_steps {grad_acc} \
        --num_train_epochs {epoch} \
        --output_dir {output} \
        --seed 1 \
        --warmup_proportion 0 \
        --overwrite_output_dir \
        --logging_steps 1 \
        --comment "{comment}" \
        --evaluate_during_training \
        --overwrite_cache
    '''
    print(os.system(command))


def train():
    # for idx in range(0, 10, 3):
    for idx in range(0, 5, 1):
        train_one(idx)


def dev_all(test_model, ir_type, i, model_type, p_num, max_seq_len):
    # for ds_type in ['dev', 'test']:

    for ds_type in ['dev', 'test']:
        command = f'''
        CUDA_LAUNCH_BLOCKING=1 MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={dev_gpu} python ./do_evaluate_4.py \
        --model_dir "{test_model}" \
        --model_type "{model_type}" \
        --do_test \
        --data_dir {data_dir} \
        --file "{data_dir_dic[ir_type](ds_type, i)}" \
        --max_seq_length {max_seq_len} \
        --batch_size 4 \
        --p_num {p_num} \
        --overwrite_cache
        '''
        print(os.system(command))

def dev_all_folds():
    for idx, test_model in enumerate(models):
        test_model = os.path.join(output, test_model)
        dev_all(test_model, ir_type, idx, model_type, p_num, max_seq_len)
        input()


if __name__ == '__main__':
    # train()
    dev_all_folds()
