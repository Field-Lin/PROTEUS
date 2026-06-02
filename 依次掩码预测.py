import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import pandas as pd
import numpy as np
import os
import sys

# --- 配置 ---
BASE_DIR = '/root/autodl-tmp/蛋白质数据'
PROJECT_FOLDER_NAME = 'ADRB2_HUMAN_Jones_2020' 
PROJECT_CORE_NAME = 'ADRB2_HUMAN_Jones_2020' 

# --- 模型路径 ---
# 微调后的模型
FINETUNED_MODEL_DIR = os.path.join(BASE_DIR, PROJECT_FOLDER_NAME, "best_model")
# 原始基础模型
BASE_MODEL_DIR = '/root/autodl-tmp/esm2_model_local/'

# --- 自动构建文件路径 (无需修改) ---
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_FOLDER_NAME)
DEVICE = 0 if torch.cuda.is_available() else -1
TOP_K_PREDICTIONS = 1
SEQ_FILE = os.path.join(PROJECT_DIR, f"{PROJECT_CORE_NAME}.csv")
LOG_FILE = os.path.join(PROJECT_DIR, "prediction_differences.txt")
OUTPUT_FILE = os.path.join(PROJECT_DIR, "prediction_results.csv")

# --- 预测参数 ---
confidence_threshold = 0.6
# 注意: 您日志中的限制是100，这里改回1000，您可以按需修改
MAX_PREDICTIONS_LIMIT = 500
SCAN_BATCH_SIZE = 8


def process_pipeline_results(pipeline_output, original_sequence, original_aas):
    """
    解析 pipeline 的原始输出，生成预测序列和差异分析文本。
    """
    analysis = ''
    predicted_seq = ''
    different_cnt = 0

    if len(pipeline_output) != len(original_sequence):
        error_msg = f"错误: Pipeline 输出长度 ({len(pipeline_output)}) 与原始序列长度 ({len(original_sequence)}) 不匹配。"
        print(error_msg)
        return "PROCESSING_ERROR", error_msg

    for i, single_pos_predictions in enumerate(pipeline_output):
        if not single_pos_predictions:
            predicted_seq += original_aas[i]
            continue
            
        top_pred = single_pos_predictions[0]
        
        if top_pred['score'] < confidence_threshold:
            predicted_seq += original_aas[i]
        else:
            predicted_seq += top_pred['token_str']
            if top_pred['token_str'] != original_aas[i]:
                analysis += f"\tposition {i}: predict {top_pred['token_str']} while original {original_aas[i]} (score: {top_pred['score']:.4f})\n"
                different_cnt += 1
                
    analysis_header = f'\ttotal different aa count: {different_cnt}\n'
    return predicted_seq, analysis_header + analysis


if __name__ == '__main__':
    # --- 1. 一次性加载模型和 Pipelines ---
    print("--- Initializing Models and Pipelines ---")
    try:
        print(f"Loading Finetuned Model from: {FINETUNED_MODEL_DIR}")
        finetuned_pipeline = pipeline("fill-mask", model=FINETUNED_MODEL_DIR, device=DEVICE)
    except Exception as e:
        print(f"FATAL: Could not load finetuned model. Error: {e}")
        print(f"Please ensure the path '{FINETUNED_MODEL_DIR}' exists and is a valid model directory.")
        sys.exit(1)

    try:
        print(f"Loading Base Model from: {BASE_MODEL_DIR}")
        base_pipeline = pipeline("fill-mask", model=BASE_MODEL_DIR, device=DEVICE)
    except Exception as e:
        print(f"FATAL: Could not load base model. Error: {e}")
        print(f"Please ensure the path '{BASE_MODEL_DIR}' exists and is a valid model directory.")
        sys.exit(1)
    print("--- Models loaded successfully. ---\n")
    tokenizer = finetuned_pipeline.tokenizer

    # --- 2. 读取和筛选数据 ---
    if not os.path.exists(SEQ_FILE):
        print(f"FATAL: Input file not found at {SEQ_FILE}")
        sys.exit(1)
        
    all_data = pd.read_csv(SEQ_FILE)
    assert 'DMS_score_bin' in all_data.columns, f"Input file {SEQ_FILE} must contain 'DMS_score_bin' column."
    assert 'sequence' in all_data.columns or 'mutated_sequence' in all_data.columns, 'Input CSV must have "sequence" or "mutated_sequence" column.'

    print(f"Original data has {len(all_data)} rows.")
    seqs_to_predict = all_data[all_data['DMS_score_bin'] == 0].copy()
    seqs_to_predict.reset_index(drop=True, inplace=True)
    
    if len(seqs_to_predict) > MAX_PREDICTIONS_LIMIT:
        print(f"WARNING: Found {len(seqs_to_predict)} sequences with DMS_score_bin == 0, exceeding the limit of {MAX_PREDICTIONS_LIMIT}.")
        print(f"Processing only the first {MAX_PREDICTIONS_LIMIT} sequences.")
        seqs_to_predict = seqs_to_predict.head(MAX_PREDICTIONS_LIMIT)
    
    print(f"After filtering and limiting, {len(seqs_to_predict)} sequences will be processed.\n")

    if seqs_to_predict.empty:
        print("No sequences to predict. Exiting.")
        sys.exit(0)

    # --- 3. 主预测循环 ---
    result_datas = []
    sequence_keyword = 'mutated_sequence' if 'mutated_sequence' in seqs_to_predict.columns else 'sequence'

    for i, current_row in seqs_to_predict.iterrows():
        progress_percent = (i + 1) / len(seqs_to_predict) * 100
        print(f"===== Processing sequence {i + 1} / {len(seqs_to_predict)} ({progress_percent:.2f} %) =====")
        
        seq_to_predict = str(current_row[sequence_keyword])
        if pd.isna(seq_to_predict) or not seq_to_predict:
            print(f"  > Skipping row {i} due to empty sequence.")
            continue

        if len(seq_to_predict) > (tokenizer.model_max_length - 2):
             seq_to_predict = seq_to_predict[:(tokenizer.model_max_length - 2)]

        seq_name = current_row.get('mutant', current_row.get('name', f'sequence_row_{i}'))

        # --- 3a. 为两个模型准备一次数据 ---
        masked_sequences = []
        original_aas = list(seq_to_predict)
        for j in range(len(seq_to_predict)):
            masked_list = list(seq_to_predict)
            masked_list[j] = tokenizer.mask_token
            masked_sequences.append("".join(masked_list))
        
        # --- 3b. 高效运行 Pipelines ---
        print(f"  > Running finetuned pipeline for {len(masked_sequences)} masks...")
        # 修正2: 直接将列表 `masked_sequences` 传递给 pipeline
        finetuned_results = finetuned_pipeline(masked_sequences, batch_size=SCAN_BATCH_SIZE, top_k=TOP_K_PREDICTIONS)
        
        print(f"  > Running base pipeline for {len(masked_sequences)} masks...")
        # 同样，将同一个列表传递给另一个 pipeline
        base_results = base_pipeline(masked_sequences, batch_size=SCAN_BATCH_SIZE, top_k=TOP_K_PREDICTIONS)

        # --- 3c. 处理结果 ---
        finetuned_seq, finetuned_analysis = process_pipeline_results(finetuned_results, seq_to_predict, original_aas)
        base_seq, base_analysis = process_pipeline_results(base_results, seq_to_predict, original_aas)

        # --- 3d. 记录和存储 ---
        with open(LOG_FILE, 'a') as f:
            f.write(f"===== Sequence: {str(seq_name)} | Original Length: {len(seq_to_predict)} =====\n")
            f.write("--- Finetuned Model Differences ---\n")
            f.write(finetuned_analysis if finetuned_analysis.strip() else "\tNo differences found based on confidence threshold.\n")
            f.write("\n--- Base Model Differences ---\n")
            f.write(base_analysis if base_analysis.strip() else "\tNo differences found based on confidence threshold.\n")
            f.write("=" * (len(str(seq_name)) + 40) + "\n\n")

        result_datas.append(list(current_row) + [finetuned_seq, base_seq])

    # --- 4. 保存最终输出 ---
    new_columns = ['predicted_sequence_trained', 'predicted_sequence']
    result_df = pd.DataFrame(result_datas, columns=list(seqs_to_predict.columns) + new_columns)
    result_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nPrediction complete. Results saved to {OUTPUT_FILE}")