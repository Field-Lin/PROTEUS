import pandas as pd
import numpy as np
import torch
from transformers import pipeline
import os
import sys
from tqdm import tqdm

BASE_DIR = '/root/autodl-tmp/蛋白质数据'
BASE_MODEL_DIR = '/root/autodl-tmp/esm2_model_local/'
TOP_K_PREDICTIONS = 1
CONFIDENCE_THRESHOLD = 0.6
MAX_PREDICTIONS_LIMIT = 500
SCAN_BATCH_SIZE = 256 # 这个batch_size现在是pipeline内部使用的，可以根据多GPU的总显存适当调大，例如64或128
SEQUENCE_COLUMN = 'mutated_sequence'

# --- 2. 辅助函数 (无需修改) ---
def process_pipeline_results(pipeline_output, original_sequence):
    analysis, predicted_seq, different_cnt = '', '', 0
    original_aas = list(original_sequence)
    if len(pipeline_output) != len(original_sequence):
        return "PROCESSING_ERROR", "Output length mismatch"
    for i, single_pos_predictions in enumerate(pipeline_output):
        if not single_pos_predictions:
            predicted_seq += original_aas[i]
            continue
        top_pred = single_pos_predictions[0]
        if top_pred['score'] < CONFIDENCE_THRESHOLD:
            predicted_seq += original_aas[i]
        else:
            predicted_seq += top_pred['token_str']
            if top_pred['token_str'] != original_aas[i]:
                analysis += f"\tposition {i}: predict {top_pred['token_str']} while original {original_aas[i]} (score: {top_pred['score']:.4f})\n"
                different_cnt += 1
    analysis_header = f'\ttotal different aa count: {different_cnt}\n'
    return predicted_seq, analysis_header + analysis

def run_prediction_for_project(project_name: str, base_pipeline):
    print("\n" + "="*80)
    print(f"===== 开始处理项目: {project_name} =====")
    print("="*80)

    project_dir = os.path.join(BASE_DIR, project_name)
    finetuned_model_dir = os.path.join(project_dir, "best_model")
    seq_file = os.path.join(project_dir, f"{project_name}.csv")
    output_file = os.path.join(project_dir, "prediction_results.csv")
    log_file = os.path.join(project_dir, "prediction_differences.txt")

    if os.path.exists(output_file):
        print(f"项目 '{project_name}' 已存在输出文件。跳过。")
        return
    if not all(os.path.exists(p) for p in [finetuned_model_dir, seq_file]):
        print(f"错误: 项目 '{project_name}' 缺少 'best_model' 文件夹或CSV文件。跳过。")
        return

    print(f"加载微调模型从: {finetuned_model_dir}")
    try:
        # device=0 会自动使用CUDA_VISIBLE_DEVICES中指定的所有GPU
        finetuned_pipeline = pipeline("fill-mask", model=finetuned_model_dir, device=0)
        tokenizer = finetuned_pipeline.tokenizer
    except Exception as e:
        print(f"加载微调模型时出错: {e}。跳过此项目。")
        return

    all_data = pd.read_csv(seq_file)
    if 'DMS_score_bin' not in all_data.columns:
        print(f"错误: CSV文件缺少 'DMS_score_bin' 列。跳过。")
        return
    
    seqs_to_predict = all_data[all_data['DMS_score_bin'] == 0].copy()
    if seqs_to_predict.empty:
        print("筛选后没有序列需要预测。")
        return
        
    if len(seqs_to_predict) > MAX_PREDICTIONS_LIMIT:
        print(f"发现 {len(seqs_to_predict)} 条序列，将只处理前 {MAX_PREDICTIONS_LIMIT} 条。")
        seqs_to_predict = seqs_to_predict.head(MAX_PREDICTIONS_LIMIT)
    
    # --- 高效批处理的核心改动 ---
    # 1. 准备所有工作：一次性生成所有序列的所有掩码版本
    print("准备所有掩码序列...")
    all_masked_sequences = []
    protein_info = [] # 用于后续重组结果
    for _, current_row in seqs_to_predict.iterrows():
        seq = str(current_row.get(SEQUENCE_COLUMN, ''))
        if not seq: continue
        if len(seq) > (tokenizer.model_max_length - 2):
            seq = seq[:(tokenizer.model_max_length - 2)]
        
        protein_info.append({
            'name': current_row.get('mutant', current_row.get('name', 'N/A')),
            'original_sequence': seq,
            'row_data': list(current_row)
        })
        for j in range(len(seq)):
            masked_list = list(seq)
            masked_list[j] = tokenizer.mask_token
            all_masked_sequences.append("".join(masked_list))

    if not all_masked_sequences:
        print("没有可处理的掩码序列。")
        return

    # 2. 一次性计算：将整个任务清单交给pipeline
    print(f"开始对总计 {len(all_masked_sequences)} 个掩码进行高效推理...")
    finetuned_results_flat = finetuned_pipeline(all_masked_sequences, batch_size=SCAN_BATCH_SIZE, top_k=TOP_K_PREDICTIONS)
    base_results_flat = base_pipeline(all_masked_sequences, batch_size=SCAN_BATCH_SIZE, top_k=TOP_K_PREDICTIONS)
    
    # 3. 重组结果
    print("重组预测结果...")
    result_datas = []
    current_index = 0
    for info in tqdm(protein_info, desc=f"Processing results for {project_name}"):
        seq_len = len(info['original_sequence'])
        
        # 从巨大结果列表中“切出”属于当前蛋白质的部分
        current_finetuned_results = finetuned_results_flat[current_index : current_index + seq_len]
        current_base_results = base_results_flat[current_index : current_index + seq_len]
        current_index += seq_len
        
        finetuned_seq, finetuned_analysis = process_pipeline_results(current_finetuned_results, info['original_sequence'])
        base_seq, base_analysis = process_pipeline_results(current_base_results, info['original_sequence'])

        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"===== Sequence: {info['name']} | Length: {seq_len} =====\n")
            f.write("--- Finetuned Model Differences ---\n")
            f.write(finetuned_analysis if finetuned_analysis.strip() else "\tNo differences found.\n")
            f.write("\n--- Base Model Differences ---\n")
            f.write(base_analysis if base_analysis.strip() else "\tNo differences found.\n")
            f.write("=" * 60 + "\n\n")
            
        result_datas.append(info['row_data'] + [finetuned_seq, base_seq])

    # 4. 保存最终输出
    new_columns = ['predicted_sequence_trained', 'predicted_sequence']
    output_df = pd.DataFrame(result_datas, columns=list(seqs_to_predict.columns) + new_columns)
    output_df.to_csv(output_file, index=False)
    print(f"\n项目 '{project_name}' 预测完成，结果已保存到 {output_file}")

if __name__ == '__main__':
    print("--- Initializing Base Model (loaded only once) ---")
    try:
        # device=0 会自动指向CUDA_VISIBLE_DEVICES中的第一块GPU，并使用所有可见的GPU
        base_pipeline = pipeline("fill-mask", model=BASE_MODEL_DIR, device=0)
    except Exception as e:
        print(f"FATAL: 无法加载基础模型于 '{BASE_MODEL_DIR}'. 错误: {e}")
        sys.exit(1)
    print("--- Base model loaded successfully. ---\n")

    all_projects = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    if not all_projects:
        print(f"在 '{BASE_DIR}' 中没有找到任何项目子文件夹。")
        sys.exit(1)
        
    print(f"发现 {len(all_projects)} 个潜在项目。开始批量预测...")
    
    for project_name in sorted(all_projects):
        torch.cuda.empty_cache()
        try:
            run_prediction_for_project(project_name, base_pipeline)
        except Exception as e:
            print(f"\n!!!!!! 在处理项目 '{project_name}' 时发生严重错误 !!!!!!")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("!!!!!! 继续处理下一个项目... !!!!!!")
            continue
    print("\n所有项目处理完毕！")