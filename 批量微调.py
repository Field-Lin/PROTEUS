import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import numpy as np
import os
import math
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# 项目的基础目录
BASE_DIR = '/root/autodl-tmp/蛋白质数据'

# --- 数据列名 ---
SEQUENCE_COLUMN = "mutated_sequence"
ACTIVITY_COLUMN = "DMS_score_bin"

# --- 模型 ---
# 修改点: 将相对路径改为绝对路径
# 之前的: MODEL_NAME = "./esm2_model_local/"
# 现在的:
MODEL_NAME = "/root/autodl-tmp/esm2_model_local/" # 基座模型使用绝对路径，保证在任何地方运行脚本都能找到它

# --- 训练参数 ---
MAX_LENGTH = 1024
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS = 1
MASK_PROB = 0.15
VAL_SET_SIZE = 0.1

# --- 环境设置 ---
RANDOM_SEED = 42
NUM_WORKERS = 0 if os.name == 'nt' else 4

# --- 设置全局随机种子 ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(RANDOM_SEED)

# --- 2. 自定义数据集 (无需修改) ---
class ProteinSequenceDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_length, mask_prob):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.special_token_ids_set = {
            tokenizer.cls_token_id, tokenizer.eos_token_id,
            tokenizer.pad_token_id, tokenizer.mask_token_id,
            tokenizer.unk_token_id
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        tokenized_output = self.tokenizer(
            sequence, truncation=True, padding='max_length',
            max_length=self.max_length, return_tensors='pt'
        )
        input_ids = tokenized_output['input_ids'].squeeze(0)
        attention_mask = tokenized_output['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()
        can_be_masked_indices = [
            i for i, token_id in enumerate(input_ids)
            if token_id not in self.special_token_ids_set
        ]
        if can_be_masked_indices:
            num_to_mask = max(1, int(len(can_be_masked_indices) * self.mask_prob))
            indices_to_mask = random.sample(can_be_masked_indices, num_to_mask)
            labels.fill_(-100)
            for i in indices_to_mask:
                labels[i] = masked_input_ids[i]
                if random.random() < 0.8:
                    masked_input_ids[i] = self.tokenizer.mask_token_id
                elif random.random() < 0.9:
                    masked_input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)
        else:
            labels.fill_(-100)
        return {"input_ids": masked_input_ids, "attention_mask": attention_mask, "labels": labels}

def run_finetuning_for_project(project_name: str):
    """
    针对单个项目执行完整的微调流程。
    """
    print("\n" + "="*80)
    print(f"===== 开始处理项目: {project_name} =====")
    print("="*80)

    # --- 动态构建路径 ---
    project_dir = os.path.join(BASE_DIR, project_name)
    data_file = os.path.join(project_dir, f"{project_name}.csv")
    output_dir = project_dir
    best_model_dir = os.path.join(output_dir, "best_model")
    
    # --- 检查是否已训练 ---
    if os.path.exists(best_model_dir):
        print(f"项目 '{project_name}' 已经存在一个 'best_model' 文件夹。跳过训练。")
        return

    # --- 1. 加载和准备数据 ---
    print(f"从 '{data_file}' 加载数据...")
    if not os.path.exists(data_file):
        print(f"错误: 数据文件 '{data_file}' 未找到。跳过此项目。")
        return
        
    df_all = pd.read_csv(data_file)
    required_columns = [SEQUENCE_COLUMN, ACTIVITY_COLUMN]
    if not all(col in df_all.columns for col in required_columns):
        print(f"错误: {project_name} 的数据文件必须包含以下列: {required_columns}。跳过此项目。")
        return

    df_all.dropna(subset=[SEQUENCE_COLUMN], inplace=True)
    df_all[SEQUENCE_COLUMN] = df_all[SEQUENCE_COLUMN].astype(str)
    df_all = df_all[df_all[SEQUENCE_COLUMN].str.len() > 0]
    df_all = df_all.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if len(df_all) == 0:
        print("错误：数据集中没有有效的序列。跳过此项目。")
        return
    print(f"成功加载 {len(df_all)} 条有效序列。")
    
    # 划分训练集和验证集
    train_df, val_df = train_test_split(
        df_all, test_size=VAL_SET_SIZE, random_state=RANDOM_SEED,
        stratify=df_all[ACTIVITY_COLUMN] if df_all[ACTIVITY_COLUMN].nunique() > 1 else None
    )

    # --- 2. 初始化模型、Tokenizer和DataLoaders ---
    print(f"\n加载Tokenizer和预训练模型: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    
    train_dataset = ProteinSequenceDataset(train_df[SEQUENCE_COLUMN].tolist(), tokenizer, MAX_LENGTH, MASK_PROB)
    val_dataset = ProteinSequenceDataset(val_df[SEQUENCE_COLUMN].tolist(), tokenizer, MAX_LENGTH, MASK_PROB)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    
    # --- 3. 训练设置 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"\n使用设备: {device}")
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(enabled=torch.cuda.is_available())
    num_training_steps = EPOCHS * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps
    )
    
    tensorboard_log_dir = Path(output_dir) / "runs"
    writer = SummaryWriter(log_dir=tensorboard_log_dir)
    print(f"TensorBoard 日志将保存在: {tensorboard_log_dir}")
    
    # --- 4. 训练与验证循环 ---
    print("开始微调...")
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        
        # 使用tqdm来显示进度条
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training")
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()
            total_train_loss += loss.item()
            # 在进度条上显示实时损失
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.4f}")
        writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc=f"Epoch {epoch+1} Validation"):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                with autocast(enabled=torch.cuda.is_available()):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch+1} 验证损失: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/validation_epoch', avg_val_loss, epoch + 1)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"  -> 发现新的最佳验证损失! 正在保存模型至 {best_model_dir}")
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)

    print(f"\n项目 '{project_name}' 训练完成。")
    print(f"性能最佳的模型保存在 {best_model_dir}")
    writer.close()

if __name__ == '__main__':
    if not os.path.isdir(BASE_DIR):
        print(f"错误: 基础目录 '{BASE_DIR}' 不存在。请检查路径。")
        exit()

    all_projects = [d for d in os.listdir(BASE_DIR) if os.path.isdir(os.path.join(BASE_DIR, d))]
    
    if not all_projects:
        print(f"在 '{BASE_DIR}' 中没有找到任何项目子文件夹。")
        exit()
        
    print(f"发现 {len(all_projects)} 个潜在项目。开始批量微调...")
    
    for project_name in sorted(all_projects):
        torch.cuda.empty_cache()
        try:
            run_finetuning_for_project(project_name)
        except Exception as e:
            print(f"\n!!!!!! 在处理项目 '{project_name}' 时发生严重错误 !!!!!!")
            print(f"错误类型: {type(e).__name__}")
            print(f"错误信息: {e}")
            print("!!!!!! 继续处理下一个项目... !!!!!!")
            continue

    print("\n所有项目处理完毕！")