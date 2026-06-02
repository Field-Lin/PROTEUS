import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForMaskedLM, get_scheduler
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import random
import numpy as np
import os
import math
from pathlib import Path
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

torch.cuda.empty_cache()

# 项目的基础目录
BASE_DIR = '/root/autodl-tmp/蛋白质数据'

# 您的项目名称 (这是您数据文件夹和.csv文件的核心名称)
# 例如: '(fitness)PPM1D_HUMAN_Miller_2022'
PROJECT_FOLDER_NAME = 'BLAT_ECOLX_Firnberg_2014'
# 通常是文件夹名去掉括号和前缀的部分
PROJECT_CORE_NAME = 'BLAT_ECOLX_Firnberg_2014'

# --- 自动构建文件路径 (无需修改) ---
PROJECT_DIR = os.path.join(BASE_DIR, PROJECT_FOLDER_NAME)
DATA_FILE = os.path.join(PROJECT_DIR, f"{PROJECT_CORE_NAME}.csv")
OUTPUT_DIR = PROJECT_DIR # 训练输出（模型、日志等）直接保存在项目文件夹内

# --- 数据列名 ---
SEQUENCE_COLUMN = "mutated_sequence"
ACTIVITY_COLUMN = "DMS_score_bin"

# --- 模型 ---
MODEL_NAME = "./esm2_model_local/" # 基座模型通常不变

# --- 训练参数 ---
MAX_LENGTH = 1024
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
EPOCHS = 1
MASK_PROB = 0.15
VAL_SET_SIZE = 0.1

# --- 环境设置 ---
TENSORBOARD_LOG_DIR = Path(OUTPUT_DIR) / "runs"
RANDOM_SEED = 42
NUM_WORKERS = 0 if os.name == 'nt' else 4

# --- 设置随机种子保证可复现性 ---
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 创建输出目录 ---
if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(TENSORBOARD_LOG_DIR): os.makedirs(TENSORBOARD_LOG_DIR)


# --- 2. 自定义数据集 (已移除滑动窗口，极大简化) ---
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

        # Tokenize, 截断 & 填充 一步到位
        tokenized_output = self.tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt' # 返回PyTorch张量
        )
        input_ids = tokenized_output['input_ids'].squeeze(0) # 移除批次维度
        attention_mask = tokenized_output['attention_mask'].squeeze(0)

        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        # 确定可以被掩码的位置 (非特殊token)
        can_be_masked_indices = [
            i for i, token_id in enumerate(input_ids)
            if token_id not in self.special_token_ids_set
        ]

        if not can_be_masked_indices:
            # 如果序列全是特殊token或为空，则不进行掩码
            labels.fill_(-100)
        else:
            # 计算要掩码的token数量
            num_to_mask = max(1, int(len(can_be_masked_indices) * self.mask_prob))
            indices_to_mask = random.sample(can_be_masked_indices, num_to_mask)

            labels.fill_(-100) # 首先将所有label设为-100
            for i in indices_to_mask:
                labels[i] = masked_input_ids[i] # 在label中记录原始token

                # 80% 替换为 <mask>
                if random.random() < 0.8:
                    masked_input_ids[i] = self.tokenizer.mask_token_id
                # 10% 替换为随机token
                elif random.random() < 0.9:
                    random_token_id = random.randint(0, self.tokenizer.vocab_size - 1)
                    masked_input_ids[i] = random_token_id
                # 10% 保持不变

        return {
            "input_ids": masked_input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# --- 3. 加载、清洗并划分数据 ---
print(f"从 '{DATA_FILE}' 加载数据...")
try:
    df_all = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"错误: 数据文件 '{DATA_FILE}' 未找到。请检查文件名和路径。")
    exit()

# 检查必需的列是否存在
required_columns = [SEQUENCE_COLUMN, ACTIVITY_COLUMN]
if not all(col in df_all.columns for col in required_columns):
    print(f"错误: 数据文件必须包含以下列: {required_columns}")
    exit()

# 数据清洗
df_all.dropna(subset=[SEQUENCE_COLUMN], inplace=True)
df_all[SEQUENCE_COLUMN] = df_all[SEQUENCE_COLUMN].astype(str)
df_all = df_all[df_all[SEQUENCE_COLUMN].str.len() > 0]
df_all = df_all.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

print(f"成功加载 {len(df_all)} 条有效序列。")
print(f"数据分布: \n{df_all[ACTIVITY_COLUMN].value_counts()}")
if len(df_all) == 0:
    print("错误：数据集中没有有效的序列。")
    exit()

# 划分训练集和验证集 (使用分层抽样)
train_df, val_df = train_test_split(
    df_all,
    test_size=VAL_SET_SIZE,
    random_state=RANDOM_SEED,
    stratify=df_all[ACTIVITY_COLUMN] # 确保训练/验证集活性分布一致
)
print(f"\n训练集大小: {len(train_df)}, 验证集大小: {len(val_df)}")
print(f"训练集活性分布:\n{train_df[ACTIVITY_COLUMN].value_counts(normalize=True)}")
print(f"验证集活性分布:\n{val_df[ACTIVITY_COLUMN].value_counts(normalize=True)}")


# --- 4. 初始化Tokenizer和模型 ---
print(f"\n加载Tokenizer和预训练模型: {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)


# --- 5. 创建DataLoaders ---
print("创建数据集和DataLoaders...")
train_sequences = train_df[SEQUENCE_COLUMN].tolist()
val_sequences = val_df[SEQUENCE_COLUMN].tolist()

train_dataset = ProteinSequenceDataset(train_sequences, tokenizer, MAX_LENGTH, MASK_PROB)
val_dataset = ProteinSequenceDataset(val_sequences, tokenizer, MAX_LENGTH, MASK_PROB)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)


# --- 6. 训练设置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"\n使用设备: {device}")

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scaler = GradScaler(enabled=torch.cuda.is_available())

num_training_steps = EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=int(num_training_steps * 0.1),
    num_training_steps=num_training_steps
)
writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
print(f"TensorBoard 日志将保存在: {TENSORBOARD_LOG_DIR}")


# --- 7. 训练与验证循环 ---
print("开始微调...")
global_step = 0
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")

    for i, batch in enumerate(train_dataloader):
        # 将数据移到GPU
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
        writer.add_scalar('Loss/train_step', loss.item(), global_step)
        global_step += 1

        if (i + 1) % 50 == 0:
            print(f"  Batch {i+1}/{len(train_dataloader)}, Step Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} 平均训练损失: {avg_train_loss:.4f}")
    writer.add_scalar('Loss/train_epoch', avg_train_loss, epoch + 1)
    writer.add_scalar('LearningRate/epoch', lr_scheduler.get_last_lr()[0], epoch + 1)

    # --- 验证 ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
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

    # --- 保存最佳模型 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print(f"  -> 发现新的最佳验证损失! 正在保存模型至 {OUTPUT_DIR}/best_model")
        best_model_dir = os.path.join(OUTPUT_DIR, "best_model")
        model.save_pretrained(best_model_dir)
        tokenizer.save_pretrained(best_model_dir)

# --- 8. 训练结束，保存最终模型 ---
print("\n训练完成。")
# final_model_dir = os.path.join(OUTPUT_DIR, "final_model")
# model.save_pretrained(final_model_dir)
# tokenizer.save_pretrained(final_model_dir)
# print(f"最终模型已保存至 {final_model_dir}")
print(f"性能最佳的模型保存在 {os.path.join(OUTPUT_DIR, 'best_model')}")

writer.close()
print(f"\nTensorBoard 日志已保存。请运行 'tensorboard --logdir={TENSORBOARD_LOG_DIR}' 查看。")