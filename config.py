from pathlib import Path
import torch

# =====================================================
# 모델 종류
# bert-base-cased -> Bert 모델명
# roberta-base -> Roberta 모델명
# distilbert-base-cased -> DistillBERT 모델명
# =====================================================

MODEL_CKPT = "bert-base-cased" # 모델 이름

TRAIN_SIZE = 10000
VAL_SIZE = 1000
TEST_SIZE = 1000
SEED = 26
BATCH_SIZE = 16
EPOCHS = 10

# 저장 경로 지정
MODEL_DIR = (
    Path("models")
    / f"{MODEL_CKPT}-{EPOCHS}epochs-{TRAIN_SIZE}"
)

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# BIO 태그 정의
LABEL_LIST = ["O", "B", "I"]

label2id = {
    label: idx
    for idx, label in enumerate(LABEL_LIST)
}

id2label = {
    idx: label
    for idx, label in enumerate(LABEL_LIST)
}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)