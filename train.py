import time
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime

from datasets import (
    load_dataset,
    Dataset,
    DatasetDict
)

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments
)

from seqeval.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from config import *

# 데이터셋 가져오기 =====================================================
def load_kptimes_dataset():

    base_url = (    # 허깅페이스 데이터셋 위치 링크
        "https://huggingface.co/datasets/"
        "midas/kptimes/resolve/main/"
    )

    data_files = {
        "train": base_url + "train.jsonl",  # 훈련셋
        "validation": base_url + "valid.jsonl", # 검증셋
        "test": base_url + "test.jsonl" # 테스트셋
    }

    return load_dataset("json", data_files=data_files)  # 데이터셋 로드

# 데이터 샘플링 =====================================================
def sample_dataset(dataset):

    train_df = (
        dataset["train"]
        .shuffle(seed=SEED) # 편향 방지 위한 셔플
        .select(range(TRAIN_SIZE))  # 일정 크기로 샘플링
        .to_pandas()    # 판다스 데이터프레임 포맷으로 변환
    )

    val_df = (
        dataset["validation"]
        .shuffle(seed=SEED)
        .select(range(VAL_SIZE))
        .to_pandas()
    )

    test_df = (
        dataset["test"]
        .shuffle(seed=SEED)
        .select(range(TEST_SIZE))
        .to_pandas()
    )

    return train_df, val_df, test_df

# 데이터 전처리 =====================================================
def preprocess_dataframe(df):

    df["doc_str"] = df["document"].apply(lambda x: " ".join(x)) # 문자열로 변경

    df.drop_duplicates(subset=["doc_str"], inplace=True)    # doc_str 기준 중복 데이터 제거

    df.dropna(inplace=True, how='any') # 행에 NaN이 있으면 해당 행 제거

    df.reset_index(drop=True, inplace=True) # 인덱스 재정렬

    df.drop(columns=["doc_str"], inplace=True)  # 임시 컬럼 삭제

    return df

# 단어 단위 BIO 태그 -> 토큰 단위 라벨로 정렬 =====================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_CKPT)   # 모델에 맞는 토크나이저 로딩 

def tokenize_and_align_labels(batch):

    tokenized_inputs = tokenizer(   # 데이터 토큰화
        batch["document"],
        truncation=True,
        max_length=512,
        is_split_into_words=True    # 이미 단어 단위로 분리된 상태임
    )

    labels = [] # 라벨 저장 리스트

    for i, label in enumerate(batch["doc_bio_tags"]):   # 각 문장 별 처리

        word_ids = tokenized_inputs.word_ids(batch_index=i) # 단어 id 처리

        previous_word_idx = None    # 이전 단어 인덱스 저장용
        label_ids = []

        for word_idx in word_ids:   # 토큰별 라벨 생성

            if word_idx is None:    # 특수 토큰의 경우
                label_ids.append(-100)  # loss 계산 시 무시

            elif word_idx != previous_word_idx: # 단어의 첫 토큰 (B 라벨)
                label_ids.append(label2id[label[word_idx]]) # 원래 라벨 사용

            else:   # 단어의 서브워드 (I 라벨)
                current_label = label[word_idx] # 원래 태그 가져오기

                if current_label == "B":    # I 태그로 변환
                    current_label = "I"

                label_ids.append(label2id[current_label])   # 변환한 라벨 저장

            previous_word_idx = word_idx    # 이전 단어 갱신

        labels.append(label_ids)    # 문장 라벨 저장

    tokenized_inputs["labels"] = labels # 라벨 추가

    return tokenized_inputs

# 성능 평가 지수 정의 =====================================================
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)  # 각 토큰 중 최고값 선택

    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(preds, labels):  # 한 문장씩 처리
        p_seq = []
        l_seq = []

        for p, l in zip(pred_seq, label_seq):   # 토큰 하나씩 확인
            if l == -100:   # 특수 토큰 제외
                continue
            p_seq.append(id2label[p])   # id -> 태그 변환
            l_seq.append(id2label[l])

        true_preds.append(p_seq)
        true_labels.append(l_seq)

    return {    # 결과 계산 및 반환
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

# Main =====================================================

def main():

    print(f"Device : {DEVICE}") # CPU? GPU? 확인

    OUTPUT_DIR = Path("outputs")    # 결과 txt 저장할 파일 구축
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_kptimes_dataset()    # 데이터셋 로드

    train_df, val_df, test_df = sample_dataset(dataset) # 데이터 샘플링

    # 데이터 전처리
    train_df = preprocess_dataframe(train_df)
    val_df = preprocess_dataframe(val_df)
    test_df = preprocess_dataframe(test_df)

    dataset = DatasetDict({ # 전처리 완료한 데이터프레임으로부터 데이터셋 생성
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })

    encoded = dataset.map(
        tokenize_and_align_labels,  # 토큰 단위 라벨로 정렬
        batched=True,
        remove_columns=dataset["train"].column_names    # RAM 줄이기 위한 컬럼 지우기
    )

    model = (   # 모델 로딩
        AutoModelForTokenClassification.from_pretrained(   # 토큰 분류
            MODEL_CKPT,
            num_labels=len(LABEL_LIST),
            id2label=id2label,
            label2id=label2id
        ).to(DEVICE)
    )

    data_collator = (   # 배치 단위 패딩
        DataCollatorForTokenClassification(tokenizer=tokenizer)
    )

    # 학습 파라미터 정의
    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),  # 학습 결과 저장 폴더

        num_train_epochs=EPOCHS,    # 에포크 수
        learning_rate=LEARNING_RATE,    # 학습률
        weight_decay=WEIGHT_DECAY,  # L2 정규화 계수 (과적합 예방)

        per_device_train_batch_size=BATCH_SIZE, # 학습 배치 크기
        per_device_eval_batch_size=(BATCH_SIZE*2),  # 평가 배치 크기

        fp16=True,  # GPU 이용 혼합 정밀도 학습 수행
        dataloader_num_workers=2,  # 데이터 로딩 프로세스 개수
        dataloader_pin_memory=True, # CPU -> GPU 전송 최적화

        eval_strategy="epoch",  # 각 에포크마다 검증 수행
        save_strategy="epoch",  # 각 에포크마다 모델 저장
        save_total_limit=1, # 최신 체크포인트만 유지

        load_best_model_at_end=True,    # 가장 성능이 좋은 모델 로드
        metric_for_best_model="f1", # 성능 평가 기준 지표
        greater_is_better=True, # 높을수록 좋음

        logging_strategy="steps",   # 각 스텝마다 로그 출력
        logging_steps=50,   # 로그 스텝 크기 정의

        report_to="none",   # 외부 로깅 툴 미사용

        gradient_accumulation_steps=1,  # 그레디언트 즉시 업데이트
        optim="adamw_torch",  # 옵티마이저 지정
    )

    trainer = Trainer(  # 학습 설정
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    print("\n===== TRAIN START =====\n")
    trainer.train() # 학습 실행

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")    # 현재 시간 저장

    history = pd.DataFrame(trainer.state.log_history)

    epoch_history = history[history["eval_f1"].notna()][[
        "epoch",
        "eval_loss",
        "eval_precision",
        "eval_recall",
        "eval_f1"
    ]]

    result_file = (OUTPUT_DIR/ f"epoch_metrics_{timestamp}.txt") # 에포크별 평가 결과 저장할 파일 생성

    with open(  # 파일에 결과값 작성
        result_file, "w", encoding="utf-8"
    ) as f:

        f.write("=" * 60 + "\n")    # 구분선
        f.write("EPOCH METRICS\n")  # 제목
        f.write("=" * 60 + "\n\n"   # 구분선
    )
        for _, row in (epoch_history.iterrows()):   # 에포크마다 작성
            f.write(f"Epoch {int(row['epoch'])}\n")  # 에포크 수
            f.write(f"Loss      : {row['eval_loss']:.4f}\n") # loss
            f.write(f"Precision : {row['eval_precision']:.4f}\n")    # 정밀도
            f.write(f"Recall    : {row['eval_recall']:.4f}\n")   # 재현율
            f.write(f"F1 Score  : {row['eval_f1']:.4f}\n")   # F1 점수 (정밀도와 재현율의 조화 평균)
            f.write("-" * 60 + "\n")    # 구분선

        best_row = epoch_history.loc[
            epoch_history["eval_f1"].idxmax()   # F1 점수 기준 최고값 선정
        ]
        f.write("\n")
        f.write("=" * 60 + "\n")    # 구분선
        f.write("BEST EPOCH\n") # 제목 
        f.write("=" * 60 + "\n")    # 구분선

        f.write(f"Epoch     : {int(best_row['epoch'])}\n")   # 에포크 수
        f.write(f"F1 Score  : {best_row['eval_f1']:.4f}\n")  # F1 점수

    print(f"Epoch metrics saved -> {result_file}")    # 콘솔 출력

    print("\n===== TEST START =====\n")

    start = time.time() # 타이머 시작
    result = trainer.predict(encoded["test"])   # 추론 시작

    end = time.time()   # 타이머 중지
    print(result.metrics)   # 평가 점수 출력

    print(f"\nInference Time : {end - start:.4f}s")  # 추론 소요 시간

    with open(  # 파일에 결과 이어서 작성
        result_file, "a", encoding="utf-8"
    ) as f:   
        f.write("\n")
        f.write("=" * 60 + "\n")    # 구분선
        f.write("FINAL TEST RESULT\n")  # 제목
        f.write("=" * 60 + "\n\n")  # 구분선
        f.write(f"Loss      : {result.metrics['test_loss']:.4f}\n")  # loss
        f.write(f"Precision : {result.metrics['test_precision']:.4f}\n")    # 정밀도
        f.write(f"Recall    : {result.metrics['test_recall']:.4f}\n")   # 재현율
        f.write(f"F1 Score  : {result.metrics['test_f1']:.4f}\n")    # F1 점수 (정밀도와 재현율의 조화 평균)
        f.write(f"Runtime   : {result.metrics['test_runtime']:.2f} sec\n")   # 소요 시간

    print(f"Test metrics saved -> {result_file}")   # 콘솔 출력

    MODEL_DIR.mkdir(    # 모델 저장할 파일 생성
        parents=True,
        exist_ok=True
    )

    trainer.save_model(str(MODEL_DIR))  # 모델 저장

    tokenizer.save_pretrained(str(MODEL_DIR))   # 토크나이저 저장

    print(f"\nSaved Model -> {MODEL_DIR}")  # 콘솔 출력

if __name__ == "__main__":  # 메인 함수 실행
    main()