import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from config import *

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)    # 모델에 맞는 토크나이저 로딩

model = (AutoModelForTokenClassification.from_pretrained(MODEL_DIR).to(DEVICE)) # 학습시킨 모델 가져오기

model.eval()    # 모델 추론 모드로 변경

def extract_keywords(text): # 키워드 추출 수행 함수

    inputs = tokenizer( # 토크나이저 호출
        text,
        return_tensors="pt",    # 결과를 PyTorch 텐서로 반환
        truncation=True,    # 토큰 길이 너무 길면 자동 잘라냄
        max_length=512  # 최대 토큰 길이 제한
    )

    inputs = {  # inputs 내부 요소 정렬
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():   # 추론 수행이므로 기울기 계산 X 

        outputs = model(**inputs)   # 모델 삽입

        probs = F.softmax(outputs.logits, dim=-1)   # 각 토큰 라벨별 확률 계산

    predictions = torch.argmax(probs, dim=-1)[0]    # 각 토큰 라벨 결정

    probs = probs[0]    # 배치 차원 제거

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])    # 토큰 자연어로 변환

    print("\n")
    print("=" * 100)    # 구분선
    print("TOKEN PREDICTION RESULT")    # 제목
    print("=" * 100)    # 구분선

    print(f"{'TOKEN':<25} {'O':>10} {'B':>10} {'I':>10} {'PRED':>10}")  # 행 인덱스 출력

    print("-" * 100)    # 구분선

    keywords = []   # 최종 키워드 리스트
    current_keyword = []    # 임시 저장 키워드

    for token, pred, prob in zip(
        tokens, # 현재 토큰 
        predictions,    # 예측된 라벨 id
        probs   # 해당 토큰의 라벨별 확률 벡터
    ):

        if token in tokenizer.all_special_tokens:   # 특수 토큰 건너뛰기
            continue

        token_clean = token.replace(    # RoBERTa나 GPT 계열 토크나이저의 단어 시작 표기 문자 삭제
            "Ġ",
            ""
        )

        pred_label = id2label[pred.item()]  # 최종 예측 라벨

        o_score = prob[label2id["O"]].item()    # O 라벨 확률

        b_score = prob[label2id["B"]].item()    # B 라벨 확률

        i_score = prob[label2id["I"]].item()    # I 라벨 확률

        print(f"{token_clean:<25} {o_score:>10.4f} {b_score:>10.4f} {i_score:>10.4f} {pred_label:>10}") # 행 인덱스 출력

        # 키워드 결합
        if pred_label == "B":   # 예측 라벨 == B

            if current_keyword: 
                keywords.append(" ".join(current_keyword))  # 키워드 목록 저장

            current_keyword = [token_clean] # 초기화

        elif (pred_label == "I" and current_keyword):   # 예측 라벨 == I

            current_keyword.append(token_clean) # 앞 키워드와 결합 후 초기화

        else:   # 예측 라벨 == O

            if current_keyword:
                keywords.append(" ".join(current_keyword))  # 임시 저장했던 키워드 목록에 저장
                current_keyword = []    # 초기화

    if current_keyword:

        keywords.append(" ".join(current_keyword))  # 임시 저장했던 키워드 목록에 저장

    keywords = list(dict.fromkeys(keywords))    # 순서를 유지하면서 중복 키워드 제거함

    print("\n")
    print("=" * 100)    # 구분선
    print("EXTRACTED KEYWORDS") # 제목
    print("=" * 100)    # 구분선

    for idx, keyword in enumerate(keywords,start=1):    # 키워드 목록 순회

        print(f"{idx:>2}. {keyword}")   # 결과값 출력

    return keywords

if __name__ == "__main__":  # 메인 함수 실행

    # 기사 본문
    article = """
    Apple announced a new partnership with OpenAI
    to integrate generative AI features into future
    iPhone devices.

    CEO Tim Cook said the collaboration will improve
    productivity and user experience.
    """

    print("\n=== KEYWORDS ===\n")   # 구분선

    for keyword in extract_keywords(article):   # 추론 수행 및 결과 출력
        print(keyword)