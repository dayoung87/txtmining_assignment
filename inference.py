import torch
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification
)

from config import *

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR
)

model = (
    AutoModelForTokenClassification
    .from_pretrained(MODEL_DIR)
    .to(DEVICE)
)

model.eval()

def extract_keywords(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        k: v.to(DEVICE)
        for k, v in inputs.items()
    }

    with torch.no_grad():

        outputs = model(**inputs)

        probs = F.softmax(
            outputs.logits,
            dim=-1
        )

    predictions = torch.argmax(
        probs,
        dim=-1
    )[0]

    probs = probs[0]

    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    print("\n")
    print("=" * 100)
    print("TOKEN PREDICTION RESULT")
    print("=" * 100)

    print(
        f"{'TOKEN':<25}"
        f"{'O':>10}"
        f"{'B':>10}"
        f"{'I':>10}"
        f"{'PRED':>10}"
    )

    print("-" * 100)

    keywords = []
    current_keyword = []

    for token, pred, prob in zip(
        tokens,
        predictions,
        probs
    ):

        if token in tokenizer.all_special_tokens:
            continue

        token_clean = token.replace(
            "Ġ",
            ""
        )

        pred_label = id2label[
            pred.item()
        ]

        o_score = prob[
            label2id["O"]
        ].item()

        b_score = prob[
            label2id["B"]
        ].item()

        i_score = prob[
            label2id["I"]
        ].item()

        print(
            f"{token_clean:<25}"
            f"{o_score:>10.4f}"
            f"{b_score:>10.4f}"
            f"{i_score:>10.4f}"
            f"{pred_label:>10}"
        )

        # 키워드 추출

        if pred_label == "B":

            if current_keyword:

                keywords.append(
                    " ".join(
                        current_keyword
                    )
                )

            current_keyword = [
                token_clean
            ]

        elif (
            pred_label == "I"
            and current_keyword
        ):

            current_keyword.append(
                token_clean
            )

        else:

            if current_keyword:

                keywords.append(
                    " ".join(
                        current_keyword
                    )
                )

                current_keyword = []

    if current_keyword:

        keywords.append(
            " ".join(
                current_keyword
            )
        )

    keywords = list(
        dict.fromkeys(
            keywords
        )
    )

    print("\n")
    print("=" * 100)
    print("EXTRACTED KEYWORDS")
    print("=" * 100)

    for idx, keyword in enumerate(
        keywords,
        start=1
    ):

        print(
            f"{idx:>2}. {keyword}"
        )

    return keywords

if __name__ == "__main__":

    article = """
    Apple announced a new partnership with OpenAI
    to integrate generative AI features into future
    iPhone devices.

    CEO Tim Cook said the collaboration will improve
    productivity and user experience.
    """

    print("\n=== KEYWORDS ===\n")

    for keyword in extract_keywords(article):
        print(keyword)