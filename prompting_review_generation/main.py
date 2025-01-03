# Ben Kabongo
# January 2025


import argparse
import json
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import generate, review_evaluation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/flan-t5-xxl")
    parser.add_argument("--train_path", type=str, default="/home/b.kabongo/aspects_datasets/Hotels/train.csv")
    parser.add_argument("--test_path", type=str, default="/home/b.kabongo/aspects_datasets/Hotels/test.csv")
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--output_dir", type=str, default="/home/b.kabongo/exps/Hotels/prompting_review_generation/flan-t5-xxl")
    parser.add_argument("--n_reviews", type=int, default=4)
    parser.add_argument("--aspects_flag", action=argparse.BooleanOptionalAction)
    parser.set_defaults(aspects_flag=False)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    parser.set_defaults(verbose=True)

    config = parser.parse_args()

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    train_df = pd.read_csv(config.train_path)
    test_df = pd.read_csv(config.test_path)

    predictions = generate(config, model, tokenizer, train_df, test_df)
    references = test_df["review"]
    output_df = pd.DataFrame({
        "user_id": test_df["user_id"],
        "item_id": test_df["item_id"],
        "predictions": predictions,
        "references": references
    })
    os.makedirs(config.output_dir, exist_ok=True)
    output_df.to_csv(os.path.join(config.output_dir, "output.csv"), index=False)

    scores = review_evaluation(predictions, references, config)
    with open(os.path.join(config.output_dir, "res.json"), "w") as f:
        json.dump(scores, f)
