# Ben Kabongo
# January 2025

import evaluate
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List


def create_prompt(user_reviews, item_reviews, include_aspects=False, aspects_info=None):
    prompt = """Here are reviews from the user:\n"""
    for i, review in enumerate(user_reviews):
        prompt += f"[User]\n{review}\n"

    prompt += """\nHere are reviews about the item:\n"""
    for i, review in enumerate(item_reviews):
        prompt += f"[Item]\n{review}\n"

    if include_aspects and aspects_info:
        sorted_aspects = sorted(aspects_info.items(), key=lambda x: -x[1]['importance'])
        prompt += """\nSorted key aspects to consider: """
        for aspect, info in sorted_aspects:
            prompt += f"- {aspect}: {info['polarity']} "

    prompt += """\nPlease generate a detailed review for the user about this item, maintaining a consistent style:\n"""
    return prompt


def generate(config, model, tokenizer, train_df, test_df):
    # Generate reviews for the test set
    generated_reviews = []
    for index in tqdm(range(len(test_df)), desc="Generating reviews", colour="green"):
        row = test_df.iloc[index]
        user_id = row["user_id"]
        item_id = row["item_id"]

        # sample review from train_df
        user_reviews = train_df[train_df["user_id"] == user_id]["review"]
        user_reviews = user_reviews.sample(min(config.n_reviews, len(user_reviews)), random_state=config.seed).tolist()
        item_reviews = train_df[train_df["item_id"] == item_id]["review"]
        item_reviews = item_reviews.sample(min(config.n_reviews, len(item_reviews)), random_state=config.seed).tolist()
        
        aspects_infos = None
        if config.aspects_flag:
            aspects_infos = row[config.aspects_infos]

        prompt = create_prompt(user_reviews, item_reviews, include_aspects=config.aspects_flag, aspects_info=aspects_infos)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=config.max_new_tokens, temperature=config.temperature, do_sample=True)
        review = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_reviews.append(review)

        if config.verbose:
            ground_thruth = row["review"]
            print("User ID:", user_id)
            print("Item ID:", item_id)
            print("Prompt:\n", prompt)
            print("Generated review:\n", review)
            print("Ground truth:\n", ground_thruth)
            print("==" * 100)
            print("\n")

    return generated_reviews


def review_evaluation(predictions: List[str], references: List[str], config) -> Dict[str, Any]:
    references_list = [[ref] for ref in references]

    bleu_metric = evaluate.load("bleu")
    bleu_results = bleu_metric.compute(predictions=predictions, references=references_list)
    bleu_results["precision"] = np.mean(bleu_results["precisions"])

    bertscore_metric = evaluate.load("bertscore")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions, references=references, lang=config.lang
    )
    bertscore_results["precision"] = np.mean(bertscore_results["precision"])
    bertscore_results["recall"] = np.mean(bertscore_results["recall"])
    bertscore_results["f1"] = np.mean(bertscore_results["f1"])

    meteor_metric = evaluate.load("meteor")
    meteor_results = meteor_metric.compute(predictions=predictions, references=references)

    rouge_metric = evaluate.load("rouge")
    rouge_results = rouge_metric.compute(predictions=predictions, references=references)

    return {
        "n_examples": len(predictions),
        "meteor": float(meteor_results["meteor"]),
        "bleu": float(bleu_results["bleu"]),
        "rouge1": float(rouge_results["rouge1"]),
        "rouge2": float(rouge_results["rouge2"]),
        "rougeL": float(rouge_results["rougeL"]),
        "rougeLsum": float(rouge_results["rougeLsum"]),
        "bertscore.precision": float(bertscore_results["precision"]),
        "bertscore.recall": float(bertscore_results["recall"]),
        "bertscore.f1": float(bertscore_results["f1"]),
    }


if __name__ == "__main__":
    user_reviews = [
        "The hotel was very clean and comfortable.",
        "Customer service was exceptional.",
        "The pool was a bit small but enjoyable.",
        "Delicious breakfast with a good variety."
    ]

    item_reviews = [
        "Great location, close to main attractions.",
        "The rooms were spacious and modern.",
        "A bit overpriced for the quality offered.",
        "Quiet and relaxing atmosphere."
    ]

    aspects_info = {
        "cleanliness": {"importance": 0.9, "polarity": "positive"},
        "service": {"importance": 0.8, "polarity": "positive"},
        "price": {"importance": 0.6, "polarity": "neutral"}
    }

    prompt = create_prompt(user_reviews, item_reviews, include_aspects=True, aspects_info=aspects_info)
    print("\nPrompt :\n", prompt)
