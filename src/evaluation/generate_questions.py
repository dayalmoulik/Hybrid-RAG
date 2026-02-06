import os
import json
import random
import re
from tqdm import tqdm
from transformers import pipeline

# ---------------- CONFIG ----------------
INPUT_DIR = "data/processed"
OUTPUT_FILE = "data/questions/questions_100.json"

TARGETS = {
    "factual": 40,
    "comparative": 25,
    "inferential": 20,
    "multi-hop": 15
}

TOTAL_QUESTIONS = sum(TARGETS.values())

# ---------------- LLM (Rewrite Only) ----------------
rewriter = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=64
)

def rewrite_question_llm(question, answer):
    prompt = (
        "Rewrite the question to be clearer and more natural.\n"
        "Do NOT change its meaning.\n"
        "Do NOT add new information.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}"
    )
    out = rewriter(prompt)[0]["generated_text"]
    return out.strip()

# ---------------- UTILS ----------------
def split_sentences(text, min_len=30):
    sents = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sents if len(s.strip()) >= min_len]

# ---------------- RULE-BASED GENERATORS ----------------
def factual_qa(doc):
    sents = doc["_sentences"]
    if not sents:
        return None
    return {
        "question": f"What does {doc['title']} refer to?",
        "answer": sents[0]
    }

def comparative_qa(doc):
    sents = doc["_sentences"]
    if len(sents) < 2:
        return None
    return {
        "question": f"How do the concepts discussed in {doc['title']} differ?",
        "answer": f"{sents[0]} {sents[1]}"
    }

CAUSE_MARKERS = (
    "because", "therefore", "thus",
    "as a result", "leads to", "results in", "causes"
)

def inferential_qa(doc):
    for s in doc["_sentences"]:
        if any(k in s.lower() for k in CAUSE_MARKERS):
            return {
                "question": "Why does this occur?",
                "answer": s
            }
    return None

def multihop_qa(doc):
    sents = doc["_sentences"]
    if len(sents) < 3:
        return None
    mid = sents[len(sents)//2]
    return {
        "question": "How does the initial concept lead to the final outcome?",
        "answer": f"{sents[0]} {mid} {sents[-1]}"
    }

GEN_MAP = {
    "factual": factual_qa,
    "comparative": comparative_qa,
    "inferential": inferential_qa,
    "multi-hop": multihop_qa
}

DIFFICULTY = {
    "factual": "easy",
    "comparative": "medium",
    "inferential": "medium",
    "multi-hop": "hard"
}

# ---------------- MAIN ----------------
def main():
    docs = []
    for f in os.listdir(INPUT_DIR):
        if f.endswith(".json"):
            with open(os.path.join(INPUT_DIR, f), encoding="utf-8") as file:
                doc = json.load(file)
                doc["_sentences"] = split_sentences(doc["text"])
                docs.append(doc)

    random.shuffle(docs)

    questions = []
    counts = {k: 0 for k in TARGETS}
    seen = set()
    q_id = 1

    print("🔧 Generating rule-based questions...")
    for category, target in TARGETS.items():
        gen_fn = GEN_MAP[category]

        for doc in tqdm(docs, desc=f"{category.capitalize()}"):
            if counts[category] >= target:
                break

            qa = gen_fn(doc)
            if not qa:
                continue

            if qa["question"].lower() in seen:
                continue

            # ---- LLM REWRITE (SAFE)
            rewritten_q = rewrite_question_llm(
                qa["question"], qa["answer"]
            )

            questions.append({
                "id": q_id,
                "question": rewritten_q,
                "ground_truth_answer": qa["answer"],
                "gold_urls": [doc["url"]],
                "category": category,
                "difficulty": DIFFICULTY[category]
            })

            seen.add(rewritten_q.lower())
            counts[category] += 1
            q_id += 1

            if len(questions) >= TOTAL_QUESTIONS:
                break
    # ---------------- FINAL BACKFILL (GUARANTEE 100) ----------------
    print("\n🔁 Backfilling remaining questions with factual QA...")

    for doc in tqdm(docs, desc="Backfill"):
        if len(questions) >= TOTAL_QUESTIONS:
            break

        qa = factual_qa(doc)
        if not qa:
            continue

        rewritten_q = rewrite_question_llm(
            qa["question"], qa["answer"]
        )
        if rewritten_q.lower() in seen:
            continue

        questions.append({
            "id": q_id,
            "question": rewritten_q,
            "ground_truth_answer": qa["answer"],
            "gold_urls": [doc["url"]],
            "category": "factual",
            "difficulty": "easy"
        })

        seen.add(rewritten_q.lower())
        counts["factual"] += 1
        q_id += 1

    # ---------------- SAVE ----------------
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        json.dump(questions, out, indent=2, ensure_ascii=False)

    print("\n✅ DONE")
    print("Category counts:", counts)
    print(f"Total questions: {len(questions)}")

if __name__ == "__main__":
    main()
