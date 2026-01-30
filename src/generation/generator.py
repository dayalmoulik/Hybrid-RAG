import time
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"
MAX_CONTEXT_TOKENS = 1024
MAX_OUTPUT_TOKENS = 256


class ResponseGenerator:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        self.model.to(self.device)
        self.model.eval()

    def build_prompt(self, query, contexts):
        """
        Build instruction-style prompt from retrieved chunks
        """
        context_text = "\n\n".join(
            [f"[Source {i+1}] {c['text']}" for i, c in enumerate(contexts)]
        )

        prompt = (
            "Answer the question using ONLY the information provided in the context.\n"
            "If the answer is not contained in the context, say \"I don't know.\".\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

        return prompt

    def generate(self, query, contexts):
        """
        Generate answer from query + retrieved contexts
        """
        prompt = self.build_prompt(query, contexts)

        inputs = self.tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_CONTEXT_TOKENS,
            return_tensors="pt"
        ).to(self.device)

        start_time = time.time()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                do_sample=False
            )

        end_time = time.time()

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        return {
            "answer": answer.strip(),
            "response_time": round(end_time - start_time, 3)
        }


if __name__ == "__main__":
    # Minimal test
    generator = ResponseGenerator()

    fake_contexts = [
        {"text": "Retrieval-Augmented Generation (RAG) combines retrieval and generation."},
        {"text": "It uses external documents to ground language model responses."}
    ]

    query = "What is Retrieval-Augmented Generation?"

    result = generator.generate(query, fake_contexts)

    print("Answer:", result["answer"])
    print("Time:", result["response_time"], "seconds")
