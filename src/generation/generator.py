from transformers import AutoTokenizer, AutoModelForCausalLM


class Generator:
    """Generate answers from retrieved context."""

    def __init__(self):
        print("Loading generation model...")

        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )

    def generate(self, query: str, context: str) -> str:
        """Generate answers from retrieved context."""

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research assistant answering "
                    "questions about a research paper. "
                    "Use only the provided context. "
                    "Do not use outside knowledge. "
                    "Answer directly and concisely. "
                    "Every factual claim must be supported by the context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{query}\n\n"
                    "Give a complete answer in 1-2 sentences. "
                    "Give only the answer. "
                    "Do not include a Sources section or citation list. "
                    "Do not copy large portions of the context."
                ),
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.2,
        )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]

        answer = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return answer.strip()