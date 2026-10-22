from transformers import AutoTokenizer, AutoModelForCausalLM


class FaithfulnessEvaluator:
    """Evaluate faithfulness and correctness of generated answers."""

    def __init__(self):
        self.model_name = "Qwen/Qwen2.5-0.5B-Instruct"

        print("Loading evaluation model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name
        )

    def _generate(self, messages, max_new_tokens=10):

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
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

        generated_tokens = outputs[0][
            inputs["input_ids"].shape[1]:
        ]

        result = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return result.strip()

    def evaluate(
        self,
        context: str,
        answer: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are evaluating the factual faithfulness "
                    "of an answer to a research paper."
                ),
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Answer:
{answer}

Determine whether the generated answer correctly answers
the question and is consistent with the expected answer.

The generated answer does not need to contain every detail
from the expected answer. A concise answer can still be
CORRECT if it captures the central answer to the question.

Respond with exactly one word:

SUPPORTED

or

NOT_SUPPORTED
""",
            },
        ]

        return self._generate(
            messages,
            max_new_tokens=10,
        )

    def evaluate_correctness(
        self,
        question: str,
        expected_answer: str,
        answer: str,
    ) -> str:

        messages = [
            {
                "role": "system",
                "content": (
                    "You are evaluating the correctness of an "
                    "answer to a research paper question."
                ),
            },
            {
                "role": "user",
                "content": f"""
Question:
{question}

Expected answer:
{expected_answer}

Generated answer:
{answer}

Determine whether the generated answer correctly answers
the question and conveys the key information in the expected
answer.

Respond with exactly one word:

CORRECT

or

INCORRECT
""",
            },
        ]

        return self._generate(
            messages,
            max_new_tokens=10,
        )