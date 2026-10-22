from src.evaluation.faithfulness import FaithfulnessEvaluator


evaluator = FaithfulnessEvaluator()

context = """
DREEP provides an in silico framework to prioritize drugs
from single-cell transcriptional profiles of tumours.
It helps in designing personalized treatment strategies
and accelerating drug repurposing studies.
"""

supported_answer = (
    "DREEP provides an in silico framework to prioritize "
    "drugs from single-cell transcriptional profiles of tumours."
)

unsupported_answer = (
    "DREEP was developed to treat infectious diseases."
)

print("=" * 70)
print("SUPPORTED ANSWER TEST")
print("=" * 70)

result = evaluator.evaluate(
    context,
    supported_answer,
)

print("Result:", result)

print("\n")
print("=" * 70)
print("UNSUPPORTED ANSWER TEST")
print("=" * 70)

result = evaluator.evaluate(
    context,
    unsupported_answer,
)

print("Result:", result)