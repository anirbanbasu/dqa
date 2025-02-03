from typing import Optional
import dspy


class QASignature(dspy.Signature):
    """
    Given a question, the model will attempt to answer it.
    """

    question: str = dspy.InputField(description="The question to be answered.")
    reasoning: Optional[str] = dspy.OutputField(
        description="The optional reasoning behind the answer. This can be empty or None."
    )
    output: str = dspy.OutputField(description="The answer to the question.")
