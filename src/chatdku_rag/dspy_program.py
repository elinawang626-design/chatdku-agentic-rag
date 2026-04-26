from __future__ import annotations

import dspy


class RouteQuestion(dspy.Signature):
    """Decide whether to use local documents only or also supplement with internet search."""

    question = dspy.InputField()
    language = dspy.InputField()
    local_hits_found = dspy.InputField(desc="yes or no")
    allow_internet = dspy.InputField(desc="yes or no")
    route = dspy.OutputField(desc="one of local_only, local_plus_internet, or internet_only")


class AnswerFromEvidence(dspy.Signature):
    """Answer a university advising question using only the provided evidence and cite source ids."""

    question = dspy.InputField()
    language = dspy.InputField()
    evidence = dspy.InputField()
    answer = dspy.OutputField(
        desc="A concise answer grounded in the evidence. Include citations like [1] or [2] where helpful."
    )


class DSPyRAGProgram:
    def __init__(self, lm: dspy.LM) -> None:
        self.lm = lm
        self.router = dspy.Predict(RouteQuestion)
        self.answerer = dspy.ChainOfThought(AnswerFromEvidence)

    def route(self, question: str, language: str, local_hits_found: bool, allow_internet: bool) -> str:
        prediction = self.router(
            question=question,
            language=language,
            local_hits_found="yes" if local_hits_found else "no",
            allow_internet="yes" if allow_internet else "no",
        )
        route = (prediction.route or "").strip().lower()
        if route in {"local_only", "local_plus_internet", "internet_only"}:
            return route
        if local_hits_found:
            return "local_only"
        return "internet_only" if allow_internet else "local_only"

    def answer(self, question: str, language: str, evidence: str) -> str:
        prediction = self.answerer(
            question=question,
            language=language,
            evidence=evidence,
        )
        return (prediction.answer or "").strip()
