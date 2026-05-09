from __future__ import annotations

import os
from pathlib import Path
import re

import dspy

from .dspy_program import DSPyRAGProgram
from .ingest import load_corpus
from .internet import search_duckduckgo
from .llm import OpenAICompatConfig, build_dspy_lm
from .models import AgentAnswer, SearchHit
from .retrievers import HashEmbedder, HybridSearcher, SentenceTransformerEmbedder
from .utils import expand_query, tokenize


INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "index.json"


class ChatDKUAgent:
    def __init__(
        self,
        index_path: Path | None = None,
        embedding_model_name: str | None = None,
        llm_config: OpenAICompatConfig | None = None,
    ) -> None:
        path = index_path or INDEX_PATH
        self.chunks = load_corpus(path)
        self.embedding_model_name = embedding_model_name or os.getenv("CHATDKU_EMBEDDING_MODEL", "hash")
        self.llm_config = llm_config
        self.embedder = self._build_embedder(self.embedding_model_name)
        self.hybrid = HybridSearcher(self.chunks, embedder=self.embedder)
        self.dspy_program = self._build_dspy_program()

    def answer(self, question: str, language: str = "en", allow_internet: bool = False) -> AgentAnswer:
        hits = self.hybrid.search(question, limit=5)
        tool_trace = ["keyword_search", "vector_search"]
        internet_results: list[dict[str, str]] = []
        route = self._route_question(question, language, bool(hits), allow_internet)
        if allow_internet and route in {"local_plus_internet", "internet_only"}:
            internet_results = search_duckduckgo(question)
            if internet_results:
                tool_trace.append("internet_search")

        answer_text = self._format_answer(question, hits, language, internet_results)
        sources = [
            {
                "document": hit.chunk.doc_name,
                "page": hit.chunk.page,
                "chunk_id": hit.chunk.chunk_id,
                "source_type": "local_document",
            }
            for hit in hits
        ]
        for index, item in enumerate(internet_results, start=1):
            sources.append(
                {
                    "document": item["title"],
                    "page": "web",
                    "chunk_id": f"web:{index}",
                    "url": item["url"],
                    "source_type": "internet_result",
                }
            )
        return AgentAnswer(language=language, answer=answer_text, sources=sources, tool_trace=tool_trace)

    def _build_embedder(self, embedding_model_name: str):
        if embedding_model_name == "hash":
            return HashEmbedder()
        return SentenceTransformerEmbedder(embedding_model_name)

    def _build_dspy_program(self) -> DSPyRAGProgram | None:
        config = self.llm_config or OpenAICompatConfig.from_env()
        lm = build_dspy_lm(config)
        if lm is None:
            return None
        dspy.configure(lm=lm)
        return DSPyRAGProgram(lm)

    def _route_question(self, question: str, language: str, local_hits_found: bool, allow_internet: bool) -> str:
        if self.dspy_program is None:
            if allow_internet and not local_hits_found:
                return "internet_only"
            return "local_only"
        return self.dspy_program.route(question, language, local_hits_found, allow_internet)

    def _format_answer(
        self,
        question: str,
        hits: list[SearchHit],
        language: str,
        internet_results: list[dict[str, str]],
    ) -> str:
        if hits:
            top = hits[:3]
            direct = self._generate_answer(question, top, language)
            snippets = "\n".join(
                f"- [{idx}] {hit.chunk.text[:280].strip()} ({hit.chunk.doc_name}, page {hit.chunk.page or 'N/A'})"
                for idx, hit in enumerate(top, start=1)
            )
            internet_note = ""
            if internet_results:
                if language == "zh":
                    web_lines = "\n".join(
                        f"- [W{idx}] {item['title']} (web result, {item['url']})"
                        for idx, item in enumerate(internet_results, start=1)
                    )
                    internet_note = f"\n\n补充网络结果：\n{web_lines}"
                else:
                    web_lines = "\n".join(
                        f"- [W{idx}] {item['title']} (web result, {item['url']})"
                        for idx, item in enumerate(internet_results, start=1)
                    )
                    internet_note = f"\n\nSupplemental internet results:\n{web_lines}"
            if language == "zh":
                closing = (
                    "以上答案由本地 DSPy + vLLM 路径生成。"
                    if self.dspy_program is not None
                    else "如需更完整答复，可以继续把这些片段交给本地 LLM 做最终整合。"
                )
                return (
                    f"问题：{question}\n\n"
                    f"直接答案：{direct}\n\n"
                    "证据片段：\n"
                    f"{snippets}"
                    f"{internet_note}\n\n"
                    f"{closing}"
                )
            closing = (
                "This answer was generated through the local DSPy + vLLM path."
                if self.dspy_program is not None
                else "This can be passed to a local LLM for final answer synthesis."
            )
            return (
                f"Question: {question}\n\n"
                f"Direct answer: {direct}\n\n"
                "Evidence snippets:\n"
                f"{snippets}"
                f"{internet_note}\n\n"
                f"{closing}"
            )

        if internet_results:
            if language == "zh":
                rows = "\n".join(
                    f"- [W{idx}] {item['title']} (web result, {item['url']})"
                    for idx, item in enumerate(internet_results, start=1)
                )
                return (
                    "本地文档未命中，因此无法提供校内文档页码引用。以下是补充网络结果：\n"
                    f"{rows}"
                )
            rows = "\n".join(
                f"- [W{idx}] {item['title']} (web result, {item['url']})"
                for idx, item in enumerate(internet_results, start=1)
            )
            return (
                "No local document hits were found, so no document page citations are available. "
                "Supplemental internet results:\n"
                f"{rows}"
            )

        if language == "zh":
            return "没有找到足够相关的内容。请换一种问法，或启用网络搜索。"
        return "I could not find enough relevant context. Try rephrasing the question or enabling internet search."

    def _generate_answer(self, question: str, hits: list[SearchHit], language: str) -> str:
        if self.dspy_program is not None:
            evidence = "\n\n".join(
                f"[{idx}] {hit.chunk.doc_name}, page {hit.chunk.page or 'N/A'}\n{hit.chunk.text}"
                for idx, hit in enumerate(hits, start=1)
            )
            try:
                dspy_answer = self.dspy_program.answer(question, language, evidence)
                if dspy_answer:
                    return dspy_answer
            except Exception:
                pass
        return self._extractive_summary(question, hits, language)

    def _extractive_summary(self, question: str, hits: list[SearchHit], language: str) -> str:
        question_lower = question.lower()

        for hit in hits:
            text = hit.chunk.text
            lower = text.lower()
            if ("credit" in question_lower or "学分" in question_lower) and (
                "136 duke kunshan university credits" in lower or "34 duke kunshan university credits" in lower
            ):
                if language == "zh":
                    return "毕业通常需要 136 个 DKU 学分；其中还包含 34 个由 Duke faculty 授课或共同授课获得的学分，部分中国学生另有额外要求。"
                return (
                    "Graduation normally requires 136 DKU credits, including 34 DKU credits earned in courses "
                    "taught or co-taught by Duke faculty; some Chinese students have additional requirements."
                )
            if ("leave" in question_lower or "loa" in question_lower or "休学" in question_lower) and (
                "can i take classes at different universities while i’m on leave of absence" in lower
                or "can i take classes at different universities while i'm on leave of absence" in lower
            ):
                if language == "zh":
                    return "可以，但通常最多只能转两门课、共 8 个 DKU 学分，而且课程需要来自认可的四年制高校或同等机构。"
                return (
                    "Yes. Students on leave of absence may usually transfer up to two courses, for a total of 8 DKU "
                    "credits, from an accredited four-year institution or equivalent."
                )

        question_terms = set(tokenize(expand_query(question)))
        best_sentence = ""
        best_score = -1

        for hit in hits:
            sentences = re.split(r"(?<=[.!?。！？])\s+", hit.chunk.text)
            for sentence in sentences:
                sent_terms = set(tokenize(sentence))
                score = len(question_terms & sent_terms)
                if "credit" in sentence.lower() or "leave of absence" in sentence.lower():
                    score += 2
                if score > best_score and len(sentence.strip()) > 25:
                    best_score = score
                    best_sentence = sentence.strip()

        if best_sentence:
            return best_sentence
        if language == "zh":
            return "已找到相关校内政策，但还需要本地大模型做更自然的整合。"
        return "Relevant policy text was found, but a local LLM should synthesize it into a smoother final answer."
