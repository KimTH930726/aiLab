"""
LLM 보조 기능: 쿼리 확장 및 동적 Critic 기준 생성.

- tool-capable 모델 (GPT 등): result_type=BaseModel로 구조화된 JSON 출력
- no-tool 모델 (BitNet 등): 줄바꿈 텍스트 출력 후 직접 파싱
LLM 호출 실패 시 빈 값을 반환하여 caller가 fallback 처리할 수 있도록 한다.
"""
from __future__ import annotations

import re
from typing import Optional

try:
    from pydantic import BaseModel
    from pydantic_ai import Agent
    _PYDANTIC_AI_AVAILABLE = True
except ImportError:
    _PYDANTIC_AI_AVAILABLE = False

# agent.py와 동일한 기준 — Function Calling 미지원 모델
_LOCAL_NO_TOOL_MODELS = {"bitnet", "local"}


def _is_no_tool(model_name: str) -> bool:
    return model_name.split(":")[-1].lower() in _LOCAL_NO_TOOL_MODELS


def _parse_lines(text: str, max_items: int) -> list[str]:
    """줄바꿈 구분 텍스트에서 항목 추출. 번호·기호 제거."""
    lines = []
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        # 앞쪽 번호/기호 제거: "1. ", "- ", "• ", "1) " 등
        line = re.sub(r'^[\d]+[.)]\s*|^[-•*]\s*', '', line).strip()
        if line:
            lines.append(line)
    return lines[:max_items]


class ExpandedQueries(BaseModel):
    """쿼리 확장 결과: 원본 쿼리의 3~4가지 검색 변형."""
    queries: list[str]


class EvalCriteria(BaseModel):
    """동적 채점 기준: 답변이 반드시 다뤄야 할 토픽 목록."""
    expected_topics: list[str]


def expand_query_with_llm(query: str, model_name: str) -> list[str]:
    """
    LLM으로 검색 쿼리를 3~4개의 변형으로 확장.

    실패 시 빈 리스트 반환 → caller가 heuristic fallback 처리.
    """
    if not _PYDANTIC_AI_AVAILABLE or not model_name:
        return []

    try:
        if _is_no_tool(model_name):
            # BitNet 등 tool calling 미지원 → 줄바꿈 텍스트 파싱
            agent: Agent = Agent(
                model_name,
                system_prompt=(
                    "당신은 검색 쿼리 최적화 전문가입니다. "
                    "사용자 쿼리의 다양한 검색 변형을 생성합니다."
                ),
            )
            prompt = (
                f"다음 쿼리의 검색 변형 3개를 생성하세요.\n"
                f"쿼리: {query}\n\n"
                f"조건:\n"
                f"- 각 변형은 다른 각도(동의어, 관련 개념, 구체화 등)로 접근\n"
                f"- 쿼리와 같은 언어로 작성\n"
                f"- 번호와 변형 텍스트만 출력 (설명 없이)\n"
                f"형식:\n1. 변형1\n2. 변형2\n3. 변형3"
            )
            result = agent.run_sync(prompt)
            text = result.output if hasattr(result, "output") else result.data
            return _parse_lines(text, 4)
        else:
            # GPT 등 tool-capable 모델 → result_type으로 구조화 출력
            agent_typed: Agent[None, ExpandedQueries] = Agent(
                model_name,
                result_type=ExpandedQueries,
                system_prompt=(
                    "You are a search query optimizer. "
                    "Generate 3-4 semantically diverse variations of the user's query "
                    "to maximize document retrieval coverage. "
                    "Respond in the same language as the user's query."
                ),
            )
            result = agent_typed.run_sync(
                f"Generate 3-4 search query variations for: {query}\n"
                f"Each variation should approach the topic from a different angle "
                f"(synonyms, broader/narrower scope, related concepts)."
            )
            output = result.output if hasattr(result, "output") else result.data
            qs = [q.strip() for q in output.queries if q.strip()]
            return qs[:4] if qs else []
    except Exception:
        return []


def generate_eval_criteria(query: str, model_name: str) -> list[str]:
    """
    LLM으로 쿼리 기반 채점 기준(예상 토픽 목록) 생성.

    실패 시 빈 리스트 반환 → caller가 REQUIRED_KEYWORDS fallback 처리.
    """
    if not _PYDANTIC_AI_AVAILABLE or not model_name:
        return []

    try:
        if _is_no_tool(model_name):
            # BitNet 등 tool calling 미지원 → 줄바꿈 텍스트 파싱
            agent: Agent = Agent(
                model_name,
                system_prompt=(
                    "당신은 RAG 시스템의 평가 기준 설계 전문가입니다. "
                    "사용자 쿼리에 대한 고품질 답변이 반드시 포함해야 할 핵심 토픽을 나열합니다."
                ),
            )
            prompt = (
                f"다음 쿼리에 대한 완전한 답변이 반드시 다뤄야 할 핵심 키워드/토픽 5~8개를 나열하세요.\n"
                f"쿼리: {query}\n\n"
                f"조건:\n"
                f"- 영어 단어 또는 짧은 구문 (1~3단어)\n"
                f"- 번호와 키워드만 출력 (설명 없이)\n"
                f"형식:\n1. keyword1\n2. keyword2\n3. keyword3"
            )
            result = agent.run_sync(prompt)
            text = result.output if hasattr(result, "output") else result.data
            return _parse_lines(text, 10)
        else:
            # tool-capable 모델 → result_type으로 구조화 출력
            agent_typed: Agent[None, EvalCriteria] = Agent(
                model_name,
                result_type=EvalCriteria,
                system_prompt=(
                    "You are an evaluation criteria designer for RAG systems. "
                    "Given a user query, list the key topics or concepts "
                    "that a high-quality answer MUST cover. "
                    "Be specific but concise (5-10 topics max). "
                    "Use simple English keywords that can be found via substring matching."
                ),
            )
            result = agent_typed.run_sync(
                f"User query: {query}\n\n"
                f"List the essential topics/keywords a comprehensive answer must include. "
                f"Output as a list of short English terms (1-3 words each)."
            )
            output = result.output if hasattr(result, "output") else result.data
            topics = [t.strip() for t in output.expected_topics if t.strip()]
            return topics[:10] if topics else []
    except Exception:
        return []
