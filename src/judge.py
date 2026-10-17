"""Evaluation of generated learning paths.

Two complementary layers:
  * Programmatic checks (objective, cheap, no model): validity, constraint
    satisfaction, hallucinated-course detection, redundancy.
  * LLM-as-judge (gpt-4o): rubric scoring of relevance, ordering coherence and
    coverage on a 1-5 scale, blind to which system produced the path.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

from db import PlaforEduDB

JUDGE_MODEL = "gpt-4o"


# --------------------------------------------------------------------------- #
# Programmatic checks
# --------------------------------------------------------------------------- #
def programmatic_metrics(db: PlaforEduDB, record: dict) -> dict:
    candidates = db.get_courses_for_competency(record["competency_id"])
    valid_ids = {c["id"] for c in candidates}
    selected_ids = [c["id"] for c in record["selected_courses"]]

    n = len(selected_ids)
    hallucinated = [cid for cid in selected_ids if cid not in valid_ids]
    unique_ids = set(selected_ids)
    return {
        "n_selected": n,
        "meets_min_courses": n >= 3,
        "n_hallucinated": len(hallucinated),
        "hallucination_rate": (len(hallucinated) / n) if n else 0.0,
        "n_duplicates": n - len(unique_ids),
        "selection_ratio": (n / len(candidates)) if candidates else 0.0,
    }


# --------------------------------------------------------------------------- #
# LLM-as-judge
# --------------------------------------------------------------------------- #
JUDGE_PROMPT = """Você é um avaliador especialista em design instrucional e curadoria de trilhas formativas para servidores públicos. Avalie de forma rigorosa e imparcial a trilha (sequência ordenada de cursos) proposta para atingir um objetivo de aprendizagem.

# Objetivo da trilha #
{objective}

# Trilha proposta (na ordem apresentada) #
{path}

# Critérios (nota inteira de 1 a 5, onde 1 = muito ruim e 5 = excelente) #
- relevance: os cursos selecionados são pertinentes ao objetivo? Penalize cursos irrelevantes e a ausência de temas essenciais.
- ordering: a ordem dos cursos forma uma progressão pedagógica coerente (fundamentos antes de tópicos avançados, pré-requisitos respeitados)?
- coverage: a trilha cobre adequadamente a competência sem ser excessivamente longa ou redundante?
- overall: qualidade geral da trilha como percurso formativo.

Seja criterioso: uma trilha que simplesmente lista todos os cursos sem curadoria não deve receber notas altas em ordering e coverage."""


class Verdict(BaseModel):
    relevance: int = Field(ge=1, le=5)
    ordering: int = Field(ge=1, le=5)
    coverage: int = Field(ge=1, le=5)
    overall: int = Field(ge=1, le=5)
    rationale: str = Field(default="")


_judge = None


def _get_judge():
    global _judge
    if _judge is None:
        _judge = ChatPromptTemplate.from_template(JUDGE_PROMPT) | ChatOpenAI(
            model=JUDGE_MODEL, temperature=0.0
        ).with_structured_output(Verdict)
    return _judge


def format_path(db: PlaforEduDB, record: dict) -> str:
    lines = []
    for i, c in enumerate(record["selected_courses"], 1):
        course = db.get_course(c["id"])
        desc = course["description"] if course else "(curso inexistente no catálogo)"
        name = course["name"] if course else c.get("name", "?")
        lines.append(f"{i}. {name}\n   {desc}")
    return "\n".join(lines) if lines else "(trilha vazia)"


def judge_path(db: PlaforEduDB, record: dict) -> dict:
    if not record["selected_courses"]:
        return {"relevance": 1, "ordering": 1, "coverage": 1, "overall": 1,
                "rationale": "trilha vazia", "judge_prompt_tokens": 0,
                "judge_completion_tokens": 0}
    chain = _get_judge()
    with get_openai_callback() as cb:
        v: Verdict = chain.invoke(
            {"objective": record["objective"], "path": format_path(db, record)}
        )
    d = v.model_dump()
    d["judge_prompt_tokens"] = cb.prompt_tokens
    d["judge_completion_tokens"] = cb.completion_tokens
    return d


def evaluate_record(db: PlaforEduDB, record: dict) -> dict:
    out = dict(record)
    out.pop("selected_courses_full", None)
    out.update(programmatic_metrics(db, record))
    out.update(judge_path(db, record))
    return out
