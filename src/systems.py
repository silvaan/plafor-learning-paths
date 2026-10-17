"""Learning-path generation systems: the proposed multi-agent workflow and baselines.

Every system exposes `run(competency_id) -> dict` returning a common record:
    {
      "system", "competency_id", "objective",
      "selected_courses": [{id, name, justification?}, ...],  # ordered
      "n_candidates", "n_selected",
      "iterations", "evaluation", "evaluation_feedback",
      "prompt_tokens", "completion_tokens", "latency_s",
    }
"""
from __future__ import annotations

import os
import time
from typing import Literal, TypedDict

import requests

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langgraph.graph import StateGraph, START, END

from db import PlaforEduDB

GEN_MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 3  # cap on the self-reflective refinement loop


def make_llm(gen_model: str = GEN_MODEL, temperature: float = 0.0):
    """Build the generator/critic LLM from a spec string.

    "gpt-4o-mini"           -> classic instruction-tuned model, temperature 0
    "gpt-5.4-mini:none"     -> same weights family with reasoning disabled (System 1)
    "gpt-5.4-mini:high"     -> reasoning enabled at high effort (System 2)
    """
    if ":" in gen_model:
        model, effort = gen_model.split(":", 1)
        return ChatOpenAI(model=model, reasoning_effort=effort)
    return ChatOpenAI(model=gen_model, temperature=temperature)

# --------------------------------------------------------------------------- #
# Prompts (preserved from the original prototype)
# --------------------------------------------------------------------------- #
CONTEXT_PROMPT = (
    "<context>O PlaforEdu é um sistema de curadoria de cursos para formação "
    "continuada de servidores da rede pública. Trilhas são percursos formativos "
    "propostos a partir do encadeamento de cursos ordenados, com o objetivo de "
    "desenvolver uma determinada competência com um determinado objetivo.</context>\n\n"
)

FILTER_COURSES_PROMPT = """# Instruções #
Você é um especialista em curadoria de cursos para estruturação de trilhas formativas. Seu objetivo é desenvolver estas trilhas de acordo com os cursos disponíveis e um objetivo fornecido.

# Informações #
- Considere que a plataforma onde os cursos estão cadastrados está organizada em itinerários formativos.
- Cada itinerário formativo possui um determinado número de competências associadas.
- Cada competência possui um descritor específico, que expressa os conhecimentos, habilidades e atitudes que serão desenvolvidos.
- Cada competência gera uma trilha formativa.
- Considere que as trilhas são percursos formativos propostos a partir do encadeamento de cursos ordenados, com o objetivo de desenvolver a competência contemplada naquela Trilha.
- O descritor de cada competência expressa o objetivo da Trilha formativa.
- Você deve identificar e avaliar cursos que atendam às demandas de capacitação, formação ou atualização, no ambiente da plataforma, apenas com o seguinte objetivo.
- Organize os cursos na ordem que achar mais adequada para atender ao objetivo proposto.

# Objetivo #
{objective}

# Cursos Disponíveis #
{available_courses}
"""

EVALUATE_COURSES_PROMPT = """# Instruções #
Você é um avaliador em curadoria de cursos para estruturação de trilhas formativas.

# Regras #
- A trilha deverá estar de acordo com o objetivo proposto.
- A trilha deverá ter pelo menos {min_courses} cursos.

# Objetivo #
{objective}

# Cursos Selecionados #
{selected_courses}
"""


# --------------------------------------------------------------------------- #
# Structured-output schemas
# --------------------------------------------------------------------------- #
class Course(BaseModel):
    id: str = Field(description="O ID do curso.")
    name: str = Field(description="O nome do curso.")
    justification: str = Field(description="Justificativa para a seleção do curso.")


class Courses(BaseModel):
    courses: list[Course] = Field(description="A lista de cursos selecionados e ordenados.")


class Evaluation(BaseModel):
    result: Literal["Bom", "Ruim"] = Field(description="O resultado da avaliação.")
    feedback: str = Field(default="", description="Feedback para corrigir problemas.")


def format_courses(courses: list[dict]) -> str:
    parts = []
    for c in courses:
        block = f"ID: {c['id']}\nNome: {c['name']}\n"
        if c.get("description"):
            block += f"Descrição: {c['description']}\n"
        if c.get("justification"):
            block += f"Justificativa: {c['justification']}\n"
        parts.append(block)
    return "\n".join(parts)


def objective_for(db: PlaforEduDB, competency_id: str) -> tuple[str, str, list[dict]]:
    comp = db.get_competency(competency_id)
    courses = db.get_courses_for_competency(competency_id)
    # The competency descriptor is the learning objective of the trilha.
    objective = f"{comp['name']}: {comp['description']}"
    return comp["name"], objective, courses


# --------------------------------------------------------------------------- #
# System 1 — Naive baseline: keep every candidate course in catalog order
# --------------------------------------------------------------------------- #
def run_naive(db: PlaforEduDB, competency_id: str) -> dict:
    name, objective, courses = objective_for(db, competency_id)
    t0 = time.perf_counter()
    selected = [{"id": c["id"], "name": c["name"], "justification": ""} for c in courses]
    return _record(
        "naive_all", competency_id, objective, selected, len(courses),
        iterations=0, evaluation="", feedback="",
        pt=0, ct=0, latency=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------- #
# System 2 — Retrieval baseline: embed objective + courses, take top-k
# --------------------------------------------------------------------------- #
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = OpenAIEmbeddings(model="text-embedding-3-small")
    return _embedder


def run_retrieval(db: PlaforEduDB, competency_id: str, k: int = 5) -> dict:
    import numpy as np

    name, objective, courses = objective_for(db, competency_id)
    t0 = time.perf_counter()
    emb = _get_embedder()
    texts = [f"{c['name']}. {c['description']}" for c in courses]
    with get_openai_callback() as cb:
        qv = np.array(emb.embed_query(objective))
        dvs = np.array(emb.embed_documents(texts))
    sims = dvs @ qv / (np.linalg.norm(dvs, axis=1) * np.linalg.norm(qv) + 1e-9)
    order = np.argsort(-sims)[: min(k, len(courses))]
    selected = [
        {"id": courses[i]["id"], "name": courses[i]["name"], "justification": ""}
        for i in order
    ]
    return _record(
        "retrieval_topk" if k == 5 else f"retrieval_k{k}", competency_id, objective, selected, len(courses),
        iterations=0, evaluation="", feedback="",
        pt=cb.prompt_tokens, ct=cb.completion_tokens, latency=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------- #
# System 3 — Single-agent (one generator call, no critic loop)  [ablation]
# --------------------------------------------------------------------------- #
MAX5_RULE = "\n# Restrição #\nSelecione no máximo 5 cursos.\n"


def run_single_agent(db: PlaforEduDB, competency_id: str, gen_model: str = GEN_MODEL,
                     temperature: float = 0.0, max5: bool = False) -> dict:
    name, objective, courses = objective_for(db, competency_id)
    llm = make_llm(gen_model, temperature)
    prompt = CONTEXT_PROMPT + FILTER_COURSES_PROMPT + (MAX5_RULE if max5 else "")
    chain = ChatPromptTemplate.from_template(prompt) | llm.with_structured_output(Courses)
    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        resp = chain.invoke(
            {"objective": objective, "available_courses": format_courses(courses)}
        )
    selected = [c.model_dump() for c in resp.courses]
    return _record(
        "single_agent_max5" if max5 else "single_agent", competency_id, objective, selected, len(courses),
        iterations=1, evaluation="", feedback="",
        pt=cb.prompt_tokens, ct=cb.completion_tokens, latency=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------- #
# System 4 — Proposed multi-agent workflow with self-reflective critic loop
# --------------------------------------------------------------------------- #
class State(TypedDict, total=False):
    objective: str
    competency_id: str
    available_courses: list[dict]
    selected_courses: list[dict]
    evaluation: str
    evaluation_feedback: str
    iterations: int
    gate_fixes: int


def id_gate(selected: list[dict], pool: list[dict]) -> tuple[list[dict], int]:
    """Repair identifiers whose name matches a pool course; drop the rest.

    Returns the gated path and the number of entries that were repaired or dropped.
    """
    by_id = {c["id"]: c for c in pool}
    by_name = {c["name"].strip().lower(): c for c in pool}
    out, seen, touched = [], set(), 0
    for c in selected:
        hit = by_id.get(c["id"]) or by_name.get(c.get("name", "").strip().lower())
        if hit is None or hit["id"] in seen:
            touched += 1
            continue
        if hit["id"] != c["id"]:
            touched += 1
        seen.add(hit["id"])
        out.append({**c, "id": hit["id"], "name": hit["name"]})
    return out, touched


def build_graph(db: PlaforEduDB, gen_model: str = GEN_MODEL, temperature: float = 0.0,
                guarded: bool = False):
    llm = make_llm(gen_model, temperature)
    gen_chain_base = llm.with_structured_output(Courses)
    eval_chain = ChatPromptTemplate.from_template(
        CONTEXT_PROMPT + EVALUATE_COURSES_PROMPT
    ) | llm.with_structured_output(Evaluation)

    def select_courses(state: State):
        courses = db.get_courses_for_competency(state["competency_id"])
        return {"available_courses": courses, "iterations": 0, "gate_fixes": 0}

    def filter_courses(state: State):
        system_prompt = CONTEXT_PROMPT + FILTER_COURSES_PROMPT
        fb = state.get("evaluation_feedback", "")
        if fb:
            system_prompt += f"\n# Feedback #\n{fb}"
        chain = ChatPromptTemplate.from_template(system_prompt) | gen_chain_base
        resp = chain.invoke(
            {
                "objective": state["objective"],
                "available_courses": format_courses(state["available_courses"]),
            }
        )
        selected = [c.model_dump() for c in resp.courses]
        fixes = state.get("gate_fixes", 0)
        if guarded:
            selected, touched = id_gate(selected, state["available_courses"])
            fixes += touched
        return {
            "selected_courses": selected,
            "iterations": state.get("iterations", 0) + 1,
            "gate_fixes": fixes,
        }

    def evaluate_courses(state: State):
        resp = eval_chain.invoke(
            {
                "objective": state["objective"],
                "selected_courses": format_courses(state["selected_courses"]),
                # guardrail: never demand more courses than the pool holds
                "min_courses": min(3, len(state["available_courses"])) if guarded else 3,
            }
        )
        return {"evaluation": resp.result, "evaluation_feedback": resp.feedback}

    def route_courses(state: State):
        if state["evaluation"] == "Ruim" and state.get("iterations", 0) < MAX_ITERATIONS:
            return "redo"
        return "end"

    wf = StateGraph(State)
    wf.add_node("select_courses", select_courses)
    wf.add_node("filter_courses", filter_courses)
    wf.add_node("evaluate_courses", evaluate_courses)
    wf.add_edge(START, "select_courses")
    wf.add_edge("select_courses", "filter_courses")
    wf.add_edge("filter_courses", "evaluate_courses")
    wf.add_conditional_edges(
        "evaluate_courses", route_courses, {"redo": "filter_courses", "end": END}
    )
    return wf.compile()


def run_multi_agent(db: PlaforEduDB, competency_id: str, gen_model: str = GEN_MODEL,
                    temperature: float = 0.0, guarded: bool = False) -> dict:
    name, objective, courses = objective_for(db, competency_id)
    graph = build_graph(db, gen_model, temperature, guarded)
    t0 = time.perf_counter()
    with get_openai_callback() as cb:
        final = graph.invoke({"competency_id": competency_id, "objective": objective})
    rec = _record(
        "multi_agent_guarded" if guarded else "multi_agent", competency_id, objective,
        final["selected_courses"], len(courses),
        iterations=final.get("iterations", 0),
        evaluation=final.get("evaluation", ""),
        feedback=final.get("evaluation_feedback", ""),
        pt=cb.prompt_tokens, ct=cb.completion_tokens, latency=time.perf_counter() - t0,
    )
    rec["gate_fixes"] = final.get("gate_fixes", 0)
    return rec


# --------------------------------------------------------------------------- #
# System 6 — Hybrid retrieve -> generate -> verify  [the design proposed in v1]
# --------------------------------------------------------------------------- #
def run_hybrid(db: PlaforEduDB, competency_id: str, gen_model: str = GEN_MODEL,
               temperature: float = 0.0, shortlist: int = 10) -> dict:
    """Embedding retrieval shortlists, one LLM call selects/orders/justifies, a
    programmatic gate keeps only real pool courses (no critic loop)."""
    name, objective, courses = objective_for(db, competency_id)
    t0 = time.perf_counter()
    retr = run_retrieval(db, competency_id, k=shortlist)
    ids = [c["id"] for c in retr["selected_courses"]]
    by_id = {c["id"]: c for c in courses}
    short = [by_id[i] for i in ids]
    llm = make_llm(gen_model, temperature)
    chain = ChatPromptTemplate.from_template(
        CONTEXT_PROMPT + FILTER_COURSES_PROMPT
    ) | llm.with_structured_output(Courses)
    with get_openai_callback() as cb:
        resp = chain.invoke({"objective": objective, "available_courses": format_courses(short)})
    selected, touched = id_gate([c.model_dump() for c in resp.courses], short)
    floor = min(3, len(short))
    if len(selected) < floor:  # pool-aware floor: pad from the shortlist, never invent
        have = {c["id"] for c in selected}
        selected += [{"id": c["id"], "name": c["name"], "justification": "(complemento por similaridade)"}
                     for c in short if c["id"] not in have][: floor - len(selected)]
    rec = _record(
        "hybrid", competency_id, objective, selected, len(courses),
        iterations=1, evaluation="", feedback="",
        pt=cb.prompt_tokens, ct=cb.completion_tokens, latency=time.perf_counter() - t0,
    )
    rec["gate_fixes"] = touched
    return rec


# --------------------------------------------------------------------------- #
# System 5 — System One decision model (Jev-style, non-autoregressive)
# --------------------------------------------------------------------------- #
# Any server speaking TypeSafe's /v1/systemone protocol works here: the hosted
# Jev, or the open-source Kev and Laya servers run locally. Each candidate course
# gets two typed decisions in one forward pass (a yes/no relevance probability and
# an ordinal depth level); the path is assembled deterministically from them, so
# no identifier is ever generated.
S1_ENDPOINTS = {
    "laya": (os.environ.get("LAYA_URL", "http://127.0.0.1:8000"), "laya-latest"),
    "kev4b": (os.environ.get("KEV_URL", "http://127.0.0.1:8009"), "kev-latest"),
    "jev": ("https://api.typesafe.ai", "jev-latest"),
}
S1_LABEL = {"laya": "laya-multilingual", "kev4b": "kev-4b", "jev": "jev"}
S1_QUESTIONS = {
    "relevant": {
        "type": "noul",
        "instructions": "O curso candidato contribui diretamente para desenvolver a competência descrita no objetivo da trilha?",
        "criteria": {
            "true": "O conteúdo do curso trata de conhecimentos ou habilidades centrais da competência",
            "false": "O curso trata de tema periférico ou não relacionado à competência",
        },
    },
    "level": {
        "type": "score",
        "instructions": "Em que ponto de uma trilha formativa este curso deve ser cursado?",
        "criteria": [
            "Início: fundamentos ou introdução ao tema",
            "Meio: aprofundamento intermediário",
            "Fim: tópico avançado ou especializado",
        ],
    },
}
S1_THRESHOLD = 0.5


def s1_state(objective: str, course: dict) -> str:
    return (f"Objetivo da trilha (competência): {objective}\n\n"
            f"Curso candidato: {course['name']}\nDescrição: {course['description']}")


def s1_decide(backend: str, state: str) -> tuple[dict, int]:
    url, model = S1_ENDPOINTS[backend]
    headers = {"content-type": "application/json"}
    if backend == "jev":
        headers["authorization"] = f"Bearer {os.environ['TYPESAFE_API_KEY']}"
    r = requests.post(f"{url}/v1/systemone", headers=headers, timeout=120,
                      json={"model": model, "state": state, "questions": S1_QUESTIONS})
    r.raise_for_status()
    body = r.json()
    return body["answers"], body.get("usage", {}).get("input_tokens", 0)


def run_system_one(db: PlaforEduDB, competency_id: str, backend: str = "laya") -> dict:
    name, objective, courses = objective_for(db, competency_id)
    t0 = time.perf_counter()
    scored, tokens = [], 0
    for c in courses:
        ans, n = s1_decide(backend, s1_state(objective, c))
        tokens += n
        scored.append({"id": c["id"], "name": c["name"],
                       "p_relevant": float(ans["relevant"]["noul"]),
                       "level": float(ans["level"]["score"])})
    by_rel = sorted(scored, key=lambda x: -x["p_relevant"])
    keep = [x for x in by_rel if x["p_relevant"] >= S1_THRESHOLD]
    floor = min(3, len(courses))
    if len(keep) < floor:
        keep = by_rel[:floor]
    keep.sort(key=lambda x: (x["level"], -x["p_relevant"]))
    selected = [{"id": x["id"], "name": x["name"],
                 "justification": f"p_relevante={x['p_relevant']:.2f}; nivel={x['level']:.2f}"}
                for x in keep]
    rec = _record(
        f"s1_{backend}", competency_id, objective, selected, len(courses),
        iterations=0, evaluation="", feedback="",
        pt=tokens, ct=0, latency=time.perf_counter() - t0,
    )
    rec["s1_scores"] = scored
    return rec


# --------------------------------------------------------------------------- #
def _record(system, competency_id, objective, selected, n_candidates,
            iterations, evaluation, feedback, pt, ct, latency):
    return {
        "system": system,
        "competency_id": competency_id,
        "objective": objective,
        "selected_courses": selected,
        "n_candidates": n_candidates,
        "n_selected": len(selected),
        "iterations": iterations,
        "evaluation": evaluation,
        "evaluation_feedback": feedback,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "latency_s": round(latency, 3),
    }


# Systems whose output depends on the generator LLM (take `gen_model`).
LLM_SYSTEMS = {
    "single_agent": run_single_agent,
    "single_agent_max5": lambda db, cid, **kw: run_single_agent(db, cid, max5=True, **kw),
    "multi_agent": run_multi_agent,
    "multi_agent_guarded": lambda db, cid, **kw: run_multi_agent(db, cid, guarded=True, **kw),
    "hybrid": run_hybrid,
}
# Systems with no generator LLM; the value is the label stored as `gen_model`.
FIXED_SYSTEMS = {
    "naive_all": (run_naive, "-"),
    "retrieval_topk": (run_retrieval, "text-embedding-3-small"),
    "retrieval_k3": (lambda db, cid: run_retrieval(db, cid, k=3), "text-embedding-3-small"),
    "retrieval_k10": (lambda db, cid: run_retrieval(db, cid, k=10), "text-embedding-3-small"),
    **{f"s1_{b}": ((lambda b: lambda db, cid: run_system_one(db, cid, b))(b), S1_LABEL[b])
       for b in S1_ENDPOINTS},
}

SYSTEMS = {
    "naive_all": run_naive,
    "retrieval_topk": run_retrieval,
    "single_agent": run_single_agent,
    "multi_agent": run_multi_agent,
}
