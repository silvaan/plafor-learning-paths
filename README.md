# PlaforEdu Learning-Path Curation — Multi-Agent LLM Workflow

Automated curation of *trilhas formativas* (ordered learning paths) for the
PlaforEdu public-sector training platform, using a LangGraph multi-agent workflow
(context → generator → self-reflective critic loop).

This repository accompanies the paper
*"Fast, Slow, or Agentic? System One Decision Models, Reasoning LLMs, and
Self-Reflective Multi-Agent Curation of Learning Paths for Public-Sector Training"*
(`papers/ieee-access/access.tex`, a mirror of the Overleaf project).

## Layout
- `src/db.py`         — SQLite DAO over the course/competency catalog.
- `src/systems.py`    — all systems: naive, retrieval (k=3/5/10), single agent (+ max-5),
                        multi-agent critic loop (+ guarded: ID gate and min(3,|A|)), hybrid
                        retrieve-generate-verify, and System One curation over any
                        `/v1/systemone` server (Jev, Kev, Laya). Generator is a spec string:
                        `gpt-4o-mini`, `gpt-5.4-mini:none`, `gpt-5.4-mini:high`.
- `src/run.py`        — generation only, resumable, → `results/gen_v2.jsonl`.
- `src/rejudge.py`    — judge v2 (reason-first, anchored, pool-aware, 5 dimensions) with
                        gpt-4o and claude-opus-5-5 → `results/judgments_v2.jsonl`.
- `src/analyze_v2.py` — tables `tab2_*.tex`, figures `fig2_*.png`, macros `macros_v2.tex`.
- `src/make_human_eval.py` — blind curator workbook (`build`) and its agreement with the judges (`score`).
- `src/judge.py`, `src/analyze.py` — v1 judge and analysis of the preliminary run
                        (`results/raw_results.jsonl`, macros `macros.tex`), still used for the v1 diagnostics.
- `src/make_schematics.py` — ER diagram, workflow graph, System One pipeline figures.
- `papers/ieee-access/` — Overleaf project (`access.tex` main).

## Reproduce
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...

# System One servers (local GPU), from ~/projects/plafor/s1:
#   Laya:  pip install "laya[serve]"; LAYA_DEVICE=cuda LAYA_MODELS=multilingual laya-serve   (:8000)
#   Kev:   git clone https://github.com/jaredpalmer/kev && cd kev && uv sync --extra serve
#          KEV_DTYPE=bf16 uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8009
#   Jev:   export TYPESAFE_API_KEY=... (hosted; system `s1_jev`)

python src/run.py --seed-from-v1          # reuse the preliminary run-0 paths
python src/run.py --systems single_agent multi_agent single_agent_max5 multi_agent_guarded hybrid \
                  retrieval_k3 retrieval_k10 --models gpt-4o-mini --runs 3
python src/run.py --systems single_agent multi_agent --models gpt-5.4-mini:none gpt-5.4-mini:high --runs 3
python src/run.py --systems s1_laya s1_kev4b && python src/run.py --derive-s1-top5
python src/rejudge.py --judges gpt-4o claude-opus-5-5
python src/analyze.py && python src/analyze_v2.py && python src/make_schematics.py
```
