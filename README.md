# PlaforEdu Learning-Path Curation

Code and data for the paper *Fast, Slow, or Agentic? System One Decision Models,
Reasoning LLMs, and Self-Reflective Multi-Agent Curation of Learning Paths for
Public-Sector Training* (`papers/ieee-access/access.tex`).

It compares LLM agents (single agent, multi-agent critic loop, hybrid), embedding
retrieval and System One decision models (Kev, Laya, Jev) for curating ordered
learning paths (*trilhas formativas*) on the PlaforEdu platform.

## Layout
- `src/db.py`: SQLite access to the course and competency catalog.
- `src/systems.py`: all curation systems.
- `src/run.py`: path generation, written to `results/gen_v2.jsonl`.
- `src/rejudge.py`: LLM judges, written to `results/judgments_v2.jsonl`.
- `src/analyze_v2.py`: tables, figures and LaTeX macros for the paper.
- `src/make_human_eval.py`: blind workbook for human curators.
- `src/judge.py`, `src/analyze.py`: preliminary (v1) study.
- `papers/ieee-access/`: manuscript.

## Reproduce
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...

# System One servers on a local GPU
#   Laya: pip install "laya[serve]"; LAYA_MODELS=multilingual laya-serve           (port 8000)
#   Kev:  git clone https://github.com/jaredpalmer/kev; uv sync --extra serve
#         uv run python -m kev.serve --run jaredpalmer/kev-4b@v1.0 --port 8009
#   Jev:  export TYPESAFE_API_KEY=...

python src/run.py --seed-from-v1
python src/run.py --systems single_agent multi_agent single_agent_max5 multi_agent_guarded hybrid \
                  retrieval_k3 retrieval_k10 --models gpt-4o-mini --runs 3
python src/run.py --systems single_agent multi_agent --models gpt-5.4-mini:none gpt-5.4-mini:high --runs 3
python src/run.py --systems s1_laya s1_kev4b && python src/run.py --derive-s1-top5
python src/rejudge.py --judges gpt-4o claude-opus-5-5
python src/analyze.py && python src/analyze_v2.py
```
