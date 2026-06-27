# SimpleMind

`SimpleMind` is my end-to-end LLM agent training and inference project built on top of [MiniMind](https://github.com/jingyaogong/minimind).

The repo is intentionally broad: it starts from a `64M` prototype model for fast RL iteration, then scales key ideas to `Qwen2.5-1.5B` for multi-step agent RL and to `Qwen2.5-7B` for inference-system experiments.

The main reason this project exists is not to show a single benchmark score, but to demonstrate a full chain:

`pretrain -> SFT -> reward design -> agent RL -> evaluation -> inference optimization`

## Core project claims

### 1. Plan-then-Execute RL works on 1.5B models

I built a Router / Expert multi-agent setup where:

- Router outputs an `execute_plan`
- Experts execute sub-tasks in dependency order
- Synthesize stage aggregates final answers

On the held-out evaluation split:

- multi-step planning accuracy: `98%` (`49/50`)
- end-to-end success rate: `58%` (`29/50`)

The main residual failure mode is not planning itself, but `tool_call` format adherence in downstream execution.

### 2. SFT data distribution directly affects RL policy initialization

In the handoff routing setup, I observed that SFT warmup does not only teach format, it also injects policy bias.

- With `78%` positive tool-call samples in SFT warmup, RL inherits a strong false-trigger prior.
- The RL stage alone cannot easily reverse that prior with weak negative reward.

This led to a three-part fix:

- align SFT class ratio toward the target distribution
- increase `none` examples during RL
- use asymmetric penalty design for false-trigger behavior

### 3. PPL-driven data filtering is useful even for small-model training

I implemented a data-quality pipeline:

- batch PPL scoring
- KDE valley detection for threshold selection
- length normalization to reduce systematic bias

The bucketed experiments show a clear inverted-U contribution curve, and trimming `35%` of low-value data does not hurt validation loss.

### 4. Continuous batching bottleneck is often Python-side, not just model-side

I implemented a teaching-oriented continuous batching engine for `Qwen2.5-7B` and compared:

- serial baseline
- sequential decode scheduling
- batched decode

At `32` concurrent requests, `P95` latency drops from `36s` to `2.4s` (`15x`). The scaling experiments show the next bottleneck is Python-side KV-cache pad/cat/slice overhead, which is exactly why production-grade paged attention needs CUDA-level implementation.

### 5. 64M prototypes are useful for RL method iteration, even when they hit a capability ceiling

The `64M` branch of this repo is where I iterated quickly on:

- reward mismatch diagnosis
- reward hacking detection
- plan-execute training flow
- RL stability debugging

The final conclusion is not that `64M` is enough, but that it is a useful experimental vehicle for finding broken reward design and training logic before moving to larger models.

## Fast navigation

If you are reading this repo from a resume or interview context, start here:

- `README.md`
  - project overview and main results
- `agent_plan_qwen.py`
  - 1.5B plan-then-execute RL main logic
- `agent_handoff_qwen.py`
  - 1.5B handoff RL pipeline
- `scripts/continuous_batching_engine.py`
  - continuous batching engine implementation
- `scripts/data_quality_scorer.py`
  - PPL-based data filtering
- `docs/实验记录/`
  - experiment reports and analysis
- `trainer/logs/selected/`
  - representative raw logs
- `benchmark_logs/selected/`
  - representative batching benchmark logs

## Project structure

```text
SimpleMind/
├── model/
├── trainer/
│   ├── train_agent.py
│   ├── train_plan.py
│   ├── train_pretrain.py
│   ├── train_full_sft.py
│   ├── train_grpo.py
│   ├── train_dpo.py
│   ├── train_ppo.py
│   └── logs/selected/
├── scripts/
│   ├── continuous_batching_engine.py
│   ├── benchmark_continuous_batching.py
│   ├── data_quality_scorer.py
│   ├── run_handoff_qwen_single.sh
│   ├── run_plan_qwen.sh
│   └── run_benchmark.sh
├── docs/
│   └── 实验记录/
├── benchmark_logs/
│   └── selected/
├── agent_handoff_qwen.py
├── agent_plan_qwen.py
├── eval_handoff.py
└── eval_plan.py
```

## Where to look for each contribution

### Plan-then-Execute RL

- entry: `agent_plan_qwen.py`
- launcher: `scripts/run_plan_qwen.sh`
- evaluation: `eval_plan.py`
- report: `docs/实验记录/plan_execute_experiment_report.md`

### Handoff RL and SFT->RL distribution transfer

- entry: `agent_handoff_qwen.py`
- launcher: `scripts/run_handoff_qwen_single.sh`
- evaluation: `eval_handoff.py`
- report: `docs/实验记录/handoff_qwen_experiment_report.md`

### Data quality filtering

- implementation: `scripts/data_quality_scorer.py`
- report: `docs/实验记录/agent_rl_experiment_report.md`

### Continuous batching

- implementation: `scripts/continuous_batching_engine.py`
- benchmark runner: `scripts/run_benchmark.sh`
- report: `docs/实验记录/continuous_batching_benchmark_report.md`

## Representative artifacts kept in the public repo

To keep the repo readable, only representative artifacts are retained:

- selected experiment reports in `docs/实验记录/`
- selected training logs in `trainer/logs/selected/`
- selected benchmark logs in `benchmark_logs/selected/`

Large volumes of repetitive raw logs and machine-specific debugging traces are intentionally removed from the public version.

## Minimal usage

```bash
pip install -r requirements.txt

# Handoff RL
bash scripts/run_handoff_qwen_single.sh

# Plan-then-Execute RL
bash scripts/run_plan_qwen.sh

# Continuous batching benchmark
bash scripts/run_benchmark.sh

# Evaluation
python eval_handoff.py
python eval_plan.py
```

## Upstream acknowledgement

This project is built on top of [jingyaogong/minimind](https://github.com/jingyaogong/minimind). Many experiments here focus on extending that base into agent RL, data-quality engineering, and inference optimization.

## License

Apache 2.0
