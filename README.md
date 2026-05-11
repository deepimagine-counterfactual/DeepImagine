# DeepImagine

Code for *DeepImagine: Clinical Trial Outcome Prediction via Stepwise Local Counterfactual Imaginations*.

## Abstract

Predicting the outcomes of prospective clinical trials remains a major challenge. Clinical trial outcomes result from complex interactions among experimental factors such as drug interventions, participant demographics, and protocols. Here, we introduce DeepImagine, a framework that predicts target trial outcomes through stepwise counterfactual imagination anchored on historical trials with observed results. Starting from a relevant historical trial, DeepImagine sequentially modifies one differing experimental factor at a time. With each step a large language model (LLM) is posed a local counterfactual: how would the current imagined outcome change with this single perturbation? The updated result is carried forward as the input to the next step, until the historical configuration exactly matches the target, yielding the final prediction. Empirically, DeepImagine consistently outperforms direct one-step prediction across several off-the-shelf LLMs, with further gains when multiple imagination pathways, initiated from different historical anchors, are aggregated. We also construct natural counterfactuals augmented with synthetic reasoning traces and train a family of specialized language models, each dedicated to learning one factor's local counterfactual transition. Integrating these learned local operators into DeepImagine yields substantial improvements over general-purpose LLM baselines. Our findings position stepwise counterfactual imagination, distinct from both correlational prediction and explicit structural causal modeling, as a promising direction for clinical trial outcome prediction.

## Repository layout

```
deepimagine/
├── configs/          # YAML configs for training, inference, data
├── data/             # trial dumps + counterfactual pairs
│   └── benchmarks/   # CT Open benchmark files
├── src/deepimagine/
│   ├── data/         # ctgov loading, schema, natural/synthetic CF builders, retrieval
│   ├── models/       # local operators, LLM backbones, FFNN + HINT baselines
│   ├── inference/    # the DeepImagine pathway, aggregation, Direct + RAG
│   ├── training/     # SFT loops for the operators and for the A1/A2 ablations
│   └── evaluation/   # CT Open scoring
├── scripts/          # CLI entrypoints
└── tests/
```

## Setup

```bash
git clone https://github.com/deepimagine-counterfactual/DeepImagine.git
cd DeepImagine
pip install -e .
```

Python 3.10+. The training runs reported in the paper used a single node with 7×80GB H100s and FlashAttention-4. Inference with API backbones (GPT-5, GPT-5-mini, o3-mini) just needs an API key.

```bash
export OPENAI_API_KEY=...   # for the API backbones and synthetic data generation
export HF_TOKEN=...         # for pulling Gemma-3-12B-it
export NVE_MODEL_PATH=...   # NV-Embed-v2 checkpoint (or HF id)
```

## Pipeline

**1. Pull and clean trial data.** We start from a clinicaltrials.gov dump and keep RCTs with enrollment ≥ 50.

```bash
bash scripts/download_trials.sh
python -m deepimagine.data.ctgov_loader --min-enrollment 50 --out data/trials.parquet
```

**2. Build counterfactual training data.** Natural counterfactuals for A and O come straight from the controlled trials themselves (same trial, different arm, or same arm, different outcome). For N and E there is no natural source, so we synthesize them by retrieving similar trial pairs and asking GPT-5 to imagine the intermediate results along the `N -> A -> O -> E` path. The same pipeline also augments A and O.

```bash
python scripts/build_natural_counterfactuals.py --trials data/trials.parquet
python scripts/build_synthetic_counterfactuals.py --trials data/trials.parquet --per-factor 45000
```

**3. Fine-tune the four local operators.**

```bash
bash scripts/train_all_operators.sh
```

This is just four invocations of `train_operator.py` with `--factor N/A/O/E`. All four share the same hyperparameters: LoRA r=32, alpha=64, AdamW, peak LR 5e-5, cosine decay, one epoch, max seq 8192.

**4. Evaluate.** The default benchmark is CT Open Summer 2025 (cutoff 2025-08-31). Each question is a binary "does arm k1 beat arm k2 on outcome O?" — DeepImagine produces a result estimate for each arm via the stepwise chain and the same LLM compares them.

```bash
python scripts/run_deepimagine.py --backbone gemma --n-anchors 7
python scripts/run_deepimagine.py --backbone gpt-5 --n-anchors 7
```

Baselines (Direct, RAG, FFNN, HINT) and the A1/A2 ablations:

```bash
python scripts/run_baselines.py --strategy direct --backbone gpt-5
python scripts/run_baselines.py --strategy rag    --backbone gpt-5 --k 7
python -m deepimagine.training.train_ablation --variant A1 --reasoning
python -m deepimagine.training.train_ablation --variant A2 --reasoning
```

## Citation

```bibtex
@unpublished{deepimagine2026,
  title  = {DeepImagine: Clinical Trial Outcome Prediction via Stepwise Local Counterfactual Imaginations},
  author = {Anonymous},
  note   = {Under review},
  year   = {2026}
}
```

## License

Code is Apache 2.0. Trial data is from [clinicaltrials.gov](https://clinicaltrials.gov/) and subject to their terms. Gemma weights are under the Gemma Terms of Use.
