# nev

Natural Language Evaluation for Proof Problems

`nev` is a lightweight framework for evaluating mathematical proof problems using large language models (LLMs). It automates proof generation, criticism, and refinement workflows, and supports multiple evaluation pipelines.

## Features

- Naive proof solving and evaluation pipeline
- Pessimistic voting and refining workflows
- Batch processing with multithreading support
- Automatic conversion of JSON logs to human-readable Markdown

## Installation

1. Clone the repository and navigate into the project folder:
   ```bash
   git clone https://github.com/Carlos-Mero/nev.git
   cd nev
   ```
2. (Optional) Create and activate a Python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Before running `nev`, create a configuration file for your OpenAI (or compatible) API credentials:

```json
{
  "OPENAI_API_KEY": "sk-...",
  "OPENAI_BASE_URL": "https://api.openai.com/v1"
}
```

Save this file as `.apiconfig.json` in the project root.

## Usage

All commands are invoked via `nev.py`. Below are common workflows:

### 1. Solve and Evaluate Problems

Prepare a JSON file (`problems.json`) containing a list of problem statements:
```json
[
  "Prove the fundamental theorem of algebra.",
  "Prove Fermat's Last Theorem.",
  "Prove the Riemann Hypothesis."
]
```
Run the default (naive) pipeline:
```bash
python nev.py \
  --proof_model deepseek-r1 \
  --eval_model deepseek-r1 \
  -p problems.json \
  --save_path results.json \
  --workers 4
```

### 2. Pessimistic Voting or Refinement

We can separately enable pessimistic verification and refinement process separately by passing the commandline argument `--reviews` and `--iterations`. You can also pass `-w` or `--workers` argument to enable multithreading sampling in API calls.

```bash
# Pessimistic voting (collect N reviews before judgment)
python nev.py --reviews 3 -p problems.json --save_path vote_results.json

# Pessimistic refine (iterative proof improvement)
python nev.py --reviews 3 --iterations 2 -p problems.json --save_path refine_results.json
```

The outputs contain:

- **JSON log**: Contains each problem, generated proof, evaluation text, auto judgement, and optional manual annotations.
- **Markdown report**: A human-readable summary (`.md` file) generated alongside the JSON log.


### 3. MathAgent Sampling Loop

The MathAgent simplifier pipeline (`mas` method) explores conjectures iteratively using planning, solving, reviewing, and refining steps. It maintains a memory of explored goals and logs detailed sampling steps.

```bash
# Run MathAgent sampling with default settings
python nev.py \
  --method mas \
  --proof_model deepseek-r1 \
  --eval_model deepseek-r1 \
  --reform_model deepseek-v3 \
  --steps 6 \   # max exploration iterations
  --reviews 3 \ # reviewers in pessimistic verification
  --iterations 2 \ # maximum refine iterations for each proof
  --log_dir samples \   # output directory for detailed logs
  --log_per_steps 10 \   # flush logs every N steps
  -p problems.json
```

Logs will be written under the `samples/` directory (or your chosen `--log_dir`), containing JSON files for each step. After sampling completes, the final proofs and judgements are saved to `ma_results.json`.

### 4. Re-evaluate Annotated Dataset

If you have a dataset with manual judgements, re-run evaluation to measure model accuracy:
```bash
python nev.py --reevaluate annotated.json --eval_model deepseek-r1 --save_path reeval.json
```

### 5. View Logs in Terminal

```bash
python nev.py --view results.json --start 0 --n_samples 3 [--false_only]
```

## Contributing

Contributions are welcome! Please submit issues or pull requests to discuss new features or fixes.
