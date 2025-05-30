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

We also supports resuming exploration from existing logs and manually provide some context or hint for the agent. You can additionally pass this argument `--resume <path_to_logdir>` to MathAgent pipeline to enable resuming. The manually made context should be in the same format as `memory.json`, here is an example when working with homogenization problems.

```json
[
  {
    "type": "assumption",
    "content": "1. \\(\\Omega\\) is a bounded open domain with a connected Lipschitz boundary \\(\\partial\\Omega\\) 2. \\(D\\) is an open domain with a finite number of connected components, each having a Lipschitz boundary \\(\\partial D\\).  3. \\(\\Omega\\setminus D\\) is connected and has a Lipschitz boundary \\(\\partial\\Omega\\cup\\partial D\\).  The connected components of \\(D\\) are denoted as \\(D_{i}\\), \\(i=1,\\ldots,N\\), where \\(N\\) is finite."
  },
  {
    "type": "hint",
    "content": "You can use the two-scale expansion method to get the cell problem and homogenized equation to solve this problem."
  },
  {
    "type": "hint",
    "content": "You should show the detail process in each steps of this problem proving"
  }
]
```

The context should be a list of dictionaries in json, where two elements are strictly required, `type` and `content`. You can add more elements in each context if you wish, they will appear in the final log but will not acturally affect the behaviour of the agent pipeline. You can directly pass the argument `--context <path_to_context_file>` to add manual context for the agent.

### 4. Re-evaluate Annotated Dataset

If you have a dataset with manual judgements, re-run evaluation to measure model accuracy:
```bash
python nev.py --reevaluate annotated.json --eval_model deepseek-r1 --save_path reeval.json
```

### 5. Other utilities

You can view some samples in the terminal with the following script.

```bash
python nev.py --view results.json --start 0 --n_samples 3 [--false_only]
```

If you are using o4-mini or o3 with MathAgent pipeline, the agent might generate a lot of unicode symbols in its response. You can use the following script to translate them into standard LaTeX syntax if needed. This will create three new files of memories with a prefix `f-` without modifying the original ones to avoid bad behaviour.

```bash
python nev.py -ft <path_to_logdir>
```

## Contributing

Contributions are welcome! Please submit issues or pull requests to discuss new features or fixes.
