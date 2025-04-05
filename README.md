# nev
Natural language evaluation for proof problems.

## Usage

We need to create a config file for OpenAI API key before using nev. It is named `.apiconfig.json` at the root folder, which should include two elements as follows:

```json
{
    "OPENAI_API_KEY": "sk-xxx",
    "OPENAI_BASE_URL": "https://xxx.xx/v1"
}
```

You should replace the contents with your own API key and base url. After that, you should include your problems in a json file, as a list of strings. For example you can create a file named `problems.json` at the root of this project, with the following contents:

```json
[
    "Prove the fundamental theorem of algebra.",
    "Prove Fermat's last theorem.",
    "Prove Riemann's hypothesis."
]
```

And then you can run this script to run naive proving and criticizing process for these problems.

```bash
python nev.py -p problems.json --naive --save_path results.json --proof_model o3-mini --eval_model o3-mini
```
