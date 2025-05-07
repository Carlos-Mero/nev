import re
import json
import logging

def find_box(pred_str: str):
    ans = pred_str.split("boxed")[-1]
    if not ans:
        return None
    if ans[0] == "{":
        stack = 1
        a = ""
        for c in ans[1:]:
            if c == "{":
                stack += 1
                a += c
            elif c == "}":
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split("$")[0].strip()
    return a if a else None

def extract_boxed(text):
    matches = re.findall(r'(true|false)', find_box(text), flags=re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    else:
        return "false"

def extract_judgement(text):
    matches = re.findall(r'\*\*(true|false)\*\*', text, re.IGNORECASE)
    if matches:
        return matches[-1].lower()
    return "false"

def remove_think_tags(text):
    cleaned_text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return cleaned_text

def extract_tag_content(text, tag):
    pattern = fr"<{tag}>.*?</{tag}>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches[-1]
    else:
        logging.warning(f"No content extracted in given tag <{tag}>.")
        return None

def extract_all_tag_content(text, tag):
    pattern = fr"<{tag}>.*?</{tag}>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if matches:
        return matches
    else:
        logging.warning(f"No content extracted in given tag <{tag}>.")
        return None

def remove_tag_content(text, tag):
    cleaned_text = re.sub(fr'<{tag}>.*?</{tag}>', '', text, flags=re.DOTALL)
    return cleaned_text

def remove_all_tag_content(text: str) -> str:
    if isinstance(text, str):
        return re.sub(r'<.*?>', '', text) # Delete all <> tags
    return text

def convert_json_to_md(json_path, md_path) -> None:
    """
    This function converts a JSON file to a markdown file.
    Arguments: json_path: The path to the JSON file.
               md_path: The path to the markdown file.
    The output markdown file contains the problems, proofs, evaluations, judgements(from both the judger and human annotators), and comments.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)

    with open(md_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(samples, start=1):
            f.write(f"## Problem {idx}\n\n")

            f.write("### Problem:\n\n")
            f.write(f"{item['problem'].strip().replace('##', '####')}\n\n")

            f.write("### Proof:\n\n")
            f.write(f"{item['proof'].strip().replace('##', '####')}\n\n")

            f.write("### Evaluation by Judger:\n\n")
            f.write(f"{item['evaluation'].strip().replace('##', '####')}\n\n")

            f.write("### Judger judgement:\n\n")
            f.write("Correct\n\n" if item['judgement'] else "Incorrect\n\n")

            if 'manual_judgement' in item:
                f.write("### Manual judgement:\n\n")
                if item['manual_judgement'] is True:
                    f.write("Correct\n\n")
                elif item['manual_judgement'] is False:
                    f.write("Incorrect\n\n")
                else:  # Null or other values
                    f.write("Not provided\n\n")
            
            if 'comment' in item:
                f.write("### Comment:\n\n")
                f.write(f"{item['comment'].strip().replace('##', '####')}\n\n")

            f.write("---\n\n")

    print(f"Converted {len(samples)} problems to markdown and saved to {md_path}")

def convert_memory_json_to_md(json_path, md_path) -> None:
    """
    This function converts a memory JSON file to a markdown file.
    Arguments: json_path: The path to the JSON file.
            md_path: The path to the markdown file.
    The output markdown file contains the types, contents, correctnesses, proofs, and comments.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    with open(md_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(samples, start=1):
            f.write(f"## Memory {idx}\n\n")

            f.write("### Type:\n\n")
            f.write(f"{item['type']}\n\n")

            f.write("### Content:\n\n")
            f.write(f"{remove_all_tag_content(item['content'].strip().replace('##', '####'))}\n\n")

            if item['correctness'] is not None:
                f.write("### Correctness:\n\n")
                f.write(f"{item['correctness']}\n\n")

            if item['proof'] is not None:
                f.write("### Proof:\n\n")
                f.write(f"{remove_all_tag_content(item['proof'].strip().replace('##', '####'))}\n\n")

            if item['comment'] is not None:
                f.write("### Comment:\n\n")
                f.write(f"{remove_all_tag_content(item['comment'].strip().replace('##', '####'))}\n\n")

def convert_memory_json_to_latex(json_path, latex_path) -> None:
    """
    This function converts a memory json samples into compilable latex file for preview.
    Arguments: json_path: The path to the JSON file.
            latex_path: The path to the latex file.
    """
    latex_header = r"""
\documentclass[12pt]{article}

% --- Packages ---
\usepackage[utf8]{inputenc}    % For UTF-8 encoding
\usepackage{geometry}          % To adjust margins
\usepackage{amsmath}           % For math environments
\usepackage{amsthm}            % For lemma and proof environments
\usepackage{amssymb}           % For math symbols
\usepackage{graphicx}          % To include images
\usepackage{hyperref}          % For hyperlinks

% --- Page Settings ---
\geometry{a4paper, margin=1in}
\newtheorem{lemma}{Lemma}

% --- Document Starts ---
\begin{document}

\title{Explore Trajectory of MathAgent}
\author{MathAgent}
\date{\today}

\maketitle

"""
    with open(json_path, "r", encoding="utf-8") as f:
        samples = json.load(f)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_header)
        for idx, item in enumerate(samples, start=1):
            for key, value in item.items():
                value = remove_all_tag_content(value)
                if key == "content":
                    f.write(f'\\begin{{lemma}}\n{value}\\end{{lemma}}\n\n')
                elif key =="proof":
                    f.write(f'\\begin{{proof}}\n{value}\\end{{proof}}\n\n')
                # TODO! we need more formatting for context elements.
                else:
                    f.write(f'\\textbf{{{key}}}: {value}\n')
        f.write(r'\end{document}')

def view_samples(args):
    with open(args.view, "r", encoding="utf-8") as file:
        samples = json.load(file)

    if args.false_only:
        samples = [s for s in samples if not s['judgement'] or ('manual_judgement' in s.keys() and not s['manual_judgement'])]
    logging.info(f"Total samples count: {len(samples)}")

    if isinstance(samples, list):
        samples = samples[args.start : args.start + args.n_samples]

        for i, s in enumerate(samples):
            print("#" * 50 + "\n")
            print(f"viewing sample {args.start + i}\n")
            for key, value in s.items():
                print(f"\033[92m{key.upper()}\033[0m: {value}\n")
    else:
        print("#" * 50 + "\n")
        print("viewing logs\n")
        for key, value in samples.items():
            print(f"\033[92m{key.upper()}\033[0m: {value}\n")
