"""lm-eval-harness regression config — P1.

Builds the config shape lm-eval-harness expects; does not invoke the harness itself (that needs a
served/local model or an API key lm-eval-harness supports directly, out of scope without GPU/serving
infra). render_lm_eval_cli_command() turns it into a copy-pasteable command for whenever real compute
is available.
"""
from __future__ import annotations


def build_lm_eval_config(
    model_id: str,
    tasks: list[str],
    num_fewshot: int = 0,
    model_args: dict | None = None,
    batch_size: int = 8,
) -> dict:
    if not tasks:
        raise ValueError("build_lm_eval_config requires at least one task")
    return {
        "model": "hf" if not model_id.startswith(("openai", "anthropic", "vllm")) else model_id.split("/")[0],
        "model_args": model_args or {"pretrained": model_id},
        "tasks": list(tasks),
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
    }


def render_lm_eval_cli_command(config: dict) -> str:
    model_args_str = ",".join(f"{k}={v}" for k, v in config["model_args"].items())
    tasks_str = ",".join(config["tasks"])
    return (
        f"lm_eval --model {config['model']} --model_args {model_args_str} "
        f"--tasks {tasks_str} --num_fewshot {config['num_fewshot']} --batch_size {config['batch_size']}"
    )
