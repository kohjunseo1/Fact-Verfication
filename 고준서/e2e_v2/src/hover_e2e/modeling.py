from __future__ import annotations

from contextlib import nullcontext
import inspect
import json
from pathlib import Path
import time
from typing import Any


def require_gpu_stack():
    try:
        import numpy as np
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification, AutoTokenizer,
            DataCollatorWithPadding, Trainer, TrainingArguments, set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "GPU dependencies are missing. In Colab run: pip install -e '.[gpu,test]'"
        ) from exc
    return np, torch, Dataset, AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer, TrainingArguments, set_seed


def require_cuda_runtime() -> str:
    """Fail before expensive pair/cache construction when Colab GPU is not active."""
    _, torch, *_ = require_gpu_stack()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for this stage. In Colab choose Runtime > Change runtime type > GPU."
        )
    return str(torch.cuda.get_device_name(0))


class HFPairScorer:
    def __init__(self, model_path: str | Path, *, max_length: int, device: str | None = None):
        started = time.perf_counter()
        print(f"[model] loading scorer: {model_path}", flush=True)
        np, torch, _, AutoModel, AutoTokenizer, _, _, _, _ = require_gpu_stack()
        self.np, self.torch = np, torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModel.from_pretrained(str(model_path))
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.max_length = int(max_length)
        print(
            f"[model] scorer ready device={self.device} "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    def score(self, left: list[str], right: list[str], *, batch_size: int = 64) -> list[float]:
        if len(left) != len(right):
            raise ValueError("left/right length mismatch")
        output = []
        for start in range(0, len(left), batch_size):
            encoded = self.tokenizer(
                left[start:start + batch_size], right[start:start + batch_size],
                padding=True, truncation=True, max_length=self.max_length, return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            autocast = (
                self.torch.autocast(device_type="cuda", dtype=self.torch.float16)
                if self.device.startswith("cuda") else nullcontext()
            )
            with self.torch.inference_mode(), autocast:
                logits = self.model(**encoded).logits
                output.extend(self.torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist())
        return [float(value) for value in output]


def _macro_f1(gold, predicted) -> float:
    scores = []
    for label in (0, 1):
        tp = int(((gold == label) & (predicted == label)).sum())
        fp = int(((gold != label) & (predicted == label)).sum())
        fn = int(((gold == label) & (predicted != label)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / 2


def train_pair_classifier(
    train_rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]], *,
    output_dir: str | Path, model_name: str = "roberta-base", max_length: int = 384,
    epochs: float = 2, learning_rate: float = 2e-5, train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1, eval_batch_size: int = 64,
    seed: int = 42, class_weights: list[float] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    np, torch, Dataset, AutoModel, AutoTokenizer, Collator, Trainer, TrainingArguments, set_seed = require_gpu_stack()
    if not train_rows or not validation_rows:
        raise ValueError("Non-empty Train and internal Validation rows are required")
    train_uids = {str(row["uid"]) for row in train_rows}
    validation_uids = {str(row["uid"]) for row in validation_rows}
    if train_uids & validation_uids:
        raise ValueError("Train/Validation UID leakage")
    set_seed(seed)
    print(f"[train] loading tokenizer: {model_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def encode(batch):
        encoded = tokenizer(
            batch["text_a"], batch["text_b"], truncation=True, max_length=max_length
        )
        encoded["labels"] = batch["label"]
        return encoded

    columns = ["text_a", "text_b", "label"]
    print(
        f"[train] tokenizing train={len(train_rows):,} validation={len(validation_rows):,}",
        flush=True,
    )
    train_ds = Dataset.from_list([{key: row[key] for key in columns} for row in train_rows]).map(
        encode, batched=True, remove_columns=columns
    )
    validation_ds = Dataset.from_list([{key: row[key] for key in columns} for row in validation_rows]).map(
        encode, batched=True, remove_columns=columns
    )
    print(f"[train] loading base model: {model_name}", flush=True)
    model = AutoModel.from_pretrained(model_name, num_labels=2)

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            weights = None if class_weights is None else torch.tensor(class_weights, device=outputs.logits.device)
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)
            return (loss, outputs) if return_outputs else loss

    arguments = {
        "output_dir": str(Path(output_dir).resolve()),
        "num_train_epochs": epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "per_device_eval_batch_size": eval_batch_size,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "logging_steps": 50,
        "report_to": "none",
        "seed": seed,
        "fp16": bool(torch.cuda.is_available()),
    }
    parameter_names = inspect.signature(TrainingArguments.__init__).parameters
    arguments["eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"] = "epoch"
    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(**arguments),
        train_dataset=train_ds,
        eval_dataset=validation_ds,
        data_collator=Collator(tokenizer),
        compute_metrics=lambda prediction: {
            "accuracy": float((prediction.predictions.argmax(-1) == prediction.label_ids).mean()),
            "macro_f1": _macro_f1(prediction.label_ids, prediction.predictions.argmax(-1)),
        },
    )
    checkpoints = sorted(
        Path(output_dir).glob("checkpoint-*"),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    resume = str(checkpoints[-1]) if checkpoints else None
    print(
        f"[train] trainer start resume={resume or 'none'} "
        f"steps_log_interval={arguments['logging_steps']}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=resume)
    print("[train] optimization complete; saving best model", flush=True)
    trainer.save_model(str(Path(output_dir) / "best_model"))
    tokenizer.save_pretrained(str(Path(output_dir) / "best_model"))
    metrics = trainer.evaluate()
    result_path = Path(output_dir) / "training_result.json"
    temporary = result_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(result_path)
    print(
        f"[train] evaluation complete elapsed={(time.perf_counter() - started) / 60:.1f}m "
        f"metrics={metrics}",
        flush=True,
    )
    return metrics
