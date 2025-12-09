#!/usr/bin/env python3

"""
Minimal save probabilities example for SmolLM2-135M-Instruct.

Filepath: ./examples/save_prob_example.py
Project: CPEN455-Project-2025W1
Description: This script demonstrates how to save the predicted probabilities of
             a (possibly ensembled) model on a test dataset.

Usage:
    uv run -m examples.save_prob_example \
        --checkpoint_paths "checkpoints/model_1.pt,checkpoints/model_2.pt,checkpoints/model_3.pt"
"""

import os
import argparse
from dotenv import load_dotenv

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from einops import rearrange

from autograder.dataset import CPEN455_2025_W1_Dataset, ENRON_LABEL_INDEX_MAP
from model import LlamaModel
from model.config import Config
from model.tokenizer import Tokenizer
from utils.download import _resolve_snapshot_path
from utils.device import set_device
from utils.prompt_template import get_prompt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---------- Core helpers (copied/adapted from bayes_inverse) ----------

def get_seq_log_prob(prompts, tokenizer, model, device):
    encoded_batch = tokenizer.encode(
        prompts, return_tensors="pt", return_attention_mask=True
    )

    input_ids = encoded_batch["input_ids"].to(device)
    attention_mask = encoded_batch["attention_mask"].to(device)

    log_prob, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )

    shifted_log_prob = log_prob[:, :-1, :]
    shifted_input_ids = input_ids[:, 1:]
    shifted_attention_mask = attention_mask[:, 1:]

    gathered_log_prob = shifted_log_prob.gather(-1, shifted_input_ids.unsqueeze(-1)).squeeze(-1)
    gathered_log_prob = gathered_log_prob * shifted_attention_mask

    return gathered_log_prob.sum(dim=-1)


def bayes_inverse_llm_classifier_ensemble(args, models, batch, tokenizer, device):
    """
    Ensemble version of the Bayes-inverse classifier.

    models: list[LlamaModel]
    batch: (indices, subjects, messages, labels)
    Returns:
        is_correct (or None if labels are -1),
        (probs_ensemble, labels_pred)
        where probs_ensemble has shape (batch_size, 2) [ham, spam].
    """
    _, subjects, messages, labels = batch

    # Build ham and spam prompts for each email
    prompts_ham = [
        get_prompt(
            subject=subj,
            message=msg,
            label=ENRON_LABEL_INDEX_MAP.inv[0],  # ham label text
            max_seq_length=args.max_seq_len,
            user_prompt=args.user_prompt,
        )
        for subj, msg in zip(subjects, messages)
    ]

    prompts_spam = [
        get_prompt(
            subject=subj,
            message=msg,
            label=ENRON_LABEL_INDEX_MAP.inv[1],  # spam label text
            max_seq_length=args.max_seq_len,
            user_prompt=args.user_prompt,
        )
        for subj, msg in zip(subjects, messages)
    ]

    # Concatenate so first half are ham-prompts, second half spam-prompts
    prompts = prompts_ham + prompts_spam  # length = 2 * batch_size

    probs_list = []


    with torch.no_grad():
        for model in models:
            seq_log_prob = get_seq_log_prob(prompts, tokenizer, model, device)

            # Reshape (2 * B,) → (B, 2), where dim 1: 0=ham, 1=spam
            seq_log_prob = rearrange(seq_log_prob, "(c b) -> b c", c=2)

            # Softmax over ham/spam dimension
            probs = F.softmax(seq_log_prob, dim=-1)  # (B, 2)
            probs_list.append(probs)

    # Stack over models: (M, B, 2) → mean over M → (B, 2)
    probs_ensemble = torch.stack(probs_list, dim=0).mean(dim=0)

    # Predicted label index (0=ham, 1=spam)
    labels_pred = torch.argmax(probs_ensemble, dim=-1)  # (B,)

    if -1 in labels:
        is_correct = None
    else:
        is_correct = labels_pred.cpu() == labels

    return is_correct, (probs_ensemble.detach().cpu(), labels_pred.detach().cpu())


def save_probs_ensemble(args, models, tokenizer, dataloader, device, name="test"):
    """
    Save ham/spam probabilities for each email in `dataloader` using an ensemble
    of models. CSV columns: data_index,prob_ham,prob_spam
    """
    os.makedirs(args.prob_output_folder, exist_ok=True)
    save_path = os.path.join(
       PROJECT_ROOT, f"{args.prob_output_folder}/{name}_dataset_probs.csv"
    )

    # Overwrite if exists
    if os.path.exists(save_path):
        os.remove(save_path)

    with torch.no_grad():
        from tqdm import tqdm

        for batch in tqdm(dataloader, desc=f"saving probabilities (ensemble: {len(models)} models)"):
            _, (probs, _) = bayes_inverse_llm_classifier_ensemble(
                args, models, batch, tokenizer, device=device
            )

            data_index, _, _, _ = batch
            indices = torch.as_tensor(data_index).view(-1).tolist()

            # probs[:, 0] = ham, probs[:, 1] = spam
            rows = zip(indices, probs[:, 0].tolist(), probs[:, 1].tolist())

            file_exists = os.path.exists(save_path)
            with open(save_path, "a", newline="") as handle:
                if not file_exists:
                    handle.write("data_index,prob_ham,prob_spam\n")
                handle.writelines(f"{idx},{ham},{spam}\n" for idx, ham, spam in rows)


# ---------- Main script ----------

if __name__ == "__main__":
    # random seed for reproducibility
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument(
        "--test_dataset_path",
        type=str,
        default="autograder/cpen455_released_datasets/test_subset.csv",
    )
    parser.add_argument(
        "--prob_output_folder",
        type=str,
        default="bayes_inverse_probs",
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default=""
    )
    parser.add_argument(
        "--checkpoint_paths",
        type=str,
        default = os.path.join("examples", "ckpt", "best1.pt") + ", " + os.path.join("examples", "ckpt", "best2.pt") + ", " +  os.path.join("examples", "ckpt", "best3.pt"),
        help=(
            "Comma-separated list of fine-tuned checkpoint .pt files. "
            "If multiple are provided, they are ensembled by averaging probabilities."
        ),
    )

    # These are not used here but kept for compatibility with other scripts
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    args = parser.parse_args()

    load_dotenv()

    # Base HF / snapshot checkpoint (for config & tokenizer)
    base_checkpoint = os.getenv("MODEL_CHECKPOINT")
    model_cache_dir = os.getenv("MODEL_CACHE_DIR")

    # Set device (GPU / MPS / CPU)
    device = set_device()

    # Load tokenizer and config from base checkpoint
    tokenizer = Tokenizer.from_pretrained(base_checkpoint, cache_dir=model_cache_dir)
    base_path = _resolve_snapshot_path(base_checkpoint, cache_dir=model_cache_dir)
    config = Config._find_config_files(base_path)

    # Build ensemble of models
    ckpt_paths = [
        p.strip()
        for p in args.checkpoint_paths.split(",")
        if p.strip()
    ]

    if len(ckpt_paths) == 0:
        raise ValueError(
            "No checkpoint paths provided. Use --checkpoint_paths to specify at least one .pt file."
        )


    models = []
    for path in ckpt_paths:

        path = os.path.join(PROJECT_ROOT, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        m = LlamaModel(config).to(device)
        ckpt = torch.load(path, map_location=device)

        # Support both styles: whole dict or plain state_dict
        state_dict = ckpt.get("model_state_dict", ckpt)
        m.load_state_dict(state_dict)
        m.eval()
        models.append(m)

        print(f"Loaded checkpoint: {path}")

    print(f"Using {len(models)} model(s) for ensemble inference.")

    test_dataset_dir = os.path.join(PROJECT_ROOT, args.test_dataset_path)

    # Load test dataset & dataloader
    test_dataset = CPEN455_2025_W1_Dataset(csv_path=test_dataset_dir)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # Save probabilities on test set
    save_probs_ensemble(
        args,
        models,
        tokenizer,
        test_dataloader,
        device=device,
        name="test",
    )
    print("Done. Saved probabilities for test dataset.")
