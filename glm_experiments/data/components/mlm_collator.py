"""Data collator for masked language modeling."""

from typing import Any, Dict, List, Union

import torch
from transformers import DataCollatorForLanguageModeling


class DataCollatorForLanguageModelingSimplified(DataCollatorForLanguageModeling):
    """Simplified data collator for MLM.

    Assumes all sequences in a batch have the same length (no padding needed).
    Based on GPN's implementation in gpn/ss/run_mlm.py.

    Args:
        tokenizer: HuggingFace tokenizer
        mlm: Whether to use masked language modeling
        mlm_probability: Probability of masking tokens
    """

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        """Collate examples into a batch.

        Args:
            examples: List of examples from dataset

        Returns:
            Batch dictionary with input_ids and labels
        """
        # Stack all tensor fields
        batch = {
            key: torch.stack([torch.tensor(example[key]) for example in examples], dim=0)
            for key in examples[0].keys()
        }

        # Extract special tokens mask if present
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        # Convert input_ids to int64 for masking (parent class expects int64)
        if batch["input_ids"].dtype == torch.uint8:
            batch["input_ids"] = batch["input_ids"].to(torch.int64)

        # Apply MLM masking
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )

        return batch
