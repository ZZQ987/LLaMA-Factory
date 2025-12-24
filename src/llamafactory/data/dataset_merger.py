# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import TYPE_CHECKING, Optional

from datasets import DatasetDict, load_from_disk

from ..extras import logging
from ..extras.misc import has_tokenized_data
from .data_utils import get_dataset_module, merge_dataset

if TYPE_CHECKING:
    from datasets import Dataset, DatasetDict

    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


def load_tokenized_dataset(tokenized_path: str) -> "DatasetDict":
    r"""Load a tokenized dataset from disk, similar to loader.py:288-297."""
    if not has_tokenized_data(tokenized_path):
        raise ValueError(f"Tokenized dataset not found at {tokenized_path}.")

    logger.info_rank0(f"Loading tokenized dataset from {tokenized_path}...")
    tokenized_data = load_from_disk(tokenized_path)
    
    # Ensure we have a DatasetDict
    if not isinstance(tokenized_data, DatasetDict):
        tokenized_data = DatasetDict({"train": tokenized_data})
    
    return tokenized_data


def merge_tokenized_datasets(
    dataset_paths: list[str],
    output_path: str,
    data_args: Optional["DataArguments"] = None,
    mix_strategy: str = "concat",
    interleave_probs: Optional[list[float]] = None,
    seed: int = 42,
    streaming: bool = False,
) -> "DatasetDict":
    r"""
    Merge multiple tokenized datasets into one.
    
    This function mimics the loading logic in loader.py:288-300 and merges multiple
    preprocessed tokenized datasets together.
    
    Args:
        dataset_paths: List of paths to tokenized datasets to merge
        output_path: Path to save the merged dataset
        data_args: Optional DataArguments for advanced merging options
        mix_strategy: Strategy to merge datasets ("concat" or "interleave")
        interleave_probs: Probabilities for interleave strategy (optional)
        seed: Random seed for interleave strategy
        streaming: Whether to use streaming mode (not recommended for merging)
    
    Returns:
        The merged DatasetDict
    """
    if not dataset_paths:
        raise ValueError("At least one dataset path must be provided.")
    
    if len(dataset_paths) == 1:
        logger.warning_rank0("Only one dataset provided, copying instead of merging.")
        dataset = load_tokenized_dataset(dataset_paths[0])
        if output_path:
            dataset.save_to_disk(output_path)
            logger.info_rank0(f"Dataset copied to {output_path}.")
        return dataset
    
    logger.info_rank0(f"Merging {len(dataset_paths)} tokenized datasets...")
    
    # Load all datasets
    datasets_dict = {}
    for i, path in enumerate(dataset_paths):
        logger.info_rank0(f"Loading dataset {i+1}/{len(dataset_paths)}: {path}")
        dataset_dict = load_tokenized_dataset(path)
        
        # Extract train dataset from each DatasetDict
        if "train" in dataset_dict:
            datasets_dict[f"dataset_{i}"] = dataset_dict["train"]
        else:
            # If no train split, take the first available split
            first_split = list(dataset_dict.keys())[0]
            datasets_dict[f"dataset_{i}"] = dataset_dict[first_split]
            logger.warning_rank0(
                f"No 'train' split found in {path}, using '{first_split}' instead."
            )
    
    # Merge datasets using the same logic as in loader.py
    datasets_list = list(datasets_dict.values())
    
    # Use data_args if provided, otherwise create minimal config
    if data_args is not None:
        merged_dataset = merge_dataset(datasets_list, data_args, seed=seed)
    else:
        # Simple merge without data_args
        from datasets import concatenate_datasets, interleave_datasets
        
        if mix_strategy == "concat":
            merged_dataset = concatenate_datasets(datasets_list)
        elif mix_strategy.startswith("interleave"):
            if interleave_probs is None:
                interleave_probs = [1.0 / len(datasets_list)] * len(datasets_list)
            elif len(interleave_probs) != len(datasets_list):
                raise ValueError(
                    f"interleave_probs length ({len(interleave_probs)}) "
                    f"must match number of datasets ({len(datasets_list)})"
                )
            
            # Normalize probabilities
            total_prob = sum(interleave_probs)
            interleave_probs = [p / total_prob for p in interleave_probs]
            
            stopping_strategy = (
                "first_exhausted" if mix_strategy.endswith("under") else "all_exhausted"
            )
            merged_dataset = interleave_datasets(
                datasets=datasets_list,
                probabilities=interleave_probs,
                seed=seed,
                stopping_strategy=stopping_strategy,
            )
        else:
            raise ValueError(f"Unknown mix_strategy: {mix_strategy}")
    
    # Wrap in DatasetDict
    if isinstance(merged_dataset, DatasetDict):
        merged_dict = merged_dataset
    else:
        merged_dict = DatasetDict({"train": merged_dataset})
    
    # Save to disk if output_path is provided
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        merged_dict.save_to_disk(output_path)
    
    return merged_dict


def merge_tokenized_datasets_cli():
    r"""Command-line interface for merging tokenized datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Merge multiple preprocessed tokenized datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "dataset_paths",
        type=str,
        nargs="+",
        help="Paths to tokenized datasets to merge",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the merged dataset",
    )
    parser.add_argument(
        "--mix_strategy",
        type=str,
        default="concat",
        choices=["concat", "interleave", "interleave_under"],
        help="Strategy to merge datasets: concat (concatenate) or interleave",
    )
    parser.add_argument(
        "--interleave_probs",
        type=float,
        nargs="+",
        default=None,
        help="Probabilities for interleave strategy (optional, default: uniform)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for interleave strategy",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output path if it already exists",
    )
    
    args = parser.parse_args()
    
    # Check output path
    if os.path.exists(args.output_path) and not args.overwrite:
        raise ValueError(
            f"输出路径已存在: {args.output_path}\n"
            "使用 --overwrite 参数来覆盖现有数据集"
        )
    
    print("=" * 60)
    print("合并 tokenized 数据集")
    print("=" * 60)
    print(f"数据集数量: {len(args.dataset_paths)}")
    print(f"合并策略: {args.mix_strategy}")
    print(f"输出路径: {args.output_path}")
    print("=" * 60)
    
    # Merge datasets
    merged_dataset = merge_tokenized_datasets(
        dataset_paths=args.dataset_paths,
        output_path=args.output_path,
        mix_strategy=args.mix_strategy,
        interleave_probs=args.interleave_probs,
        seed=args.seed,
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("合并结果:")
    print("=" * 60)
    for split_name, split_data in merged_dataset.items():
        print(f"  {split_name}: {len(split_data)} 样本")
    print("=" * 60)
    
    print(f"\n✓ 合并完成!")
    print(f"\n使用方法:")
    print(f"  在配置文件中设置: tokenized_path: {args.output_path}")


if __name__ == "__main__":
    merge_tokenized_datasets_cli()

