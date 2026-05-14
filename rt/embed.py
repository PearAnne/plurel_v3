import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import orjson
import strictfire
import torch
from ml_dtypes import bfloat16
from sentence_transformers import SentenceTransformer


def get_pre_dir(dataset_name: str) -> Path:
    home = Path(os.environ.get("HOME", "."))
    default_pre_root = home / "scratch" / "pre"
    pre_root = Path(os.environ.get("PLUREL_PRE_ROOT", str(default_pre_root))).expanduser()
    return pre_root / dataset_name


class TextEmbedder:
    def __init__(
        self,
        batch_size: int,
        embedding_model: str,
        device_type: str,
        compile_model: bool,
    ) -> None:
        self.model = SentenceTransformer(
            f"sentence-transformers/{embedding_model}",
            device=device_type,
            model_kwargs={
                "dtype": torch.bfloat16 if device_type == "cuda" else torch.float32,
            },
        )
        if compile_model:
            self.model = torch.compile(self.model)
        self.batch_size = batch_size

    def __call__(
        self,
        text_list: Sequence[str],
        device: str | list[str] | None = None,
    ) -> Any:
        return self.model.encode(
            text_list,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            device=device,
        )


def main(
    dataset_name: str,
    device: str | list[str] | None = None,
    batch_size: int = 8192,
    embedding_model: str = "all-MiniLM-L12-v2",
    compile_model: bool | None = None,
) -> None:
    if device is None:
        # Get list of all available CUDA devices
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            device = [f"cuda:{i}" for i in range(num_devices)]
            print(f"Found {num_devices} CUDA device(s): {device}")
        else:
            device = "cpu"
            print("No CUDA devices available, using CPU")

    if isinstance(device, list):
        device_type = torch.device(device[0]).type
    else:
        device_type = torch.device(device).type

    if compile_model is None:
        compile_model = device_type == "cuda"

    pre_dir = get_pre_dir(dataset_name)
    text_path = pre_dir / "text.json"
    with open(text_path) as f:
        raw = f.read()
    text_list = orjson.loads(raw)

    text_embedder = TextEmbedder(
        batch_size,
        embedding_model=embedding_model,
        device_type=device_type,
        compile_model=compile_model,
    )
    emb_list = text_embedder(text_list, device=device)

    emb_path = pre_dir / f"text_emb_{embedding_model}.bin"
    emb = np.stack(emb_list).astype(bfloat16)
    emb.tofile(emb_path)


if __name__ == "__main__":
    strictfire.StrictFire(main)
