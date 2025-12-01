from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

OverwritePolicy = Literal["ask", "overwrite", "skip"]


# ----------------------------- Utilities -----------------------------


def get_empty_device_cache_func(device: str):
    if device == "cuda":
        return _empyt_device_cache_cuda
    elif device == "mps" and hasattr(torch, "mps"):
        return _empyt_device_cache_mps
    else:
        return _empyt_device_cache_cpu


def _empyt_device_cache_cuda():
    torch.cuda.empty_cache()


def _empyt_device_cache_mps():
    torch.mps.empty_cache()


def _empyt_device_cache_cpu():
    # for the first time, do nothing
    pass


def _choose_device(preferred: str | None = None) -> str:
    """Bestes verfügbares Device ermitteln; MPS ist oft instabil -> Fallback."""
    if preferred:
        p = preferred.lower()
        if p == "cuda" and torch.cuda.is_available():
            return "cuda"
        if p == "mps" and torch.backends.mps.is_available():
            # Achtung: sentence-transformers + MPS ist je nach Modell/Torch-
            return "mps"
        if p == "cpu":
            return "cpu"
        raise ValueError(f"Unbekanntes preferred_device: {preferred}")
    # Auto
    if torch.cuda.is_available():
        return _choose_device(preferred="cuda")
    if torch.backends.mps.is_available():
        """Achtung: sentence-transformers + MPS ist je nach Modell/Torch-Version limitiert.
        Verwende CPU"""
        return _choose_device(preferred="cpu")
    return _choose_device(preferred="cpu")


def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _confirm_overwrite(path: Path) -> bool:
    """Interaktives Y/n für einzelne Zieldatei."""
    try:
        ans = (
            input(f"Datei existiert bereits: {path}\nÜberschreiben? [y/N]: ")
            .strip()
            .lower()
        )
    except EOFError:
        return False
    return ans in {"y", "yes", "j", "ja"}


def _should_write(path: Path, policy: OverwritePolicy) -> bool:
    if not path.exists():
        return True
    if policy == "overwrite":
        return True
    if policy == "skip":
        return False
    # policy == "ask"
    return _confirm_overwrite(path)


def _safe_write_memmap(dst: Path, shape: Tuple[int, int], dtype: np.dtype) -> np.memmap:
    _ensure_parent(dst)
    return np.memmap(dst, dtype=dtype, mode="w+", shape=shape)


class ColumnNameMapping:
    col_product_id: str = "product_id"
    col_query_id: str = "query_id"
    col_query_text: str = "query"
    col_title: str = "product_title"
    col_description: str = "product_description"
    col_bullets: str = "product_bullet_point"
    col_brand: str = "product_brand"


# ----------------------------- Base Class -----------------------------


@dataclass
class BaseEmbedder(ColumnNameMapping):
    """
    Generischer Embedding-Pipeline-Baustein für ESCI-kompatible Schemata.
    Kodiert Spalten als Embeddings in .mmap, legt ID-Maps als Parquet ab
    und schreibt Metadaten-JSON (reprozierbar & modellagnostisch).

    - Gerätemanagement: cuda/mps/cpu mit Fallback
    - Speichereffizient: np.memmap + wahlweise float16/fp32 Speicherung
    - Overwrite-Policy: "ask" (Default), "overwrite", "skip"
    """

    model_name: str
    pr_path: Path = field(
        default_factory=lambda: Path(
            "shopping_queries_dataset/reduced/shopping_queries_dataset_products_reduced.parquet"
        )
    )
    ex_path: Path = field(
        default_factory=lambda: Path(
            "shopping_queries_dataset/reduced/shopping_queries_dataset_examples_reduced.parquet"
        )
    )
    out_dir: Path = field(default_factory=lambda: Path("data/embeddings"))

    # Zusammenfassungstext optional generieren
    make_product_summary: bool = True
    product_summary_col: str = "product_summary"

    # Encoding-Parameter
    normalize: bool = False
    storage_dtype: np.dtype = np.float16
    use_device: str | None = field(init=True, default=None)
    trust_remote_code: bool = True

    # prompts (modellabhängig)
    prompt_name_queries: Optional[str] = None
    prompt_name_products: Optional[str] = None  # selten nötig

    # Overwrite
    overwrite: OverwritePolicy = "overwrite"

    # intern
    _model: SentenceTransformer = field(init=False)
    df_examples: pd.DataFrame = field(init=False)
    df_products: pd.DataFrame = field(init=False)

    # -------------------- Lifecycle --------------------

    def __post_init__(self) -> None:
        self.use_device = _choose_device(self.use_device)
        print(f"{self.use_device} wird verwendet")
        torch.set_grad_enabled(False)

        # Load model
        self._model = SentenceTransformer(
            self.model_name,
            device=self.use_device,
            trust_remote_code=self.trust_remote_code,
        )

        # Warnung bei MPS (gelegentliche Kernel-Lücken)
        if self.use_device == "mps":
            print(
                "[Warnung] MPS ist aktiv. Manche sentence-transformers/Operatoren sind auf MPS "
                "nicht vollständig implementiert. Bei Fehlern bitte mit use_device='cpu' neu starten.",
                file=sys.stderr,
            )

        # Load data
        self.df_examples = pd.read_parquet(self.ex_path)
        self.df_products = pd.read_parquet(self.pr_path)

    # -------------------- Public API --------------------

    def run_all(
        self,
        product_text_cols: Sequence[str] | None = None,
        write_products: bool = True,
        write_queries: bool = True,
    ) -> Dict[str, Dict[str, int]]:
        """
        Führt Embedding für Produkte & Queries aus und schreibt alle Artefakte.
        Returns: Dims je Feld ({'products': {col: dim}, 'queries': {'query': dim}})
        """

        out_root = self.get_out_root()
        prod_out = out_root / "products"
        query_out = out_root / "queries"
        prod_out.mkdir(parents=True, exist_ok=True)
        query_out.mkdir(parents=True, exist_ok=True)

        dims: Dict[str, Dict[str, int]] = {"products": {}, "queries": {}}

        product_text_cols = list(product_text_cols or self.default_product_text_cols())
        # --- optional Produktzusammenfassung erzeugen ---
        pr = self.df_products
        if self.make_product_summary and self.product_summary_col not in pr.columns:
            pr = self.add_product_summary(pr)
            if self.product_summary_col not in product_text_cols:
                product_text_cols.insert(0, self.product_summary_col)

        product_text_cols = [c for c in product_text_cols if c]

        # --- Queries: de-duped & Embeddings ---
        if write_queries:
            exu = self.df_examples[[self.col_query_id, self.col_query_text]]
            self._write_id_map(
                df=exu,
                id_col=self.col_query_id,
                dst=query_out / "query_ids.parquet",
            )
            dim_q = self._encode_series_to_memmap(
                texts=exu[self.col_query_text].astype(str),
                mmap_path=query_out / f"{self.col_query_text}.mmap",
                prompt_name=self.prompt_name_queries,
                max_seq_length=39,
            )
            dims["queries"][self.col_query_text] = dim_q

            # Meta
            self._write_queries_meta(
                out_root,
                query_out,
                dim_q,
                num_rows=len(exu),
            )

        # --- Produkte: IDs & Embeddings ---
        max_seq_lengths = {
            self.col_title: 258,
            self.col_description: 2359,
            self.col_bullets: 1884,
            self.col_brand: 38,
            self.product_summary_col: 3488,
        }
        if write_products:
            self._write_id_map(
                df=pr,
                id_col=self.col_product_id,
                dst=prod_out / "product_ids.parquet",
            )
            for col in tqdm(product_text_cols):
                dim = self._encode_series_to_memmap(
                    texts=(
                        pr[col].astype(str)
                        if col in pr.columns
                        else pd.Series([""] * len(pr))
                    ),
                    mmap_path=prod_out / f"{col}.mmap",
                    prompt_name=self.prompt_name_products,
                    max_seq_length=max_seq_lengths[col],
                )
                dims["products"][col] = dim

            # Meta
            self._write_products_meta(
                out_root,
                prod_out,
                dims["products"],
                num_rows=len(pr),
            )

        print("Fertig. Embeddings unter", out_root.resolve())
        return dims

    @classmethod
    def add_product_summary(cls, pr: pd.DataFrame) -> pd.DataFrame:
        pr[cls.product_summary_col] = pr.apply(cls._build_product_summary, axis=1)
        return pr

    def get_out_root(self) -> Path:
        out_sufix = self._get_out_sufix()
        out_root = self.out_dir / f"{self.model_name.replace('/', '_')}_{out_sufix}"
        return out_root

    def _get_out_sufix(self) -> str:
        sufix = f"{self.use_device}"
        param = f"{self.get_encode_kwargs()}"
        param_hash = hashlib.md5(param.encode("utf-8")).hexdigest()[:8]
        return f"{sufix}_{param_hash}"

    # -------------------- Defaults & Builders --------------------

    def default_product_text_cols(self) -> List[str]:
        return [
            self.product_summary_col if self.make_product_summary else None,
            self.col_title,
            self.col_description,
            self.col_bullets,
            self.col_brand,
        ][
            :
        ]  # shallow copy; None wird später entfernt

    @classmethod
    def _build_product_summary(cls, row: pd.Series) -> str:
        t = row.get(cls.col_title, "")
        d = row.get(cls.col_description, "")
        b = row.get(cls.col_bullets, "")
        br = row.get(cls.col_brand, "")
        return f"Title: {t}\nBullet Point: {b}\nDescription: {d}\nBrand: {br}"

    # -------------------- IO Writers --------------------

    def _write_id_map(self, df: pd.DataFrame, id_col: str, dst: Path) -> None:
        if not _should_write(dst, self.overwrite):
            print(f"[Skip] {dst}")
            return
        _ensure_parent(dst)
        pd.DataFrame(
            {id_col: df[id_col].astype(str).values, "source_index": df.index.values}
        ).to_parquet(dst, index=False)

    def _write_products_meta(
        self,
        out_root: Path,
        prod_out: Path,
        field_dims: Dict[str, int],
        num_rows: int,
    ) -> None:
        meta_path = out_root / "products_meta.json"
        if not _should_write(meta_path, self.overwrite):
            print(f"[Skip] {meta_path}")
            return
        payload = {
            "type": "products",
            "model_name": self.model_name,
            **self.get_encode_kwargs(),
            "num_rows": int(num_rows),
            "ids_path": str((prod_out / "product_ids.parquet").relative_to(out_root)),
            "id_col": self.col_product_id,
            "source_index_col": "source_index",
            "dataset_products_path": str(self.pr_path),
            "fields": [
                {
                    "name": name,
                    "dim": int(dim),
                    "path": f"products/{name}.mmap",
                }
                for name, dim in field_dims.items()
            ],
        }
        _ensure_parent(meta_path)
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _write_queries_meta(
        self,
        out_root: Path,
        query_out: Path,
        query_dim: int,
        num_rows: int,
    ) -> None:
        meta_path = out_root / "queries_meta.json"
        if not _should_write(meta_path, self.overwrite):
            print(f"[Skip] {meta_path}")
            return
        payload = {
            "type": "queries",
            "model_name": self.model_name,
            **self.get_encode_kwargs(),
            "num_rows": int(num_rows),
            "ids_path": str((query_out / "query_ids.parquet").relative_to(out_root)),
            "id_col": self.col_query_id,
            "source_index_col": "source_index",
            "dataset_examples_path": str(self.ex_path),
            "field": {
                "name": self.col_query_text,
                "dim": int(query_dim),
                "path": f"queries/{self.col_query_text}.mmap",
            },
        }
        _ensure_parent(meta_path)
        with open(meta_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _encode_series_to_memmap(
        self,
        texts: pd.Series,
        mmap_path: Path,
        prompt_name: Optional[str],
        max_seq_length: Optional[int] = None,
    ) -> int:
        """Encodiert eine Serie nach .mmap und gibt die Dim zurück (ohne Trunkation)."""
        # 1) Clean Input
        s = texts.fillna("").astype(str)

        # 2) Encode-Parameter
        model_kwargs = self.get_encode_kwargs(prompt_name=prompt_name)
        model_kwargs["convert_to_numpy"] = True
        model = self._model
        device = self.use_device
        tok = model.tokenizer

        # 3) Batchgröße aus geschätzter Max-Sequenzlänge ableiten
        env = os.environ.get("MAX_TOKENS_PER_BATCH")
        max_tokens_per_batch = (
            int(env) if env else (2**15 if device == "cpu" else 2**16)
        )
        print(f"Max Tokens per Batch: {max_tokens_per_batch}")
        # BUGFIX: s statt texts tokenisieren
        if max_seq_length is None:
            tokenized = tok(s.tolist(), truncation=False)
            input_ids = tokenized.get("input_ids", [])
            texts_lens = [len(ids) for ids in input_ids]
            max_seq_length = max(texts_lens) if texts_lens else 1
        batch_size = max(1, max_tokens_per_batch // max_seq_length)

        # 4) Probe, um Embedding-Dimension zu ermitteln
        probe = model.encode(["__probe__"], batch_size=1, **model_kwargs)
        if probe.ndim != 2:
            raise RuntimeError("Probe-Embedding hat unerwartete Form.")
        dim = int(probe.shape[1])

        # 5) Überspringen, falls Datei existiert und overwrite=False
        if not _should_write(mmap_path, self.overwrite):
            print(f"[Skip] {mmap_path}")
            return dim

        mm = _safe_write_memmap(
            mmap_path, shape=(len(s), dim), dtype=self.storage_dtype
        )
        empty_device_cache_func = get_empty_device_cache_func(device)

        # 6) Encoding in Batches
        for start_idx in tqdm(
            range(0, len(s), batch_size),
            desc=f"Encoding to {mmap_path.name}",
            total=(len(s) + batch_size - 1) // batch_size,
        ):
            end_idx = min(start_idx + batch_size, len(s))
            batch_texts = s.iloc[start_idx:end_idx].tolist()
            emb_batch = model.encode(
                batch_texts,
                batch_size=batch_size,
                **model_kwargs,
            )
            mm[start_idx:end_idx, :] = emb_batch.astype(self.storage_dtype, copy=False)
            empty_device_cache_func()

        mm.flush()
        del mm
        return dim

    def get_encode_kwargs(self, prompt_name: Optional[str] = None) -> dict:
        """Hilfsmethode für Unterklassen, um encode()-Parameter zu spezifizieren."""
        kwargs = {
            "device": self.use_device,
            "normalize_embeddings": self.normalize,
            "convert_to_numpy": True,
            "show_progress_bar": False,
        }
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        return kwargs

    def get_count_tokens_examples_products(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Hilfsmethode zum Zählen der Tokens in Examples und Products DataFrames."""
        tok = self._model.tokenizer

        def count_tokens_in_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
            if col not in df.columns:
                return pd.DataFrame()
            token_counts = (
                df[col]
                .fillna("")
                .astype(str)
                .apply(lambda x: len(tok(x, truncation=False)["input_ids"]))
            )
            return pd.DataFrame({f"{col}_token_count": token_counts})

        ex_token_counts = count_tokens_in_column(self.df_examples, self.col_query_text)
        pr_token_counts = pd.DataFrame()
        for col in self.default_product_text_cols():
            if col:
                col_counts = count_tokens_in_column(self.df_products, col)
                pr_token_counts = pd.concat([pr_token_counts, col_counts], axis=1)

        return ex_token_counts, pr_token_counts


# ---------------------- Model-specific Embedder ----------------------


@dataclass
class JinaSmallEnEmbedder(BaseEmbedder):
    """
    Preset für jinaai/jina-embeddings-v2-small-en.
    - Keine speziellen prompt_name-Vorgaben
    - Speicherökonomische Defaults (fp16 Speicherung)
    - Standard: CPU (stabil), aber CUDA wird automatisch erkannt, wenn verfügbar.
    """

    model_name: str = "jinaai/jina-embeddings-v2-small-en"


@dataclass
class Qwen3Embedding06BEmbedder(BaseEmbedder):
    """
    Preset für Qwen/Qwen3-Embedding-0.6B.
    - Empfohlene Prompt-Namen:
        * Queries:  "query"
        * Produkte: "passage"  (üblich für Dokument-/Passage-Embeddings)
    - MPS kann instabil sein -> Default lieber CPU; CUDA wenn vorhanden.
    """

    model_name: str = "Qwen/Qwen3-Embedding-0.6B"
    prompt_name_queries: Optional[str] = "query"
    prompt_name_products: Optional[str] = None


def run_models(models: List[type[BaseEmbedder]]) -> None:
    for emb_class in tqdm(models, desc="Embedders"):
        begin_time = time.time()
        emb = emb_class()
        emb.run_all()
        end_time = time.time()
        print(
            f"{emb_class.__name__} abgeschlossen in {end_time - begin_time:.2f} Sekunden."
        )
        print("\n" * 3)


if __name__ == "__main__":
    """
    poetry run python embedder.py "all"
    """
    # arg parsing etc. kann hier ergänzt werden
    # if arg -model jinai -> JinaSmallEnEmbedder()
    # if arg -model qwen -> Qwen3Embedding06BEmbedder()
    begin_time = time.time()

    model_arg = sys.argv[1].lower()
    if model_arg == "jina":
        models = [JinaSmallEnEmbedder]
    elif model_arg == "qwen":
        models = [Qwen3Embedding06BEmbedder]
    elif model_arg == "all":
        models = [JinaSmallEnEmbedder, Qwen3Embedding06BEmbedder]
    else:
        print("Unbekanntes Modell-Argument. Bitte 'jina' oder 'qwen' verwenden.")
        sys.exit(1)
    start_time = time.time()
    run_models(models)
    end_time = time.time()
    print(f"Gesamtzeit: {end_time - start_time:.2f} Sekunden.")
    exit(0)
