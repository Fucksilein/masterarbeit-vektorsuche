# ------------------------------------------------------------
# Metadaten / Stores
# ------------------------------------------------------------


from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from metrics import ESCI_TO_GAIN_DEFAULT

# --- Torch/CUDA optional ---
USE_TORCH = torch.cuda.is_available()


@dataclass
class FieldSpec:
    name: str
    dim: int
    n_rows: int
    path: Path
    dtype: np.dtype = field(init=False, default=np.float16)

    @classmethod
    def from_meta_dict(
        cls, meta_dict: dict, meta_dir: Path
    ) -> Union["FieldSpec", List["FieldSpec"]]:
        num_rows = meta_dict["num_rows"]

        def from_field_dict(field_dict: Dict) -> "FieldSpec":
            name = field_dict["name"]
            dim = int(field_dict["dim"])
            path = meta_dir / field_dict["path"]
            return FieldSpec(name=name, dim=dim, path=path, n_rows=num_rows)

        if "fields" in meta_dict:
            return [from_field_dict(f) for f in meta_dict["fields"]]
        elif "field" in meta_dict:
            return from_field_dict(meta_dict["field"])
        else:
            raise ValueError(
                "FieldSpec: meta_dict must contain 'field' or 'fields' key"
            )


@dataclass
class BaseMeta:
    meta_file: Path
    type: str
    model_name: str
    device: str
    normalize: bool
    num_rows: int
    id_col: str
    source_index_col: str
    ids_df: pd.DataFrame

    @staticmethod
    def load_dict(meta_dir: Path, filename: str) -> Dict:
        meta_dir_filename = meta_dir / filename
        meta = json.loads(meta_dir_filename.read_text())
        # konvert ids path to unixe if necessary
        # meta["ids_path"] = convert_to_unixe_path(meta["ids_path"])
        # lade
        ids_path = meta_dir / meta["ids_path"].replace("\\", "/")

        ids_df = pd.read_parquet(ids_path)
        base_dict = {
            "meta_file": meta_dir_filename,
            "type": str(meta["type"]),
            "model_name": meta["model_name"],
            "device": str(meta["device"]),
            "normalize": bool(meta["normalize_embeddings"]),
            "num_rows": int(meta["num_rows"]),
            "id_col": str(meta["id_col"]),
            "source_index_col": str(meta["source_index_col"]),
            "ids_df": ids_df,
        }
        return base_dict


@dataclass
class ProductsMeta(BaseMeta):
    fields: List[FieldSpec]
    products_path: Path
    meta_dir: Path

    @staticmethod
    def load(meta_dir: Path) -> "ProductsMeta":
        filename = "products_meta.json"
        base_dict = BaseMeta.load_dict(meta_dir, filename)
        meta = json.loads((meta_dir / filename).read_text())
        fields = FieldSpec.from_meta_dict(meta, meta_dir)
        products_path = meta["dataset_products_path"]
        return ProductsMeta(
            **base_dict,
            fields=fields,
            products_path=products_path,
            meta_dir=meta_dir,
        )

    def to_dict(self) -> Dict:
        return {
            "meta_file": str(self.meta_file),
            "type": self.type,
            "model_name": self.model_name,
            "device": self.device,
            "normalize": self.normalize,
            "num_rows": self.num_rows,
            "id_col": self.id_col,
            "source_index_col": self.source_index_col,
            "ids_df_path": str(self.ids_df),
            "fields": [
                {
                    "name": field.name,
                    "dim": field.dim,
                    "n_rows": field.n_rows,
                    "path": str(field.path),
                    "dtype": str(field.dtype),
                }
                for field in self.fields
            ],
            "products_path": str(self.products_path),
        }


@dataclass
class QueriesMeta(BaseMeta):
    example_df: pd.DataFrame
    field_spec: FieldSpec

    def __post_init__(self):
        self.example_df = self.example_df.astype({self.id_col: str})

    @staticmethod
    def load(meta_dir: Path) -> "QueriesMeta":
        filename = "queries_meta.json"
        base_dict = BaseMeta.load_dict(meta_dir, filename)
        meta = json.loads((meta_dir / filename).read_text())
        field_spec = FieldSpec.from_meta_dict(meta, meta_dir)
        example_df = pd.read_parquet(meta["dataset_examples_path"].replace("\\", "/"))
        return QueriesMeta(
            **base_dict,
            field_spec=field_spec,
            example_df=example_df,
        )


@dataclass
class EmbeddingStore:
    mm: np.ndarray  # (N, D), memmap oder ndarray
    dtype: np.dtype
    shape: Tuple[int, int]
    is_normalized: None | bool = field(default=False)
    mm_tensor: None | torch.Tensor = field(init=False, repr=False, default=None)

    def __post_init__(self):
        if self.mm.shape != self.shape:
            raise ValueError(
                f"Shape mismatch: expected {self.shape}, got {self.mm.shape}"
            )
        if self.mm.dtype != self.dtype:
            raise ValueError(
                f"Dtype mismatch: expected {self.dtype}, got {self.mm.dtype}"
            )
        if self.dtype not in (np.float16, np.float32):
            raise ValueError(
                f"Unsupported dtype: {self.dtype}, only float16/float32 supported"
            )

    @classmethod
    def from_field_spec(cls, field_spec: FieldSpec) -> "EmbeddingStore":
        mm = np.memmap(
            field_spec.path,
            dtype=field_spec.dtype,
            mode="r",
            shape=(field_spec.n_rows, field_spec.dim),
        )
        return cls(mm, mm.dtype, mm.shape)

    @classmethod
    def from_array(cls, array: np.ndarray) -> "EmbeddingStore":
        arr = np.asarray(array)
        return cls(arr, arr.dtype, arr.shape)

    @staticmethod
    def mm_to_tensor(mm: np.ndarray) -> torch.Tensor:
        if mm.dtype == np.float16:
            return torch.from_numpy(mm).to(torch.float16)
        elif mm.dtype == np.float32:
            return torch.from_numpy(mm).to(torch.float32)
        else:
            raise ValueError(
                f"Unsupported dtype: {mm.dtype}, only float16/float32 supported"
            )

    @property
    def mm_as_tensor(self) -> torch.Tensor:
        if self.mm_tensor is None:
            self.mm_tensor = self.mm_to_tensor(self.mm)
        return self.mm_tensor

    def normalize_all(self):
        if not self.is_normalized:
            norms = np.linalg.norm(self.mm, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Vermeide Division durch Null
            self.mm = self.mm / norms
            if self.mm_tensor is not None:
                self.mm_tensor = None  # Tensor neu erstellen
            self.is_normalized = True

    def get_block(
        self, start: int, end: int, as_float32: bool = True, use_torch: bool = False
    ) -> torch.Tensor | np.ndarray:
        if use_torch:
            view = self.mm_as_tensor[start:end]
            return (
                view.to(torch.float32)
                if (as_float32 and view.dtype != torch.float32)
                else view
            ).to(device="cuda", non_blocking=True)

        else:
            view = self.mm[start:end]
            return (
                view.astype(np.float32, copy=False)
                if (as_float32 and view.dtype != np.float32)
                else view
            )

    def get_rows_by_indices(
        self, indices: Iterable[int], as_float32: bool = True, batch_size: int = 2**16
    ) -> np.ndarray:
        indices = list(map(int, indices))
        D = self.shape[1]
        out = np.empty(
            (len(indices), D), dtype=np.float32 if as_float32 else self.dtype
        )
        pos = 0
        for i in range(0, len(indices), batch_size):
            idx_batch = indices[i : i + batch_size]
            batch = self.mm[idx_batch]
            if as_float32 and batch.dtype != np.float32:
                batch = batch.astype(np.float32, copy=False)
            out[pos : pos + len(idx_batch)] = batch
            pos += len(idx_batch)
        return out

    def get_row(self, i: int, as_float32: bool = True) -> np.ndarray:
        return self.get_block(i, i + 1, as_float32=as_float32)[0]

    def __len__(self) -> int:
        return self.shape[0]

    def get_substore_by_rows(self, cand_rows):
        sub_mm = self.get_rows_by_indices(cand_rows, as_float32=False)
        emb_store = EmbeddingStore.from_array(sub_mm)
        emb_store.is_normalized = self.is_normalized
        return emb_store


@dataclass
class ProductFieldStore:
    name: str
    store: EmbeddingStore  # (N, D)


@dataclass
class QueryStore:
    queries: QueriesMeta
    store: EmbeddingStore  # (N, D)
    id_to_row: Dict[str, int]
    row_to_id: Dict[int, str]
    query_df: pd.DataFrame = field(init=False)

    def __post_init__(self) -> None:
        self.store.normalize_all()
        ids_df = self.queries.ids_df
        ex = self.queries.example_df

        # 1) Konsistenzprüfungen Wide-Format
        required_cols = {self.queries.id_col, "product_id", "esci_label"}
        missing = required_cols - set(ex.columns)
        if missing:
            raise ValueError(
                f"QueryStore: fehlende Spalten im example_df: {sorted(missing)}"
            )

        if set(ids_df[self.queries.id_col]) != set(ex[self.queries.id_col]):
            raise ValueError(
                "QueryStore: IDs in example_df und ids_df unterscheiden sich"
            )
        if len(ids_df) != len(ex):
            raise ValueError(
                "QueryStore: Length mismatch zwischen example_df und ids_df"
            )

        # Wide: exakt 1 Zeile pro query_id in ex
        query_df = ex.merge(
            ids_df[[self.queries.id_col, "source_index"]],
            on=self.queries.id_col,
            how="inner",
            validate="one_to_one",
        )
        # Saubere Index-Setzung
        if query_df[self.queries.id_col].duplicated().any():
            dups = (
                query_df[self.queries.id_col][
                    query_df[self.queries.id_col].duplicated()
                ]
                .unique()
                .tolist()
            )
            raise ValueError(f"QueryStore: duplicate query_ids im Wide-DF: {dups}")

        # 3) Listen-Spalten normalisieren (np.array/tuple → list; None → [])
        def _ensure_list(x: Any) -> List[Any]:
            if isinstance(x, list):
                return x
            if isinstance(x, (tuple, set)):
                return list(x)
            if isinstance(x, np.ndarray):
                return x.tolist()
            # Falls versehentlich skalar hereinkommt, raise
            raise ValueError(f"_ensure_list: unerwarteter Typ {type(x)} für Wert {x}")

        query_df["product_id"] = query_df["product_id"].apply(_ensure_list)
        query_df["esci_label"] = query_df["esci_label"].apply(_ensure_list)

        # 4) Index setzen
        query_df = query_df.set_index(self.queries.id_col, drop=False)
        self.query_df = query_df

        # 5) Gain berechnen
        self.add_esci_to_gain(ESCI_TO_GAIN_DEFAULT)

    def add_esci_to_gain(self, esc_to_gain: Dict[str, float]) -> None:
        """Mappt ESCI-Label-Listen je Query zu Gain-Listen gleicher Länge."""
        if not esc_to_gain:
            raise ValueError("add_esci_to_gain: leeres Mapping esc_to_gain")
        min_gain = float(min(esc_to_gain.values()))

        def map_list(labels: Sequence[Any]) -> List[float]:
            return [float(esc_to_gain.get(str(lbl), min_gain)) for lbl in labels]

        gains = self.query_df["esci_label"].map(map_list)
        self.query_df["gain"] = gains

        # Längen-Konsistenz
        len_pid = self.query_df["product_id"].map(len)
        len_gain = self.query_df["gain"].map(len)
        mismatch_mask = len_pid != len_gain
        if bool(mismatch_mask.any()):
            bad_qids = self.query_df.index[mismatch_mask].tolist()
            raise ValueError(f"Längen-Mismatch product_id/gain für qids: {bad_qids}")

    @staticmethod
    def load(base_dir: Path) -> "QueryStore":
        queries = QueriesMeta.load(base_dir)
        store = EmbeddingStore.from_field_spec(field_spec=queries.field_spec)
        id_to_row = {
            str(qid): int(i)
            for i, qid in enumerate(queries.ids_df[queries.id_col].astype(str).values)
        }
        row_to_id = {
            int(i): str(qid)
            for i, qid in enumerate(queries.ids_df[queries.id_col].astype(str).values)
        }
        return QueryStore(queries, store, id_to_row, row_to_id)

    def get_vectors_for_ids(
        self, qids: List[str], as_float32: bool = True
    ) -> np.ndarray:
        rows = [self.id_to_row[str(q)] for q in qids]
        # nur die benötigten Zeilen laden (kein Full-Load!)
        return self.store.get_rows_by_indices(rows, as_float32=as_float32)

    def get_query_sample(
        self, n: int, random_state: int = 42
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Zufallsstichprobe von n Queries (Vektor + ID-Liste), ohne den vollen Store in RAM zu ziehen.
        """
        rng = np.random.default_rng(random_state)
        all_unique_qids = list(self.id_to_row.keys())
        indices = rng.choice(all_unique_qids, size=n, replace=False)
        rows = [self.id_to_row[qid] for qid in indices]
        vectors = self.store.get_rows_by_indices(rows, as_float32=True)
        ids = [self.row_to_id[r] for r in rows]
        return vectors, ids

    def get_pids_esci_gain_for_id(self, qid: str) -> Dict[str, float]:
        qid_row = self.query_df.loc[qid, :]
        pid_to_gain = dict(
            zip(
                qid_row["product_id"],
                qid_row["gain"],
            )
        )
        return pid_to_gain


def get_base_dirs(model_names_input: str) -> List[Path]:
    embedding_path = Path("data/embeddings/")
    base_dirs = []
    if model_names_input == "all":
        model_names_input = "Qwen+jinaai"
    for p in embedding_path.iterdir():
        for model in model_names_input.split("+"):
            if p.is_dir() and model in p.name:
                base_dirs.append(p)
    return base_dirs
