# method_comparison.py
from __future__ import annotations

import argparse
import json
import logging
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from meta_classes import (
    USE_TORCH,
    EmbeddingStore,
    ProductFieldStore,
    ProductsMeta,
    QueryStore,
    get_base_dirs,
)
from metrics import ESCI_TO_GAIN_DEFAULT, get_all_metrics

AVAILABLE_METHODS = ["1", "2", "3", "4"]

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Konfiguration / Defaults
# ------------------------------------------------------------
if USE_TORCH:
    DEFAULT_PRODUCT_BLOCK = 2**16  # Größe der Produktblöcke für Matrix-Multiplikation
    DEFAULT_QUERY_BLOCK = 2**10  # Größe der Query-Blöcke
else:
    DEFAULT_PRODUCT_BLOCK = 2**14  # Größe der Produktblöcke für Matrix-Multiplikation
    DEFAULT_QUERY_BLOCK = 2**8  # Größe der Query-Blöcke


def _init_topk_torch(nq: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    scores = torch.full((nq, k), float("-inf"), device="cuda", dtype=torch.float32)
    idxs = torch.full((nq, k), -1, device="cuda", dtype=torch.int64)
    return scores, idxs


# ------------------------------------------------------------
# Methoden / Suche (ohne Index)
# ------------------------------------------------------------
class VectorSearchMethodStoreStatic:
    @staticmethod
    def vars_for_vsms(
        base_dir: pathlib.Path,
    ) -> Tuple[
        Dict[str, ProductFieldStore],
        Dict[str, int],
        ProductsMeta,
    ]:
        """
        Lädt Produkte, Felder und Gewichte für Methode 2/3.
        """
        products_meta = ProductsMeta.load(base_dir)

        pid_to_row = {
            str(pid): int(i)
            for i, pid in enumerate(
                products_meta.ids_df[products_meta.id_col].astype(str).values
            )
        }

        fields: Dict[str, ProductFieldStore] = {}
        for f in products_meta.fields:
            fields[f.name] = ProductFieldStore(
                f.name,
                EmbeddingStore.from_field_spec(f),
            )

        return fields, pid_to_row, products_meta

    @staticmethod
    def _load_weights_from_json(
        base_dir: pathlib.Path, methode: str, catalog: bool
    ) -> Dict[str, float]:
        """
        Lädt die besten Gewichte aus JSON. Fallback:
        - Falls Datei 2 fehlt -> versuche Datei 3
        - Falls Datei 3 fehlt -> leeres Dict + Error-Log
        """

        fname = (
            f"best_weights_method_fast{methode}.json"
            if catalog
            else f"best_weights_method_fast{methode}_on_candidates.json"
        )
        p = base_dir / fname
        if not p.exists():
            if methode == "2":
                return VectorSearchMethodStoreStatic._load_weights_from_json(
                    base_dir, "3", catalog=catalog
                )
            else:
                logger.error(
                    f"Gewichtungsdatei '{fname}' nicht gefunden im Verzeichnis '{base_dir}'."
                )
                return {}
        obj = json.loads(p.read_text())
        return {k: float(v) for k, v in obj.get("weights", {}).items()}

    @staticmethod
    def get_methode_2_product_field_store_static(
        fields: Dict[str, ProductFieldStore],
        names_weights_dict: Dict[str, float],
    ) -> ProductFieldStore:
        """
        Methode 2: gewichtete Summation der Feld-Embeddings mit L2‑Renormierung.
        """
        if not names_weights_dict:
            raise ValueError(
                "weight_method_2 ist leer. Mindestens ein Feld muss gewichtet werden."
            )

        vector_dim = next(iter(fields.values())).store.shape[1]
        product_count = next(iter(fields.values())).store.shape[0]
        combined = np.zeros((product_count, vector_dim), dtype=np.float32)
        for fname, w in names_weights_dict.items():
            if fname not in fields:
                raise KeyError(f"Feld '{fname}' nicht in 'fields' enthalten")

            current_store = fields[fname].store
            if current_store.is_normalized:
                raise ValueError(
                    f"EmbeddingStore für Feld '{fname}' darf nicht normalisiert sein für Methode 2"
                )
            mat = current_store.get_block(0, product_count, as_float32=True)
            combined += float(w) * mat
        embedding_store = EmbeddingStore.from_array(combined)
        embedding_store.normalize_all()
        return ProductFieldStore(name="combined_method_2", store=embedding_store)

    # Method 1: summary only and Method 2: weighted add embeddings (renormiert)
    @staticmethod
    def search_summary_streaming(
        query_vectors: np.ndarray, store: "EmbeddingStore", top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not store.is_normalized:
            raise ValueError(
                "EmbeddingStore for search_summary_streaming must be normalized"
            )
        if USE_TORCH and torch.cuda.is_available():
            return VectorSearchMethodStoreStatic._search_summary_streaming_torch(
                query_vectors, store, top_k
            )
        else:
            return VectorSearchMethodStoreStatic._search_summary_streaming_cpu(
                query_vectors, store, top_k
            )

    @staticmethod
    def _search_summary_streaming_torch(
        query_vectors: np.ndarray, store: "EmbeddingStore", top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if not store.is_normalized:
            raise ValueError("EmbeddingStore must be normalized for torch search")
        Q = int(query_vectors.shape[0])
        P = int(store.shape[0])
        K = int(min(top_k, P))

        q_tensor = torch.from_numpy(query_vectors).to(
            device="cuda", dtype=torch.float32, non_blocking=True
        )
        best_vals_global, best_idx_global = _init_topk_torch(Q, K)

        default_query_block = DEFAULT_QUERY_BLOCK
        default_product_block = DEFAULT_PRODUCT_BLOCK
        max_sims = default_query_block * default_product_block

        if Q < default_query_block:
            # Rechne Blockgrößen dynamisch, aber clamp auf P
            new_prod_block = int(max_sims / max(Q, 1))
            default_product_block = min(max(new_prod_block, 1), P)
            default_query_block = Q

        if P < default_product_block:
            new_query_block = int(max_sims / max(P, 1))
            default_query_block = min(max(new_query_block, 1), Q)
            default_product_block = P

        for q_start in range(0, Q, default_query_block):
            q_end = min(q_start + default_query_block, Q)
            q_block = q_tensor[q_start:q_end]  # [Qb, D]

            best_vals, best_idx = _init_topk_torch(q_block.size(0), K)

            for p_start in range(0, P, default_product_block):
                p_end = min(p_start + default_product_block, P)
                prod_block = store.get_block(
                    p_start, p_end, as_float32=True, use_torch=True
                )  # [B, D] (GPU)

                sims_block = q_block @ prod_block.T  # [Qb, B]
                kb = min(K * 2, sims_block.size(1))  # leichtes Overfetching
                vals, idx_local = torch.topk(
                    sims_block, k=kb, dim=1, largest=True, sorted=False
                )
                idx_global = idx_local + p_start

                best_idx, best_vals = VectorSearchMethodStoreStatic._merge_topk_torch(
                    best_idx, best_vals, idx_global, vals, K
                )

            best_idx_global[q_start:q_end, :] = best_idx
            best_vals_global[q_start:q_end, :] = best_vals

        return best_idx_global.cpu().numpy(), best_vals_global.cpu().numpy()

    @staticmethod
    def _search_summary_streaming_cpu(
        query_vectors: np.ndarray, store: "EmbeddingStore", top_k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        Q = int(query_vectors.shape[0])
        P = int(store.shape[0])
        K = int(min(top_k, P))

        best_vals_global = np.full((Q, K), -np.inf, dtype=np.float32)
        best_idx_global = np.full((Q, K), -1, dtype=np.int64)
        default_query_block = DEFAULT_QUERY_BLOCK
        default_product_block = DEFAULT_PRODUCT_BLOCK
        max_sims = default_query_block * default_product_block

        if Q < default_query_block:
            # Rechne Blockgrößen dynamisch, aber clamp auf P
            new_prod_block = int(max_sims / max(Q, 1))
            default_product_block = min(max(new_prod_block, 1), P)
            default_query_block = Q

        if P < default_product_block:
            new_query_block = int(max_sims / max(P, 1))
            default_query_block = min(max(new_query_block, 1), Q)
            default_product_block = P
        for q_start in range(0, Q, default_query_block):
            q_end = min(q_start + default_query_block, Q)
            q_block = query_vectors[q_start:q_end]  # (Qb, D)

            best_vals = np.full((q_block.shape[0], K), -np.inf, dtype=np.float32)
            best_idx = np.full((q_block.shape[0], K), -1, dtype=np.int64)

            for p_start in range(0, P, default_product_block):
                p_end = min(p_start + default_product_block, P)
                prod_block = store.get_block(
                    p_start, p_end, as_float32=True, use_torch=False
                )  # (B, D)
                sims_block = q_block @ prod_block.T  # (Qb, B)

                k_block = min(K, sims_block.shape[1])
                b_idx_local = np.argpartition(-sims_block, k_block - 1, axis=1)[
                    :, :k_block
                ]
                b_val = np.take_along_axis(sims_block, b_idx_local, axis=1)
                b_idx = b_idx_local + p_start

                best_idx, best_vals = VectorSearchMethodStoreStatic._merge_topk(
                    best_idx, best_vals, b_idx, b_val, K
                )

            best_idx_global[q_start:q_end] = best_idx
            best_vals_global[q_start:q_end] = best_vals

        return best_idx_global, best_vals_global

    # Method 3: weighted sum of per-field scores
    @staticmethod
    def search_sum_weighted_similarity_streaming(
        query_vectors: np.ndarray,
        field_weight_stores: List[Tuple[str, float, "EmbeddingStore"]],
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if USE_TORCH and torch.cuda.is_available():
            return VectorSearchMethodStoreStatic._search_sum_weighted_similarity_streaming_torch(
                query_vectors, field_weight_stores, top_k
            )
        else:
            return VectorSearchMethodStoreStatic._search_sum_weighted_similarity_streaming_cpu(
                query_vectors, field_weight_stores, top_k
            )

    @staticmethod
    def _search_sum_weighted_similarity_streaming_torch(
        query_vectors: np.ndarray,
        field_weight_stores: List[Tuple[str, float, "EmbeddingStore"]],
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        Q = int(query_vectors.shape[0])
        P = int(field_weight_stores[0][2].shape[0])
        K = int(min(top_k, P))

        q_tensor = torch.from_numpy(query_vectors).to(
            device="cuda", dtype=torch.float32, non_blocking=True
        )
        best_vals_global, best_idx_global = _init_topk_torch(Q, K)
        default_query_block = DEFAULT_QUERY_BLOCK
        default_product_block = DEFAULT_PRODUCT_BLOCK
        max_sims = default_query_block * default_product_block

        if Q < default_query_block:
            # Rechne Blockgrößen dynamisch, aber clamp auf P
            new_prod_block = int(max_sims / max(Q, 1))
            default_product_block = min(max(new_prod_block, 1), P)
            default_query_block = Q

        if P < default_product_block:
            new_query_block = int(max_sims / max(P, 1))
            default_query_block = min(max(new_query_block, 1), Q)
            default_product_block = P
        for q_start in range(0, Q, default_query_block):
            q_end = min(q_start + default_query_block, Q)
            q_block = q_tensor[q_start:q_end]  # [Qb, D]

            best_vals, best_idx = _init_topk_torch(q_block.size(0), K)

            for p_start in range(0, P, default_product_block):
                p_end = min(p_start + default_product_block, P)

                sims_sum = None  # [Qb, B]
                for _, w, store in field_weight_stores:
                    p_block = store.get_block(
                        p_start, p_end, as_float32=True, use_torch=True
                    )  # [B, D] (GPU)
                    sims = q_block @ p_block.T
                    sims_sum = (
                        sims.mul_(w)
                        if sims_sum is None
                        else sims_sum.add_(sims, alpha=w)
                    )

                kb = min(K * 2, sims_sum.size(1))
                vals, idx_local = torch.topk(
                    sims_sum, k=kb, dim=1, largest=True, sorted=False
                )
                idx_global = idx_local + p_start

                best_idx, best_vals = VectorSearchMethodStoreStatic._merge_topk_torch(
                    best_idx, best_vals, idx_global, vals, K
                )

            best_idx_global[q_start:q_end, :] = best_idx
            best_vals_global[q_start:q_end, :] = best_vals

        return best_idx_global.cpu().numpy(), best_vals_global.cpu().numpy()

    @staticmethod
    def _search_sum_weighted_similarity_streaming_cpu(
        query_vectors: np.ndarray,
        field_weight_stores: List[Tuple[str, float, "EmbeddingStore"]],
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not field_weight_stores:
            raise ValueError("field_weight_stores ist leer")

        Q = int(query_vectors.shape[0])
        P = int(field_weight_stores[0][2].shape[0])
        K = int(min(top_k, P))

        best_vals_global = np.full((Q, K), -np.inf, dtype=np.float32)
        best_idx_global = np.full((Q, K), -1, dtype=np.int64)

        default_query_block = DEFAULT_QUERY_BLOCK
        default_product_block = DEFAULT_PRODUCT_BLOCK
        max_sims = default_query_block * default_product_block

        if Q < default_query_block:
            # Rechne Blockgrößen dynamisch, aber clamp auf P
            new_prod_block = int(max_sims / max(Q, 1))
            default_product_block = min(max(new_prod_block, 1), P)
            default_query_block = Q

        if P < default_product_block:
            new_query_block = int(max_sims / max(P, 1))
            default_query_block = min(max(new_query_block, 1), Q)
            default_product_block = P

        for q_start in range(0, Q, default_query_block):
            q_end = min(q_start + default_query_block, Q)
            q_block = query_vectors[q_start:q_end]  # (Qb, D)

            best_vals = np.full((q_block.shape[0], K), -np.inf, dtype=np.float32)
            best_idx = np.full((q_block.shape[0], K), -1, dtype=np.int64)

            for p_start in range(0, P, default_product_block):
                p_end = min(p_start + default_product_block, P)

                weighted = None  # (Qb, B)
                for _, w, store in field_weight_stores:
                    block = store.get_block(
                        p_start, p_end, as_float32=True, use_torch=False
                    )  # (B, D)
                    sims = q_block @ block.T  # (Qb, B)
                    weighted = (sims * w) if weighted is None else (weighted + w * sims)

                k_block = min(K, weighted.shape[1])
                b_idx_local = np.argpartition(-weighted, k_block - 1, axis=1)[
                    :, :k_block
                ]
                b_val = np.take_along_axis(weighted, b_idx_local, axis=1)
                b_idx = b_idx_local + p_start

                best_idx, best_vals = VectorSearchMethodStoreStatic._merge_topk(
                    best_idx, best_vals, b_idx, b_val, K
                )

            best_idx_global[q_start:q_end] = best_idx
            best_vals_global[q_start:q_end] = best_vals

        return best_idx_global, best_vals_global

    @staticmethod
    def _merge_topk(global_idx, global_val, block_idx, block_val, k: int):
        """
        Merged pro Query die Top-k aus dem bisherigen globalen Puffer mit den Top-k aus dem Block.
        Alle Arrays: shape (Qb, k).
        """
        Qb, K = global_val.shape
        # Kombinieren
        combo_vals = np.concatenate([global_val, block_val], axis=1)  # (Qb, 2k)
        combo_idx = np.concatenate([global_idx, block_idx], axis=1)  # (Qb, 2k)
        # Unsortierte Top-k auswählen
        sel = np.argpartition(-combo_vals, K - 1, axis=1)[:, :K]
        top_vals = np.take_along_axis(combo_vals, sel, axis=1)
        top_idx = np.take_along_axis(combo_idx, sel, axis=1)
        # Innerhalb k sortieren
        order = np.argsort(-top_vals, axis=1)
        top_vals = np.take_along_axis(top_vals, order, axis=1)
        top_idx = np.take_along_axis(top_idx, order, axis=1)
        return top_idx, top_vals

    @staticmethod
    def _merge_topk_torch(
        best_idx: torch.Tensor,
        best_vals: torch.Tensor,
        cand_idx: torch.Tensor,
        cand_vals: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        all_vals = torch.cat([best_vals, cand_vals], dim=1)  # [Q, k + kb]
        all_idx = torch.cat([best_idx, cand_idx], dim=1)  # [Q, k + kb]
        vals, pos = torch.topk(all_vals, k=k, dim=1, largest=True, sorted=True)
        rows = torch.arange(all_vals.size(0), device=all_vals.device).unsqueeze(1)
        idx = all_idx[rows, pos]
        return idx, vals


@dataclass
class VectorSearchMethodStore(VectorSearchMethodStoreStatic):
    products_meta: ProductsMeta
    product_id_to_row: Dict[str, int]
    fields: Dict[str, ProductFieldStore]
    weight_method_2: Optional[Dict[str, float]] = field(init=False)
    weight_method_3: Optional[Dict[str, float]] = field(init=False)
    init_methods: dict = field(default_factory=dict, init=False)
    catalog: bool = True

    @property
    def product_field_store_method_1(self) -> ProductFieldStore:
        if "methode_1" not in self.init_methods:
            raise ValueError(
                "Methode 1 ist nicht initialisiert. Bitte 'init_methode_1' aufrufen."
            )
        return self.fields["product_summary"]

    @property
    def product_field_store_method_2(self) -> ProductFieldStore:
        if "methode_2" not in self.init_methods:
            raise ValueError(
                "Methode 2 ist nicht initialisiert. Bitte 'init_methode_2' aufrufen."
            )
        return self.fields["combined_method_2"]

    @property
    def method_3_field_weight_stores(self) -> List[Tuple[str, float, "EmbeddingStore"]]:
        if "methode_3" not in self.init_methods:
            raise ValueError(
                "Methode 3 ist nicht initialisiert. Bitte 'init_methode_3' aufrufen."
            )
        return list(
            (fname, weight, self.fields[fname].store)
            for fname, weight in self.weight_method_3.items()
        )

    @property
    def product_field_store_methode_4(self) -> ProductFieldStore:
        if "methode_4" not in self.init_methods:
            raise ValueError(
                "Methode 4 ist nicht initialisiert. Bitte 'init_methode_4' aufrufen."
            )
        return self.fields["product_title"]

    def init_methode_1(self):
        product_summary = "product_summary"
        if product_summary not in self.fields:
            raise ValueError(f"Feld '{product_summary}' muss in fields enthalten sein")
        self.fields[product_summary].store.normalize_all()
        self.init_methods["methode_1"] = True

    def init_methode_2(self, weight_method_2: Dict[str, float]):
        field_name = "combined_method_2"
        if field_name in self.fields:
            self.init_methods["methode_2"] = True
        else:
            product_summary = "product_summary"
            if product_summary in weight_method_2:
                raise ValueError(
                    f"Feld '{product_summary}' darf in weight_method_2 nicht gewichtet werden"
                )
            if not set(weight_method_2.keys()).issubset(self.fields.keys()):
                raise ValueError(
                    "Gewichtete Felder in weight_method_2 nicht in fields enthalten"
                )
            if not weight_method_2:
                raise ValueError("weight_method_2 ist leer.")
            self.weight_method_2 = weight_method_2
            fields = self.fields
            pfs = self.get_methode_2_product_field_store_static(fields, weight_method_2)

            self.fields[field_name] = pfs
            self.init_methods["methode_2"] = True

    def reinit_methode_2(self, weight_method_2: Dict[str, float]):
        field_name = "combined_method_2"
        if field_name in self.fields:
            del self.fields[field_name]
        self.init_methods.pop("methode_2", None)
        self.init_methode_2(weight_method_2)

    def init_methode_3(self, weight_method_3: dict[str, float]):
        product_summary = "product_summary"
        if product_summary in weight_method_3:
            raise ValueError(
                f"Feld '{product_summary}' darf in weight_method_3 nicht gewichtet werden"
            )
        if not set(weight_method_3.keys()).issubset(self.fields.keys()):
            raise ValueError(
                "Gewichtete Felder in weight_method_3 nicht in fields enthalten"
            )
        if not weight_method_3:
            raise ValueError("weight_method_3 ist leer.")
        for fname in weight_method_3.keys():
            self.fields[fname].store.normalize_all()
        self.weight_method_3 = weight_method_3
        self.init_methods["methode_3"] = True

    def reinit_methode_3(self, weight_method_3: Dict[str, float]):
        self.init_methods.pop("methode_3", None)
        self.init_methode_3(weight_method_3)

    def init_methode_4(self):
        product_title = "product_title"
        if product_title not in self.fields:
            raise ValueError(f"Feld '{product_title}' muss in fields enthalten sein")
        self.fields[product_title].store.normalize_all()
        self.init_methods["methode_4"] = True

    @staticmethod
    def load(base_dir: pathlib.Path) -> "VectorSearchMethodStore":
        fields, pid_to_row, products = VectorSearchMethodStoreStatic.vars_for_vsms(
            base_dir
        )
        return VectorSearchMethodStore(
            products_meta=products, product_id_to_row=pid_to_row, fields=fields
        )

    @staticmethod
    def load_and_init_all(
        base_dir: pathlib.Path, catalog: bool
    ) -> "VectorSearchMethodStore":
        vsm_store = VectorSearchMethodStore.load(base_dir)
        vsm_store.catalog = catalog
        w2 = VectorSearchMethodStoreStatic._load_weights_from_json(
            base_dir, "2", catalog=catalog
        )
        w3 = VectorSearchMethodStoreStatic._load_weights_from_json(
            base_dir, "3", catalog=catalog
        )
        vsm_store.init_methode_1()
        vsm_store.init_methode_2(w2)
        vsm_store.init_methode_3(w3)
        vsm_store.init_methode_4()
        return vsm_store

    def get_vector_store_product_store_for_canditas(
        self, cand_rows: List[int]
    ) -> "VectorSearchMethodStore":
        """
        Erstellt einen neuen VectorSearchMethodStore nur für die gegebenen Produkt-Zeilen.
        """

        products_meta = {**asdict(self.products_meta)}

        ids_df_cand = self.products_meta.ids_df.iloc[cand_rows].reset_index(drop=True)
        products_meta["ids_df"] = ids_df_cand
        products_meta = ProductsMeta(**products_meta)

        new_fields: Dict[str, ProductFieldStore] = {}
        for fname, pfs in self.fields.items():
            new_store = pfs.store.get_substore_by_rows(cand_rows)
            new_fields[fname] = ProductFieldStore(name=fname, store=new_store)
        product_id_to_row = {
            str(pid): int(i)
            for i, pid in enumerate(
                products_meta.ids_df[products_meta.id_col].astype(str).values
            )
        }
        vsms = VectorSearchMethodStore(
            products_meta=products_meta,
            product_id_to_row=product_id_to_row,
            fields=new_fields,
        )
        for methode in self.init_methods.keys():
            match methode:
                case "methode_1":
                    vsms.init_methode_1()
                case "methode_2":
                    vsms.init_methode_2(self.weight_method_2)  # type: ignore
                case "methode_3":
                    vsms.init_methode_3(self.weight_method_3)  # type: ignore
                case "methode_4":
                    vsms.init_methode_4()
        return vsms

    # ----------------------- Methoden -----------------------
    # Method 1: summary only
    def search_product_summary(
        self, query_vectors: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.init_methods.get("methode_1", False):
            raise ValueError(
                "Methode 1 ist nicht initialisiert. Bitte 'init_methode_1' aufrufen."
            )
        store = self.product_field_store_method_1.store
        return self.search_summary_streaming(query_vectors, store, top_k)

    # Method 2: weighted add embeddings (renormiert)
    def search_weighted_add_embeddings(
        self, query_vectors: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.init_methods.get("methode_2", False):
            raise ValueError(
                "Methode 2 ist nicht initialisiert. Bitte 'init_methode_2' aufrufen."
            )
        store = self.product_field_store_method_2.store
        return self.search_summary_streaming(query_vectors, store, top_k)

    # Method 3: weighted sum of per-field scores
    def search_sum_weighted_similarity(
        self,
        query_vectors: np.ndarray,
        top_k: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        - Holt je Feld Top‑K_each Kandidaten für ALLE Queries in EINEM Durchlauf (blockweise über Produkte).
        - Aggregiert pro Query die gewichteten Scores gleicher Produkt-IDs (Summe über Felder).
        - Liefert finale Top‑K Indices & Scores.
        """
        if not self.init_methods.get("methode_3", False):
            raise ValueError(
                "Methode 3 ist nicht initialisiert. Bitte 'init_methode_3' aufrufen."
            )
        field_weight_stores = self.method_3_field_weight_stores
        return self.search_sum_weighted_similarity_streaming(
            query_vectors, field_weight_stores, top_k
        )

    # Method 4: title embeddings only
    def search_title_embeddings(
        self, query_vectors: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:

        field_name = "product_title"
        if self.init_methods.get("methode_4", False) is not True:
            raise ValueError(
                "Methode 4 ist nicht initialisiert. Bitte 'init_methode_4' aufrufen."
            )
        store = self.fields[field_name].store
        return self.search_summary_streaming(query_vectors, store, top_k)


@dataclass
class Evaluation:
    vector_search_method_store: VectorSearchMethodStore
    query_store: QueryStore

    def get_all_query_metrics_for_query_ids_candidates(
        self, query_ids: List[str], top_k: int, methodes: Optional[List[str]] = None
    ):
        return Evaluation.get_all_query_metrics_for_query_ids_candidates_static(
            self.vector_search_method_store,
            self.query_store,
            query_ids,
            top_k,
            methodes,
        )

    @staticmethod
    def get_all_query_metrics_for_query_ids_candidates_static(
        vector_search_method_store: VectorSearchMethodStore,
        query_store: QueryStore,
        query_ids: List[str],
        top_k: int,
        methodes: Optional[List[str]] = None,
    ):
        pid2row = vector_search_method_store.product_id_to_row
        all_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}
        for qid in query_ids:
            pid_to_gain = query_store.get_pids_esci_gain_for_id(qid)
            cand_rows = [pid2row[pid] for pid in pid_to_gain.keys() if pid in pid2row]
            vsm_store = (
                vector_search_method_store.get_vector_store_product_store_for_canditas(
                    cand_rows
                )
            )
            eval_instance = Evaluation(vsm_store, query_store)
            metrics = eval_instance.get_all_query_metrics_for_query_ids(
                [qid], top_k, eval_on="catalog", methodes=methodes
            )  # Dict[methode, Dict[query_id, Dict[metric_name, value]]]
            for m, mstore in metrics.items():
                all_metrics.setdefault(m, {}).update(mstore)
        return all_metrics  # Dict[methode, Dict[query_id, Dict[metric_name

    @staticmethod
    def get_describe_metrics_generally(
        query_ids: List[str],
        query_store: QueryStore,
        store_products: ProductsMeta,
        top_idx: np.ndarray,
        top_sims: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """
        Baut Metriken über Queries und gibt deren .describe() als Dict zurück.
        Ungültige Treffer (Index < 0 oder nicht-finite Scores) werden ignoriert.
        """
        metrics = Evaluation.get_metrics_generally_raw(
            query_ids, query_store, store_products, top_idx, top_sims
        )
        return (
            pd.DataFrame.from_dict(metrics, orient="index")
            .describe()
            .to_dict(orient="index")
        )

    @staticmethod
    def get_metrics_generally_raw(
        query_ids: list[str],
        query_store: QueryStore,
        store_products: ProductsMeta,
        top_idx: np.ndarray,
        top_sims: np.ndarray,
    ) -> dict[str, dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        min_gain: float = min(ESCI_TO_GAIN_DEFAULT.values())
        for q_idx, qid in enumerate(query_ids):
            pid_to_gain = query_store.get_pids_esci_gain_for_id(qid)
            gains_from_search: List[float] = []
            for prod_row, sim in zip(top_idx[q_idx, :], top_sims[q_idx, :]):
                if int(prod_row) < 0 or not np.isfinite(sim):
                    continue
                pid = str(
                    store_products.ids_df.iloc[int(prod_row)][store_products.id_col]
                )
                gains_from_search.append(float(pid_to_gain.get(pid, min_gain)))

            metrics[qid] = get_all_metrics(
                gains_from_search,
                list(pid_to_gain.values()),
                ks=[10, 15, 30, 45, 60, 75],
            )
        return metrics  # Dict[query_id, Dict[metric_name, value]]

    def get_metrics_for_seed_sample(
        self,
        candidates,
        n: int = 100,
        top_k: int = 75,
        methode: str = "1",
        random_state: int = 42,
    ) -> pd.DataFrame:
        q_vectors, q_ids = self.query_store.get_query_sample(
            n, random_state=random_state
        )
        metrics = self.get_all_query_metrics_for_query_ids(
            q_ids, top_k, eval_on=candidates, methodes=[methode]
        )
        return pd.DataFrame(metrics.get(f"methode_{methode}", {})).T.describe()

    def get_all_query_metrics_for_query_ids(
        self, query_ids, top_k: int, eval_on: str, methodes: Optional[List[str]] = None
    ):
        if eval_on == "candidates":
            return self.get_all_query_metrics_for_query_ids_candidates(
                query_ids, top_k, methodes
            )
        else:
            methodes = methodes or AVAILABLE_METHODS[:]
            # check if methodes are valid
            for methode in methodes:
                if methode not in AVAILABLE_METHODS:
                    raise ValueError(
                        f"Unbekannte Methode '{methode}', nur {AVAILABLE_METHODS} erlaubt"
                    )
            query_store = self.query_store
            q_vectors = query_store.get_vectors_for_ids(query_ids, as_float32=True)
            store_products = self.vector_search_method_store.products_meta
            all_metrics = {}
            for methode in methodes:
                match methode:
                    case "1":
                        methoden_call = (
                            self.vector_search_method_store.search_product_summary
                        )
                    case "2":
                        methoden_call = (
                            self.vector_search_method_store.search_weighted_add_embeddings
                        )
                    case "3":
                        methoden_call = (
                            self.vector_search_method_store.search_sum_weighted_similarity
                        )
                    case "4":
                        methoden_call = (
                            self.vector_search_method_store.search_title_embeddings
                        )
                    case _:
                        raise ValueError(
                            f"Unbekannte Methode '{methode}', nur {AVAILABLE_METHODS} erlaubt"
                        )
                top_idx, top_sims = methoden_call(query_vectors=q_vectors, top_k=top_k)
                metrics = Evaluation.get_metrics_generally_raw(
                    query_ids, query_store, store_products, top_idx, top_sims
                )  # Dict[query_id, Dict[metric_name, value]]
                all_metrics[f"methode_{methode}"] = metrics
            return all_metrics

    def save_all_query_metrics_to_csv(self):
        """
        Speichert für alle Queries je Methode eine CSV (Queries=Zeilen, Metriken=Spalten).
        Pfadschema:
          results/all_query_metrics/{catalog|candidates}/{MODEL}/
            1.csv, 2.csv, 3.csv, 4.csv
        """
        all_query_ids = self.query_store.id_to_row.keys()
        model_name = self.vector_search_method_store.products_meta.meta_dir.name
        on = "catalog" if self.vector_search_method_store.catalog else "candidates"

        all_metrics = self.get_all_query_metrics_for_query_ids(
            all_query_ids, top_k=75, eval_on=on, methodes=AVAILABLE_METHODS
        )

        results_dir = pathlib.Path("results") / "all_query_metrics" / model_name / on
        results_dir.mkdir(parents=True, exist_ok=True)
        self.data_to_csv_gz(all_metrics, results_dir)

        print(f"All query metrics saved to CSVs under {results_dir}")

    @staticmethod
    def data_to_csv_gz(
        data: dict[str, dict[str, dict[str, float]]], out_dir: pathlib.Path
    ):
        for (
            method_key,
            method_metric,
        ) in data.items():  # Dict[query_id -> Dict[metric -> val]]
            methode = method_key.replace("methode_", "")
            df = pd.DataFrame(method_metric)
            df.sort_index(inplace=True)
            df = df.round(6)
            dest = out_dir / f"{methode}.csv.gz"
            df.T.to_csv(dest, index=True, compression="gzip")


if __name__ == "__main__" and True:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str)
    ap.add_argument(
        "--run_test",
        type=bool,
        default=False,
    )

    args = ap.parse_args()
    TEST = args.run_test
    base_dirs = get_base_dirs(args.model_name)
    for base_dir in base_dirs:
        print(f"Evaluating model in {base_dir}")
        start = time.time()
        for ca in ["catalog", "candidates"]:
            vsm_store = VectorSearchMethodStore.load_and_init_all(
                base_dir, catalog=(ca == "catalog")
            )
            query_store = QueryStore.load(base_dir)
            eval_instance = Evaluation(vsm_store, query_store)
            end = time.time()
            print(f"Loading time: {end - start:.2f} seconds")
            if TEST:
                # Beispiel: Methode 3 mit 100 Zufalls-Queries
                n = 100
                metrics_df = eval_instance.get_metrics_for_seed_sample(
                    candidates=ca, n=n, methode="3"
                )
                ndcg_at = 10
                desc = metrics_df[f"NDCG@{ndcg_at}"]
                print(f"NDCG@{ndcg_at} über {n} Queries:\n{desc}")

            else:
                eval_instance.save_all_query_metrics_to_csv()

"""
python method_comparison.py --model_name jinaai_jina-embeddings-v2-small-en_cuda_76ac2760
"""
"""
python method_comparison.py --model_name Qwen_Qwen3-Embedding-0.6B_cuda_76ac2760
"""
