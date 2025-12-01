from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class DataReducer:
    """
    Reduziert den Amazon ESCI-Datensatz (oder kompatible Schemata) entlang
    reproduzierbarer Filterkriterien und protokolliert die Zwischenergebnisse.

    Schritte (vgl. Tabelle in der Arbeit):
    1) Auswahl der Sprache/Region (hier: product_locale == "us")
    2) Entfernen unvollständiger Produktdaten (leere/NaN Attribute)
    3) Entfernen von Queries ohne gültige Produkte
    4) Mindestanzahl an Produkten pro Query (>= min_products_per_query)
    5) Auswahl relevanter Spalten & Reset Index
    6) Sortierung nach Query- und Beschreibungs-Länge (absteigend)
    7) long -> wide Aggregation je Query (product_id & esci_label als
    8) (Optional) Persistenz als Parquet

    Hinweise:
    - Spaltennamen sind parametrisierbar und standardmäßig ESCI-kompatibel.
    - Statt print() wird strukturiert geloggt.
    - Zusammenfassungen werden als Strings zurückgegeben (geeignet für Paper-Statistiken).
    """

    # Eingabe
    df_examples: pd.DataFrame
    df_products: pd.DataFrame

    # Filter/Parameter
    locale_value: str = "us"
    min_products_per_query: int = 10

    # Spaltenkonfiguration (ESCI-kompatible Defaults)
    col_product_locale: str = "product_locale"
    col_query_id: str = "query_id"
    col_product_id: str = "product_id"
    col_query_text: str = "query"
    col_title: str = "product_title"
    col_description: str = "product_description"
    col_bullets: str = "product_bullet_point"
    col_brand: str = "product_brand"

    # Relevante Produktattribute, die nicht leer sein dürfen
    required_product_attrs: Sequence[str] = field(
        default_factory=lambda: [
            "product_description",
            "product_bullet_point",
            "product_brand",
        ]
    )

    # Spalten, die im Output verbleiben sollen
    keep_example_cols: Sequence[str] = field(
        default_factory=lambda: ["query_id", "product_id", "query", "esci_label"]
    )
    keep_product_cols: Sequence[str] = field(
        default_factory=lambda: [
            "product_id",
            "product_title",
            "product_description",
            "product_bullet_point",
            "product_brand",
        ]
    )

    # Spalten, die zum Schluss (falls vorhanden) entfernt werden
    drop_example_cols_final: Sequence[str] = field(
        default_factory=lambda: ["product_locale", "small_version", "large_version"]
    )
    drop_product_cols_final: Sequence[str] = field(
        default_factory=lambda: ["product_locale"]
    )

    # Logging
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__), init=False
    )
    log_level: int = logging.INFO

    def __post_init__(self) -> None:
        if not self.logger.handlers:
            logging.basicConfig(
                level=self.log_level,
                format="%(asctime)s - %(levelname)s - %(message)s",
            )
        self._validate_columns()

    # -------------------------- public API --------------------------

    def reduce(self, save_dir: Optional[Path] = None) -> Tuple[
        Tuple[pd.DataFrame, Optional[Path]],
        Tuple[pd.DataFrame, Optional[Path]],
        List[str],
    ]:
        """
        Führt alle Reduktionsschritte aus und gibt (df_examples_reduced, path),
        (df_products_reduced, path) und eine textuelle Zusammenfassung zurück.
        """
        start_examples = self.df_examples.copy()
        start_products = self.df_products.copy()

        # 1) Locale-Filter
        ex, pr = self._filter_locale(start_examples, start_products)
        self._log_counts("Nach Locale-Filter", ex, pr)

        # 2) Produkte mit leeren Attributen entfernen & Beispiele synchronisieren
        ex, pr = self._drop_incomplete_products(ex, pr)
        self._log_counts("Nach Entfernen leerer Produktattribute", ex, pr)

        # 3) Minimum Produkte pro Query
        ex = self._enforce_min_products_per_query(ex)
        self._log_counts("Nach Mindestanzahl Produkte/Query", ex, pr)

        # 4) Relevante Spalten reduzieren
        ex, pr = self._select_relevant_columns(ex, pr)
        self._log_counts("Nach Auswahl relevanter Spalten", ex, pr)

        # 5) sortiere nach der zeichen länge der query, bzw. der beschreibung, absteigend

        ex = (
            ex.assign(__query_len=ex[self.col_query_text].astype(str).str.len())
            .sort_values(by="__query_len", ascending=False)
            .drop(columns="__query_len")
        )

        pr = (
            pr.assign(__desc_len=pr[self.col_description].astype(str).str.len())
            .sort_values(by="__desc_len", ascending=False)
            .drop(columns="__desc_len")
        )
        # 6) long -> wide Aggregation je Query (NEU)
        #    - query: first
        #    - product_id: list
        #    - esci_label: list
        agg_dict = {
            self.col_query_text: "first",
            self.col_product_id: list,
            "esci_label": list,
        }

        ex = ex.groupby(self.col_query_id, as_index=True).agg(agg_dict)

        # 7) Index resetten
        ex, pr = ex.reset_index(drop=False), pr.reset_index(drop=True)
        self._log_counts("Nach Reset Index", ex, pr)

        # 8) Optionale Persistenz
        ex_path, pr_path = (None, None)
        if save_dir is not None:
            ex_path, pr_path = self._save_parquet(ex, pr, save_dir)
            self.logger.info("Reduzierte Daten gespeichert: %s | %s", ex_path, pr_path)

        summary = self._build_summary(start_examples, start_products, ex, pr)
        return (ex, ex_path), (pr, pr_path), summary

    # ----------------------- internal helpers -----------------------

    def _validate_columns(self) -> None:
        """Stellt sicher, dass alle benötigten Spalten existieren."""
        need_examples = {
            self.col_query_id,
            self.col_product_id,
            self.col_query_text,
            self.col_product_locale,
        }
        need_products = {
            self.col_product_id,
            self.col_title,
            self.col_description,
            self.col_bullets,
            self.col_brand,
            self.col_product_locale,
        }
        missing_ex = need_examples - set(self.df_examples.columns)
        missing_pr = need_products - set(self.df_products.columns)

        if missing_ex or missing_pr:
            raise ValueError(
                f"Fehlende Spalten — examples: {sorted(missing_ex)} | products: {sorted(missing_pr)}"
            )

        # Map required_product_attrs auf tatsächliche Spaltennamen
        # (Falls Nutzer abweichende Namen in required_product_attrs setzt.)
        for col in self.required_product_attrs:
            if col not in self.df_products.columns:
                raise ValueError(
                    f"required_product_attrs enthält '{col}', aber diese Spalte existiert nicht."
                )

    def _filter_locale(
        self, ex: pd.DataFrame, pr: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ex = ex[ex[self.col_product_locale] == self.locale_value]
        pr = pr[pr[self.col_product_locale] == self.locale_value]
        return ex, pr

    def _drop_incomplete_products(
        self, ex: pd.DataFrame, pr: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Entfernt NaN, leere Strings und Whitespace-only
        mask = pd.Series(True, index=pr.index)
        for col in self.required_product_attrs:
            col_mask = pr[col].notna() & (pr[col].astype(str).str.strip() != "")
            mask &= col_mask
        pr = pr.loc[mask]
        if False:
            # for testing reduce pr
            ex_temp = ex.groupby(self.col_query_id).agg(
                {
                    self.col_query_text: "first",
                    self.col_product_id: list,
                    "esci_label": list,
                }
            )
            ex_temp = ex_temp.sample(n=2000, random_state=42)
            p_ids = ex_temp[self.col_product_id].explode().unique()
            pr = pr[pr[self.col_product_id].isin(p_ids)]

        valid_ids = set(pr[self.col_product_id].unique())
        ex = ex[ex[self.col_product_id].isin(valid_ids)]

        return ex, pr

    def _enforce_min_products_per_query(self, ex: pd.DataFrame) -> pd.DataFrame:
        counts = ex.groupby(self.col_query_id)[self.col_product_id].size()
        valid_queries = counts[counts >= self.min_products_per_query].index
        return ex[ex[self.col_query_id].isin(valid_queries)]

    def _select_relevant_columns(
        self, ex: pd.DataFrame, pr: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # Spalten ggf. vorhanden -> droppen
        ex = ex.drop(
            columns=[c for c in self.drop_example_cols_final if c in ex.columns]
        )
        pr = pr.drop(
            columns=[c for c in self.drop_product_cols_final if c in pr.columns]
        )

        # Relevante Spalten schneiden (nur behalten, die existieren)
        ex_cols = [c for c in self.keep_example_cols if c in ex.columns]
        pr_cols = [c for c in self.keep_product_cols if c in pr.columns]
        return ex[ex_cols], pr[pr_cols]

    def _save_parquet(
        self, ex: pd.DataFrame, pr: pd.DataFrame, out_dir: Path
    ) -> Tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        ex_path = out_dir / "shopping_queries_dataset_examples_reduced.parquet"
        pr_path = out_dir / "shopping_queries_dataset_products_reduced.parquet"
        ex.to_parquet(ex_path, index=False)  # pyarrow/fastparquet kompatibel
        pr.to_parquet(pr_path, index=False)
        return ex_path, pr_path

    @staticmethod
    def _is_listy_series(s: pd.Series) -> bool:
        """Heuristik: enthält die Series listen-/sequenzartige Einträge?"""
        from numpy import dtype

        if s.dtype != "object":
            return False
        return s.apply(lambda x: isinstance(x, (list, tuple, np.ndarray))).any()

    def _pair_count(self, ex: pd.DataFrame) -> int:
        """Robuste Paarzählung: long (eine Zeile = ein Paar) ODER wide (listen in product_id)."""
        if self.col_product_id not in ex.columns:
            return len(ex)  # Fallback
        col = ex[self.col_product_id]
        if self._is_listy_series(col):
            return int(
                col.apply(lambda v: len(v) if isinstance(v, Sequence) else 0).sum()
            )
        # long-Format
        return int(len(ex))

    def _log_counts(self, stage: str, ex: pd.DataFrame, pr: pd.DataFrame) -> None:
        self.logger.info(
            "%s — Queries: %s | Produkte: %s | Paare: %s",
            stage,
            ex[self.col_query_id].nunique(),
            len(pr),
            self._pair_count(ex),
        )

    def _build_summary(
        self,
        ex_start: pd.DataFrame,
        pr_start: pd.DataFrame,
        ex_end: pd.DataFrame,
        pr_end: pd.DataFrame,
    ) -> List[str]:
        def pct(part: float, whole: float) -> float:
            return 0.0 if whole == 0 else 100.0 * part / whole

        lines = []
        lines.append(
            f"US-Produkte: {len(pr_end):,} von {len(pr_start):,} "
            f"({pct(len(pr_end), len(pr_start)):.2f}%)"
        )
        lines.append(
            f"US-Queries (unique): {ex_end[self.col_query_id].nunique():,} von "
            f"{ex_start[self.col_query_id].nunique():,} "
            f"({pct(ex_end[self.col_query_id].nunique(), ex_start[self.col_query_id].nunique()):.2f}%)"
        )
        lines.append(
            f"Query-Produkt-Paare: {len(ex_end):,} von {len(ex_start):,} "
            f"({pct(len(ex_end), len(ex_start)):.2f}%)"
        )
        return lines

    # --------------------- dataset description API ---------------------

    def _to_long_if_wide(self, ex: pd.DataFrame) -> pd.DataFrame:
        """Erzeuge long-Format, falls product_id (und optional esci_label) Listen enthalten."""
        if self.col_product_id not in ex.columns:
            return ex
        is_listy_pid = self._is_listy_series(ex[self.col_product_id])
        is_listy_lbl = "esci_label" in ex.columns and self._is_listy_series(
            ex["esci_label"]
        )

        if is_listy_pid and is_listy_lbl:
            return ex.explode([self.col_product_id, "esci_label"], ignore_index=True)
        if is_listy_pid:
            return ex.explode(self.col_product_id, ignore_index=True)
        return ex

    def describe_dataset(
        self,
        ex_reduced: pd.DataFrame,
        pr_reduced: pd.DataFrame,
        top_k: int = 5,
        locale: str = "de_DE",
        pretty: bool = True,
    ) -> Mapping[str, object]:
        """
        Liefert eine strukturierte Beschreibung des (reduzierten) Datensatzes.
        Nutzt self.df_examples/self.df_products als Referenz (vor der Reduktion).

        Returns:
            Dict mit Kennzahlen, Verteilungen und Top-Listen.
            Bei pretty=True zusätzlich 'pretty_lines' (List[str]) für direkte Ausgabe.
        """
        # --- helpers ---
        fmt_int = self._make_int_formatter(locale)

        def pct(part: float, whole: float) -> float:
            return 0.0 if whole == 0 else 100.0 * part / whole

        # --- basic counts / coverage ---
        # Bezugswerte: ursprüngliche (unreduzierte) DataFrames
        ex_full, pr_full = self.df_examples, self.df_products

        # WICHTIG: für Zählungen/Verteilungen stets long verwenden
        ex_long = self._to_long_if_wide(ex_reduced)

        n_queries_full = ex_full[self.col_query_id].nunique()
        n_queries_red = ex_reduced[self.col_query_id].nunique()

        n_prod_full_total = pr_full[self.col_product_id].nunique()
        n_prod_red_total = pr_reduced[self.col_product_id].nunique()
        n_prod_red_in_examples = ex_long[self.col_product_id].nunique()

        coverage_products = pct(n_prod_red_in_examples, n_prod_full_total)

        # Label-Verteilung
        label_col = "esci_label"
        label_dist = None
        if label_col in ex_long.columns:
            label_dist = (
                ex_long[label_col].value_counts(normalize=True) * 100
            ).sort_index()

        # Produkte/Query-Stats (long)
        ppq_series = (
            ex_long.groupby(self.col_query_id)[self.col_product_id]
            .nunique()
            .astype("int64")
        )
        ppq_stats = ppq_series.describe()

        # Top-Produkte (long)
        counts_per_product = (
            ex_long.groupby(self.col_product_id)[self.col_query_id]
            .nunique()
            .sort_values(ascending=False)
        )
        top_products = counts_per_product.head(top_k)

        # map product_id -> truncated title (falls vorhanden)
        title_map = None
        if self.col_title in pr_reduced.columns:
            title_map = pr_reduced.set_index(self.col_product_id)[
                self.col_title
            ].astype(str)

        # Baue ein anzeigbares Index-Label
        def _display_name(pid: str) -> str:
            if title_map is None or pid not in title_map:
                return str(pid)
            title = title_map.loc[pid]
            return (title[:60] + "…") if len(title) > 60 else title

        top_products_named = top_products.rename(index=_display_name)

        # --- top-k relevant / irrelevant (falls Labels verfügbar) ---
        top_rel, top_irrel = None, None
        if label_col in ex_long.columns:

            def _top_for(label: str) -> pd.Series:
                s = (
                    ex_long[ex_long[label_col] == label]
                    .groupby(self.col_product_id)[self.col_query_id]
                    .nunique()
                    .sort_values(ascending=False)
                    .head(top_k)
                )
                return s.rename(index=_display_name)

            # ESCI-Konvention: E=exact, S=substitute, C=complement, I=irrelevant
            top_rel = _top_for("E")
            top_irrel = _top_for("I")

        # --- result dict ---
        result: dict = {
            "counts": {
                "queries_full_unique": int(n_queries_full),
                "queries_reduced_unique": int(n_queries_red),
                "products_full_unique": int(n_prod_full_total),
                "products_reduced_unique": int(n_prod_red_total),
                "products_reduced_in_examples_unique": int(n_prod_red_in_examples),
                "pairs_reduced": int(len(ex_long)),
                "coverage_products_percent": float(coverage_products),
            },
            "label_distribution_percent": (
                label_dist.round(2) if label_dist is not None else None
            ),
            "products_per_query_stats": ppq_stats.round(2),
            "top_products_by_query_count": top_products_named,
            "top_relevant_products_E": top_rel,
            "top_irrelevant_products_I": top_irrel,
            "esci_label_explanation_de": (
                "E: exact – exaktes Match\n"
                "S: substitute – geeigneter Ersatz\n"
                "C: complement – sinnvolle Ergänzung\n"
                "I: irrelevant – kein Bezug zur Suchanfrage"
            ),
        }

        if pretty:
            lines = [
                f"Anzahl Queries (reduziert): {fmt_int(n_queries_red)} von {fmt_int(n_queries_full)} "
                f"= {pct(n_queries_red, n_queries_full):.2f}%",
                f"Anzahl Produkte (reduziert, total): {fmt_int(n_prod_red_total)} von {fmt_int(n_prod_full_total)} "
                f"= {pct(n_prod_red_total, n_prod_full_total):.2f}%",
                f"Produkte im reduzierten Beispiel-Datensatz (unique): {fmt_int(n_prod_red_in_examples)} "
                f"→ Produktabdeckung: {coverage_products:.2f}%",
                f"Query-Produkt-Paare (reduziert): {fmt_int(len(ex_long))}",
            ]

            if label_dist is not None:
                lines.append("\nVerteilung ESCI-Labels (in %):")
                for k, v in label_dist.sort_index().round(2).items():
                    lines.append(f"  {k}: {v:.2f}%")
                lines.append("\n" + result["esci_label_explanation_de"])

            lines.append("\nStatistische Kennzahlen: Produkte pro Query")
            for stat_name, val in ppq_stats.round(2).items():
                lines.append(f"  {stat_name}: {val}")

            lines.append(f"\nTop {top_k} Produkte (vorkommende Queries):")
            for name, cnt in top_products_named.items():
                lines.append(f"  {name}: {cnt}")

            if top_rel is not None:
                lines.append(f"\nTop {top_k} Produkte (meist als E=exact markiert):")
                for name, cnt in top_rel.items():
                    lines.append(f"  {name}: {cnt}")

            if top_irrel is not None:
                lines.append(
                    f"\nTop {top_k} Produkte (meist als I=irrelevant markiert):"
                )
                for name, cnt in top_irrel.items():
                    lines.append(f"  {name}: {cnt}")

            result["pretty_lines"] = lines

        return result

    # --------------------- small utility for formatting ---------------------

    def _make_int_formatter(self, locale: str = "de_DE"):
        """
        Erzeugt eine Integer-Formatfunktion mit deutscher Gruppierung (ohne harte Abhängigkeit von 'babel').
        Falls 'babel' vorhanden ist, wird es verwendet; ansonsten Fallback über format().
        """
        try:
            from babel.numbers import format_decimal as _fmt  # type: ignore

            def _f(x: int) -> str:
                return _fmt(int(x), locale=locale).replace(".", " ")

            return _f
        except Exception:
            # Fallback: 1 234 567
            def _f(x: int) -> str:
                s = f"{int(x):,}".replace(",", " ")
                return s

            return _f


if __name__ == "__main__":
    data_set_dir = Path("shopping_queries_dataset")

    # Load the dataset
    df_examples = pd.read_parquet(
        data_set_dir / "shopping_queries_dataset_examples.parquet"
    )
    df_products = pd.read_parquet(
        data_set_dir / "shopping_queries_dataset_products.parquet"
    )
    df_sources = pd.read_csv(data_set_dir / "shopping_queries_dataset_sources.csv")

    print(f"Beispiel-Datensatz: {len(df_examples)} Zeilen")
    print(f"Produkt-Datensatz: {len(df_products)} Zeilen")
    print(f"Quellen-Datensatz: {len(df_sources)} Zeilen")

    # Reduce the dataset
    reducer = DataReducer(df_examples=df_examples, df_products=df_products)
    (df_ex_reduced, ex_path), (df_pr_reduced, pr_path), summary = reducer.reduce(
        save_dir=data_set_dir / "reduced"
    )
    print("\n".join(summary))
    print()
    desc = reducer.describe_dataset(df_ex_reduced, df_pr_reduced, pretty=True)

    # Console-Ausgabe (schön formatiert):
    print("\n--- Erweiterte Beschreibung des reduzierten Datensatzes ---\n")
    print("\n".join(desc["pretty_lines"]))

    print(f"Path zu reduzierten Beispielen: {ex_path}")
    print(f"Path zu reduzierten Produkten: {pr_path}")
