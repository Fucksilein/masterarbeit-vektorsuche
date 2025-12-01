import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances


@dataclass
class Config:
    # input CSV must have columns "word" and "category"
    csv_file: Path = Path("data/example_cat_words.csv")
    output_dir: Path = Path("../src/code_output/")
    embedding_model: str = "BAAI/bge-m3"
    random_seed: Optional[int] = 25
    pca_components: int = 2
    round_decimals: int = 2

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)


class EmbeddingVisualizer:
    """Kapselt die Logik zur Erstellung von Embeddings und Dimensionsreduktion."""

    def __init__(self, config: Config):
        self.cfg = config
        self.model = SentenceTransformer(
            self.cfg.embedding_model, trust_remote_code=True
        )
        self.reducer = PCA(
            n_components=self.cfg.pca_components,
            random_state=self.cfg.random_seed,
        )

    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.cfg.csv_file)

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)

    def reduce_dimensions(self, embeddings: np.ndarray) -> np.ndarray:
        return self.reducer.fit_transform(embeddings)

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        words = df_copy["word"].tolist()
        embeddings = self.generate_embeddings(words)
        coords = self.reduce_dimensions(embeddings)
        df_copy["x"] = coords[:, 0].round(self.cfg.round_decimals)
        df_copy["y"] = coords[:, 1].round(self.cfg.round_decimals)
        return df_copy


def create_matplotlib_plot(df: pd.DataFrame, category_colors: dict) -> plt.Figure:
    """Erstellt den Matplotlib Scatter-Plot mit angepassten Text-Labels."""
    colors = df["category"].map(category_colors)
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(df["x"], df["y"], c=colors.to_list(), s=80, ec="black", lw=0.5)

    texts = [
        ax.text(row["x"], row["y"], row["word"], fontsize=12)
        for _, row in df.iterrows()
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    # don't need title, make \caption work better in tex
    # ax.set_title("PCA Projektion von Wort Einbettungen", fontsize=16)
    ax.set_xlabel("X-Koordinate")
    ax.set_ylabel("Y-Koordinate")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=cat,
            markerfacecolor=color,
            markersize=10,
        )
        for cat, color in category_colors.items()
    ]
    ax.legend(handles=legend_elements, title="Kategorie")
    plt.tight_layout()
    return fig


def export_embedding_table(
    df: pd.DataFrame, category_colors: dict, out_path: Path, decimals: int
):
    """Exportiert die Koordinaten-Tabelle nach LaTeX."""
    # KORRIGIERT: Die Funktion hängt nicht mehr von einem `fig`-Objekt ab.
    df_style = (
        df[["word", "category", "x", "y"]]
        .copy()
        .style.format({"x": f"{{:.{decimals}f}}", "y": f"{{:.{decimals}f}}"})
    )

    # KORRIGIERT: Färbt den Text der Wörter basierend auf ihrer Kategorie.
    df["color"] = df["category"].map(category_colors)
    word_to_color = pd.Series(df.color.values, index=df.word).to_dict()

    def color_word(word_val):
        color = word_to_color.get(word_val, "black")
        return f"color: {color}"

    df_style = df_style.map(color_word, subset=["word"])

    latex_str = df_style.to_latex(
        position="!htbp",
        label="tab:embedding_table",
        caption="Koordinaten der Wörter nach PCA-Reduktion",
        hrules=True,
        convert_css=True,
    )
    out_path.write_text(latex_str)
    print(f"→ Embedding-Tabelle gespeichert: {out_path}")


def export_distance_table(df: pd.DataFrame, category_colors: dict, out_path: Path):
    """Erstellt eine Cosinus-Distanzmatrix und exportiert sie nach LaTeX."""
    # KORRIGIERT: Die Funktion wurde von der `fig`-Abhängigkeit befreit.
    embeddings = np.array(df[["x", "y"]])
    distances = cosine_distances(embeddings)
    distance_df = pd.DataFrame(distances, columns=df["word"], index=df["word"])

    styler = distance_df.style.format("{:.2f}")

    # KORRIGIERT: Mapping von Wort zu Farbe direkt aus den Daten erstellen.
    df["color"] = df["category"].map(category_colors)
    word_to_color = pd.Series(df.color.values, index=df.word).to_dict()

    # Färbt die Zelle, wenn Zeile und Spalte zur selben Kategorie gehören.
    def color_cell_by_category(val, row_name, col_name):
        row_category = df.loc[df["word"] == row_name, "category"].values[0]
        col_category = df.loc[df["word"] == col_name, "category"].values[0]
        return (
            f"background-color: {word_to_color[row_name]}"
            if row_category == col_category
            else ""
        )

    styler = styler.apply(
        lambda col: [
            color_cell_by_category(val, col.name, row_idx)
            for row_idx, val in col.items()
        ],
        axis=0,
    )

    # Wendet einen Textfarbverlauf auf die Distanzwerte an.
    cmap = LinearSegmentedColormap.from_list("custom_gradient", ["#000000", "#FF8888"])
    styler = styler.text_gradient(cmap=cmap)

    latex_str = styler.to_latex(
        position="!htbp",
        label="tab:2_distance_matrix",
        caption="Cosinus-Distanzmatrix der PCA-reduzierten Embeddings",
        hrules=True,
        convert_css=True,
        column_format="|"
        + "r|" * (len(distance_df.columns) + 1),  # Spaltenindex für bessere Lesbarkeit
    )
    latex_str = convert_to_side_table(latex_str)
    latex_str = latex_str.replace("word", "")
    out_path.write_text(latex_str)
    print(f"→ Distanztabelle gespeichert: {out_path}")


def convert_to_side_table(latex_str: str) -> str:
    # find  col index between \toprule and \midrule
    old_string_index = re.search(r"\\toprule(.*?)\\\\", latex_str, re.DOTALL).group(1)
    string_index = old_string_index[:]
    helper = string_index.split("&")
    for s in helper[1:]:
        string_index = string_index.replace(s, f"\\rotatebox{{90}}{{{s}}}")
    latex_str = latex_str.replace(old_string_index, string_index)

    # KORRIGIERT: LaTeX-Anpassungen für eine bessere Darstellung.
    latex_str = latex_str.replace("\\begin{table}", "\\begin{sidewaystable}")
    latex_str = latex_str.replace("\\end{table}", "\\end{sidewaystable}")
    latex_str = latex_str.replace(
        "\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}"
    )
    latex_str = latex_str.replace("\\end{tabular}", "\\end{tabular}}")
    return latex_str


if __name__ == "__main__":
    cfg = Config()
    viz = EmbeddingVisualizer(cfg)

    # 1. Daten laden und verarbeiten
    dataframe = viz.load_data()
    dataframe = viz.prepare_data(dataframe)

    # NEU: Farb-Mapping zentral definieren.
    color_mapping = {
        "Obst": "#FF9999",  # Hellrot
        "Farbe": "#99CCFF",  # Hellblau
        "Land": "#99FF99",  # Hellgrün
        "Beruf": "#FFCC99",  # Hellorange
        "Sport": "#FF99CC",  # Hellrosa
        "Tier": "#CCCCFF",  # Helllila
    }
    # 2. Matplotlib-Plot erstellen
    INTERACTIVE_MODE = False  # Setze auf False, um PGF-Export zu aktivieren

    if not INTERACTIVE_MODE:
        matplotlib.use("pgf")
        matplotlib.rcParams.update(
            {
                "pgf.texsystem": "xelatex",
                "font.family": "serif",
                "text.usetex": True,
                "pgf.rcfonts": False,
            }
        )

    figure = create_matplotlib_plot(df=dataframe, category_colors=color_mapping)
    if INTERACTIVE_MODE:
        plt.show()
    else:
        # Speichert den Plot im PGF-Format für LaTeX
        figure.savefig(cfg.output_dir / "embeddings_plot.pgf", bbox_inches="tight")
        print(f"→ PGF-Plot gespeichert: {cfg.output_dir / 'embeddings_plot.pgf'}")

    # 3. Tabellen für LaTeX exportieren
    export_embedding_table(
        df=dataframe,
        category_colors=color_mapping,
        out_path=cfg.output_dir / "embeddings_table.tex",
        decimals=cfg.round_decimals,
    )
    export_distance_table(
        df=dataframe,
        category_colors=color_mapping,
        out_path=cfg.output_dir / "2_1_distance_table.tex",
    )
