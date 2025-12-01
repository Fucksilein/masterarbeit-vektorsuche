import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Arc
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances


# --- Konfiguration ---
@dataclass
class Config:
    """Kapselt alle Konfigurationsparameter für das Skript."""

    output_dir: Path = Path("src/code_output/")
    embedding_model: str = "BAAI/bge-m3"
    pca_seed: int = 42
    pca_components: int = 2
    # Gewichte für Titel, Beschreibung und Preis
    product_weights: List[float] = field(default_factory=lambda: [0.6, 0.3, 0.1])

    # Daten für die Analyse
    data: Dict = field(
        default_factory=lambda: {
            "products": [
                {
                    "title": "Landkarte von Deutschland",
                    "description": "Wie so oft in modernen Zeiten steht in der Beschreibung ganz etwas anderes als im Titel. Gerade bei modernen Inhalten, die vielleicht sogar mithilfe künstlicher Intelligenz erstellt wurden, ist das nicht unüblich. Dennoch werden auch dessen Nachbarländer: Belgien, Niederlande, Dänemark, Polen, Tschechien, Österreich, Schweiz, Frankreich, Luxemburg und Liechtenstein, erwähnt",
                    "price": 9.99,
                },
                {
                    "title": "Kulinarisches Österreich",
                    "description": "In altes Buch handgeschriebes Österreich Kochbuch, die die beliebtesten Gerichte und Rezepte des Landes zeigt. Diese Essenslandkarte enthält eine Vielzahl traditioneller österreichischer Spezialitäten wie Wiener Schnitzel, Sachertorte und Apfelstrudel.",
                    "price": 129.90,
                },
            ],
            "query": "Künstliche Intelligenz in der Gastronomie unter 20 Euro",
        }
    )

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Stelle sicher, dass die Gewichte zu einer Numpy-Array normalisiert werden
        self.product_weights = np.array(self.product_weights)
        self.product_weights /= np.sum(self.product_weights)


class Product:
    """Datenklasse für ein einzelnes Produkt und seine Embeddings."""

    def __init__(
        self,
        title: str,
        description: str,
        price: float,
        encoder: callable,
        weights: np.ndarray,
    ):
        self.title = title
        self.description = description
        self.price = price
        self.encoder = encoder
        self.weights = weights

        # Generiere Embeddings für die verschiedenen Teile
        self.combined_text = f"Titel: {self.title}\nBeschreibung: {self.description}\nPreis: {self.price} €"
        self.prod_emb = self.encoder(self.combined_text)
        self.title_emb = self.encoder(self.title)
        self.desc_emb = self.encoder(self.description)
        self.price_emb = self.encoder(f"{self.price} €")

    @property
    def weighted_emb(self) -> np.ndarray:
        """Kombiniert die einzelnen Embeddings gewichtet."""
        return (
            self.weights[0] * self.title_emb
            + self.weights[1] * self.desc_emb
            + self.weights[2] * self.price_emb
        )


class ProductEmbeddingVisualizer:
    """Kapselt die Logik zur Erstellung und Visualisierung von Produkt-Embeddings."""

    def __init__(self, config: Config):
        self.cfg = config
        self.model = SentenceTransformer(
            self.cfg.embedding_model, trust_remote_code=True
        )
        self.reducer = PCA(
            n_components=self.cfg.pca_components, random_state=self.cfg.pca_seed
        )
        self.encoder = functools.partial(
            self.model.encode, convert_to_numpy=True, normalize_embeddings=True
        )

    def prepare_data(self) -> pd.DataFrame:
        """Verarbeitet die Rohdaten, erstellt Embeddings und reduziert die Dimensionen."""
        p1 = Product(
            **self.cfg.data["products"][0],
            encoder=self.encoder,
            weights=self.cfg.product_weights,
        )
        p2 = Product(
            **self.cfg.data["products"][1],
            encoder=self.encoder,
            weights=self.cfg.product_weights,
        )
        q_emb = self.encoder(self.cfg.data["query"])

        labels = [
            "p1_prod",
            "p2_prod",
            "p1_weighted",
            "p2_weighted",
            "p1_title",
            "p1_desc",
            "p1_price",
            "p2_title",
            "p2_desc",
            "p2_price",
            "query",
        ]
        embeddings = np.array(
            [
                p1.prod_emb,
                p2.prod_emb,
                p1.weighted_emb,
                p2.weighted_emb,
                p1.title_emb,
                p1.desc_emb,
                p1.price_emb,
                p2.title_emb,
                p2.desc_emb,
                p2.price_emb,
                q_emb,
            ]
        )

        proj_coords = self.reducer.fit_transform(embeddings)
        df = pd.DataFrame(proj_coords, columns=["x", "y"], index=labels)

        # Füge Original-Embeddings für spätere Distanzberechnungen hinzu (optional)
        # df['original_embedding'] = list(embeddings)
        return df


# --- Plotting Hilfsfunktionen ---


def draw_vector(ax: plt.Axes, end: np.ndarray, label: str, color: str, **kwargs):
    """Zeichnet einen Vektor vom Ursprung zum Endpunkt."""
    ax.arrow(
        0,
        0,
        end["x"],
        end["y"],
        head_width=0.015,
        head_length=0.02,
        fc=color,
        ec=color,
        label=label,
        length_includes_head=True,
        **kwargs,
    )


def draw_angle_arc(
    ax: plt.Axes, v1: np.ndarray, v2: np.ndarray, radius: float, color: str
):
    """Zeichnet den kleinsten Winkelbogen zwischen zwei Vektoren."""
    angle1 = np.degrees(np.arctan2(v1["y"], v1["x"]))
    angle2 = np.degrees(np.arctan2(v2["y"], v2["x"]))

    # Winkel in [0, 360) normalisieren
    angle1 %= 360
    angle2 %= 360

    # Differenz berechnen
    diff = (angle2 - angle1) % 360
    if diff > 180:
        # Falls der größere Winkel gewählt wurde, drehe die Richtung um
        angle1, angle2 = angle2, angle1
        diff = 360 - diff

    start_angle = angle1
    end_angle = angle1 + diff

    arc = Arc(
        (0, 0),
        radius * 2,
        radius * 2,
        angle=0,
        theta1=start_angle,
        theta2=end_angle,
        color=color,
        linewidth=1.5,
        linestyle="--",
    )
    ax.add_patch(arc)


def create_distance_annotation(ax: plt.Axes, text: str):
    """Platziert eine formatierte Textbox oben rechts im Plot."""
    ax.text(
        0.98,
        0.02,
        text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8, ec="black"),
    )


def set_common_layout(ax: plt.Axes, title: str):
    """Definiert das gemeinsame Layout für alle Plots."""
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel("PCA Komponente 1")
    ax.set_ylabel("PCA Komponente 2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower left", fontsize=9)


# --- Plot-Erstellungsfunktionen ---


def create_plot_1(df: pd.DataFrame, colors: Dict) -> plt.Figure:
    """Plot 1: Konkatenation von Textfeldern."""
    fig, ax = plt.subplots(figsize=(10, 6))
    q2d, p1, p2 = df.loc["query"], df.loc["p1_prod"], df.loc["p2_prod"]

    draw_vector(ax, q2d, "Query", colors["query"], zorder=10)
    draw_vector(ax, p1, "Produkt 1 (kombiniert)", colors["p1"], zorder=5)
    draw_vector(ax, p2, "Produkt 2 (kombiniert)", colors["p2"], zorder=5)

    draw_angle_arc(ax, q2d, p1, 0.1, colors["p1"])
    draw_angle_arc(ax, q2d, p2, 0.15, colors["p2"])

    dist_p1 = cosine_distances(q2d.values.reshape(1, -1), p1.values.reshape(1, -1))[
        0, 0
    ]
    dist_p2 = cosine_distances(q2d.values.reshape(1, -1), p2.values.reshape(1, -1))[
        0, 0
    ]

    text = (
        f"$\\textbf{{Methode\\ 1:\\ Distanzen}}$\n"
        f"Query ↔ P1 (Landkarte): ${dist_p1:.3f}$\n"
        f"Query ↔ P2 (Kulinarik): ${dist_p2:.3f}$"
    )
    create_distance_annotation(ax, text)
    set_common_layout(ax, "Methode 1: Konkatenation von Textfeldern")
    plt.tight_layout()
    return fig


def create_plot_2(df: pd.DataFrame, colors: Dict, weights: np.ndarray) -> plt.Figure:
    """Plot 2: Gewichtete Kombination von Vektoren."""
    fig, ax = plt.subplots(figsize=(10, 6))
    q2d = df.loc["query"]
    p1_w, p2_w = df.loc["p1_weighted"], df.loc["p2_weighted"]

    # Finale gewichtete Vektoren
    draw_vector(ax, p1_w, "Produkt 1 (gewichtet)", colors["p1"], zorder=10)
    draw_vector(ax, p2_w, "Produkt 2 (gewichtet)", colors["p2"], zorder=10)

    # Gestapelte Komponentenvektoren
    for p, name in [("p1", "P1"), ("p2", "P2")]:
        c_title = weights[0] * df.loc[f"{p}_title"]
        c_desc = weights[1] * df.loc[f"{p}_desc"]
        start_desc = c_title
        start_price = c_title + c_desc

        ax.arrow(
            0,
            0,
            c_title["x"],
            c_title["y"],
            ls="--",
            fc=colors[f"{p}_title_light"],
            ec=colors[f"{p}_title_light"],
        )
        ax.arrow(
            start_desc["x"],
            start_desc["y"],
            c_desc["x"],
            c_desc["y"],
            ls="--",
            fc=colors[f"{p}_desc_light"],
            ec=colors[f"{p}_desc_light"],
        )
        ax.arrow(
            start_price["x"],
            start_price["y"],
            df.loc[f"{p}_price"]["x"] * weights[2],
            df.loc[f"{p}_price"]["y"] * weights[2],
            ls="--",
            fc=colors[f"{p}_price_light"],
            ec=colors[f"{p}_price_light"],
        )

    draw_vector(ax, q2d, "Query", colors["query"], zorder=12)
    draw_angle_arc(ax, q2d, p1_w, 0.1, colors["p1"])
    draw_angle_arc(ax, q2d, p2_w, 0.15, colors["p2"])

    dist_p1 = cosine_distances(q2d.values.reshape(1, -1), p1_w.values.reshape(1, -1))[
        0, 0
    ]
    dist_p2 = cosine_distances(q2d.values.reshape(1, -1), p2_w.values.reshape(1, -1))[
        0, 0
    ]
    text = (
        f"$\\textbf{{Methode\\ 2:\\ Distanzen}}$\n"
        f"Query ↔ P1 (Landkarte): ${dist_p1:.3f}$\n"
        f"Query ↔ P2 (Kulinarik): ${dist_p2:.3f}$"
    )
    create_distance_annotation(ax, text)
    set_common_layout(ax, "Methode 2: Gewichtete Kombination von Vektoren")
    ax.legend(
        loc="lower left",
        fontsize=9,
        handles=[
            plt.Line2D(
                [0], [0], color=colors["p1"], lw=2, label="Produkt 1 (gewichtet)"
            ),
            plt.Line2D(
                [0], [0], color=colors["p2"], lw=2, label="Produkt 2 (gewichtet)"
            ),
            plt.Line2D([0], [0], color=colors["query"], lw=2, label="Query"),
        ],
    )
    plt.tight_layout()
    return fig


def create_plot_3(df: pd.DataFrame, colors: Dict, weights: np.ndarray) -> plt.Figure:
    """Plot 3: Paarweise Distanzberechnung."""
    fig, ax = plt.subplots(figsize=(10, 6))
    q2d = df.loc["query"]

    draw_vector(ax, q2d, "Query", colors["query"], zorder=10)
    for p, name in [("p1", "P1"), ("p2", "P2")]:
        for att, att_name in [
            ("title", "Titel"),
            ("desc", "Beschr."),
            ("price", "Preis"),
        ]:
            draw_vector(
                ax,
                df.loc[f"{p}_{att}"],
                f"{name} {att_name}",
                colors[f"{p}_{att}"],
                ls="dotted",
                zorder=5,
            )

    dist_text_parts = ["$\\textbf{Methode\\ 3:\\ Paarweise\\ Distanzen}$"]
    agg_dists = {}
    for p, name in [("p1", "Landkarte"), ("p2", "Kulinarik")]:
        dists = {}
        for att in ["title", "desc", "price"]:
            dists[att] = cosine_distances(
                q2d.values.reshape(1, -1), df.loc[f"{p}_{att}"].values.reshape(1, -1)
            )[0, 0]
        agg_dist = np.sum(weights * np.array(list(dists.values())))
        agg_dists[p] = agg_dist

        dist_text_parts.append(
            f"$\\textbf{{P{p[1]}\\ ({name})}}$ | Agg.: $\\textbf{{{agg_dist:.3f}}}$\n"
            f"  Titel: ${dists['title']:.3f}$, Beschr.: ${dists['desc']:.3f}$, Preis: ${dists['price']:.3f}$"
        )

    create_distance_annotation(ax, "\n".join(dist_text_parts))
    set_common_layout(ax, "Methode 3: Paarweise Distanzberechnung")
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    INTERACTIVE_MODE = (
        False  # Setze auf True für interaktive Plots, False für PGF-Export
    )

    cfg = Config()
    visualizer = ProductEmbeddingVisualizer(cfg)

    # 1. Daten laden, verarbeiten und reduzieren
    coords_df = visualizer.prepare_data()

    # 2. Farb-Mapping zentral definieren
    color_mapping = {
        "p1": "#00796B",  # Türkis
        "p2": "#FFA000",  # Orange
        "query": "#D32F2F",  # Rot
        "p1_title": "#4DB6AC",
        "p1_desc": "#80CBC4",
        "p1_price": "#B2DFDB",
        "p2_title": "#FFC107",
        "p2_desc": "#FFD54F",
        "p2_price": "#FFECB3",
        "p1_title_light": "#B2DFDB",
        "p1_desc_light": "#E0F2F1",
        "p1_price_light": "#E0F2F1",
        "p2_title_light": "#FFECB3",
        "p2_desc_light": "#FFF8E1",
        "p2_price_light": "#FFF8E1",
    }

    # 3. Matplotlib-Backend für LaTeX/PGF konfigurieren (falls nicht interaktiv)
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

    # 4. Plots erstellen und speichern/anzeigen
    plot_functions = {
        "verknuepfung_methode1.pgf": (
            create_plot_1,
            {"df": coords_df, "colors": color_mapping},
        ),
        "verknuepfung_methode2.pgf": (
            create_plot_2,
            {"df": coords_df, "colors": color_mapping, "weights": cfg.product_weights},
        ),
        "verknuepfung_methode3.pgf": (
            create_plot_3,
            {"df": coords_df, "colors": color_mapping, "weights": cfg.product_weights},
        ),
    }

    for filename, (func, kwargs) in plot_functions.items():
        print(f"Erstelle Plot: {filename}...")
        figure = func(**kwargs)

        if INTERACTIVE_MODE:
            plt.show()
        else:
            output_path = cfg.output_dir / filename
            figure.savefig(output_path, bbox_inches="tight")
            print(f"→ Plot gespeichert: {output_path}")
