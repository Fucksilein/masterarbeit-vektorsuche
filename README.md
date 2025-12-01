# masterarbeit-vektorsuche
Code, Experimente und Auswertungen zur Masterarbeit zur Optimierung der Vektorsuche mit Embeddings und Gewichtungsmethoden.

## Reproduzierbarkeit der Ergebnisse

### Embeddings neu berechnen
Wenn alle Embeddings vollständig neu berechnet werden sollen (ACHTUNG: je nach Hardware mehrere Tage Laufzeit), sind dafür die folgenden vier Skripte vorgesehen.  
Vor dem Ausführen bitte eine `.env`-Datei erstellen oder `default.env` entsprechend anpassen.

* `001_run_install.sh`
* `002_run_prepare_data_embedder.sh`
* `003_run_find_weights.sh`
* `004_run_compare.sh`
### Analyse der Ergebnisse
Alle Auswertungen befinden sich im Notebook: `code/auswertung_method_comparison.ipynb` 
Falls Embeddings neu berechnet wurden, müssen im Notebook die Modell-IDs angepasst werden.

## Lizenz
Der Code in diesem Repository steht unter der MIT-Lizenz. Siehe die Datei `LICENSE` für Details.

## Zitation

Wenn du dieses Repository wissenschaftlich verwendest, zitiere es bitte wie folgt:

Fuchs, T. (2025). *Begleitendes Repository zur Masterarbeit
"Vergleich und Optimierung von Ansätzen zur Verbesserung der Vektorsuche in großen Datensätzen – Eine Analyse von Embedding- und Gewichtungsmethoden"*. GitHub-Repository, verfügbar unter: https://github.com/Fucksilein/masterarbeit-vektorsuche

### BibTeX

```bibtex
@online{fuchs2025masterarbeitrepo,
  author  = {Fuchs, Thomas},
  title   = {Begleitendes Repository zur Masterarbeit: Vergleich und Optimierung von Ansätzen zur Verbesserung der Vektorsuche in großen Datensätzen -- Eine Analyse von Embedding- und Gewichtungsmethoden},
  year    = {2025},
  url     = {https://github.com/Fucksilein/masterarbeit-vektorsuche},
  urldate = {2025-11-23},
}
