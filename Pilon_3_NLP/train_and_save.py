"""
train_and_save.py
=================
Script de antrenare complet pentru Pilonul 3 – NLP.

Ruleaza ACEST fisier o singura data inainte de a porni aplicatia Streamlit.
Va antrena toti cei 3 clasificatori, va genera graficele si va salva modelul cel mai bun.

Utilizare:
    cd Pilon_3_NLP
    python train_and_save.py

Dupa rulare vei gasi:
    pilon3_model.pkl         – modelul salvat (folosit de aplicatia Streamlit)
    outputs/confusion_matrix.png
    outputs/comparison.png
    outputs/results_table.csv
"""

import os
import sys
import time

# Adauga folderul curent in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlp_pipeline import (
    load_imdb_dataset,
    SentimentPipeline,
    plot_confusion_matrix,
    plot_comparison,
)
import matplotlib.pyplot as plt
import pandas as pd

# ─── CONFIGURARE ──────────────────────────────────────────────────────────────

VOCAB_SIZE       = 15000
DATASET_FRACTION = 1.0      # folosim tot dataset-ul (schimba in 0.1 pentru test rapid)
OUTPUT_DIR       = "outputs"
MODEL_PATH       = "pilon3_model.pkl"

CLASSIFIERS = [
    "naive_bayes",
    "logistic_regression",
    "neural_network",
]

# ─── ANTRENARE ────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  PILON 3 – NLP | ANTRENARE MODELE SENTIMENT")
    print("=" * 60)

    # Incarca dataset o singura data
    df = load_imdb_dataset(fraction=DATASET_FRACTION)

    results = []
    best_pipeline = None
    best_acc      = 0.0

    for clf_name in CLASSIFIERS:
        print(f"\n>>> Antrenare: {clf_name} <<<")
        
        pipeline = SentimentPipeline(
            vocab_size=VOCAB_SIZE,
            classifier_name=clf_name,
            dataset_fraction=DATASET_FRACTION,
        ).fit(df)

        acc  = pipeline.metrics["accuracy"]
        f1   = pipeline.metrics["f1_score"]
        t    = pipeline.metrics["train_time"]

        results.append({
            "name":     clf_name,
            "accuracy": acc,
            "f1":       f1,
            "time":     t,
        })

        # Matrice de confuzie individuala
        fig = plot_confusion_matrix(
            pipeline,
            save_path=os.path.join(OUTPUT_DIR, f"confusion_{clf_name}.png")
        )
        plt.close(fig)

        # Pastreaza cel mai bun model
        if acc > best_acc:
            best_acc      = acc
            best_pipeline = pipeline

    # ── Grafic comparativ ──
    print("\n[INFO] Generare grafic comparativ...")
    fig = plot_comparison(results, save_path=os.path.join(OUTPUT_DIR, "comparison.png"))
    plt.show()
    plt.close(fig)

    # ── Tabel CSV ──
    df_results = pd.DataFrame(results)
    df_results.columns = ["Algoritm", "Acuratete", "F1-Score", "Timp (s)"]
    df_results["Acuratete"] = df_results["Acuratete"].map(lambda x: f"{x:.4f}")
    df_results["F1-Score"]  = df_results["F1-Score"].map(lambda x: f"{x:.4f}")
    df_results["Timp (s)"]  = df_results["Timp (s)"].map(lambda x: f"{x:.2f}")
    csv_path = os.path.join(OUTPUT_DIR, "results_table.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n[INFO] Tabel rezultate salvat: {csv_path}")
    print(df_results.to_string(index=False))

    # ── Salveaza cel mai bun model ──
    print(f"\n[INFO] Cel mai bun model: {best_pipeline.classifier_name} "
          f"(acuratete {best_acc:.4f})")
    best_pipeline.save(MODEL_PATH)
    
    print(f"\n{'='*60}")
    print("  ANTRENARE COMPLETA! Acum ruleaza:")
    print("  streamlit run ../app_main.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()