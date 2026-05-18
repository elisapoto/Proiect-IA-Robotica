"""
Pilon 3 – NLP Pipeline
======================
Preprocesare, vectorizare TF-IDF si antrenare modele de clasificare sentiment.
Dataset: IMDb Movie Reviews (50.000 recenzii) de pe Hugging Face.

Parametri configurabili:
  - VOCAB_SIZE         : dimensiunea vocabularului TF-IDF
  - CLASSIFIER         : 'naive_bayes' | 'logistic_regression' | 'neural_network'
  - DATASET_FRACTION   : 0.1 | 0.5 | 1.0  (ce % din dataset sa folosim)
"""

import os
import re
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

# ─── Download resurse NLTK (o singura data) ───────────────────────────────────
def download_nltk_resources():
    for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ─── 1. INCARCARE DATASET ─────────────────────────────────────────────────────

def load_imdb_dataset(fraction: float = 1.0):
    """
    Incearca sa incarce IMDb din mai multe surse:
      1. Fisier local CSV (daca exista)
      2. Hugging Face datasets
      3. Dataset sintetic de demo (daca nimic altceva nu merge)
    """
    local_path = os.path.join(os.path.dirname(__file__), "imdb_dataset.csv")
    
    # Sursa 1: fisier local
    if os.path.exists(local_path):
        print(f"[INFO] Incarcare dataset local: {local_path}")
        df = pd.read_csv(local_path)
        df = df[["review", "sentiment"]].dropna()
        df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
    else:
        # Sursa 2: Hugging Face
        try:
            from datasets import load_dataset
            print("[INFO] Descarcare IMDb de pe Hugging Face...")
            ds = load_dataset("imdb")
            train_df = pd.DataFrame(ds["train"])
            test_df  = pd.DataFrame(ds["test"])
            df = pd.concat([train_df, test_df], ignore_index=True)
            df = df.rename(columns={"text": "review"})
            df["label"] = df["label"]
        except Exception as e:
            print(f"[WARN] Hugging Face indisponibil: {e}")
            print("[INFO] Se foloseste dataset sintetic de demonstratie...")
            df = _generate_demo_dataset()
    
    # Aplicare fractiune
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)
    
    print(f"[INFO] Dataset incarcat: {len(df)} exemple "
          f"({df['label'].sum()} pozitive, {(df['label']==0).sum()} negative)")
    return df


def _generate_demo_dataset():
    """Dataset sintetic de demo cu 2000 exemple."""
    positive_templates = [
        "This movie was absolutely fantastic and I loved every minute of it.",
        "An incredible film with outstanding performances from the entire cast.",
        "Brilliant storytelling and beautiful cinematography make this a must-see.",
        "I was completely blown away by this masterpiece of cinema.",
        "One of the best films I have ever seen in my life.",
        "The director did an amazing job bringing this story to life.",
        "A touching and heartfelt movie that left me wanting more.",
        "Superb acting and a gripping plot kept me on the edge of my seat.",
        "This film exceeded all my expectations and then some.",
        "A wonderful cinematic experience that I would highly recommend.",
    ]
    negative_templates = [
        "This movie was a complete waste of my time and money.",
        "Terrible acting and a boring plot ruined this film entirely.",
        "I cannot believe how bad this movie turned out to be.",
        "One of the worst films I have ever had to sit through.",
        "The story made no sense and the characters were awful.",
        "A disappointing and poorly made film with no redeeming qualities.",
        "I fell asleep halfway through this dreadful movie.",
        "The worst cinematic experience I have ever had.",
        "Avoid this film at all costs it is absolutely horrible.",
        "A total disaster from start to finish with no saving grace.",
    ]
    
    import random
    random.seed(42)
    reviews, labels = [], []
    for _ in range(1000):
        reviews.append(random.choice(positive_templates) + " " +
                       random.choice(positive_templates))
        labels.append(1)
        reviews.append(random.choice(negative_templates) + " " +
                       random.choice(negative_templates))
        labels.append(0)
    
    return pd.DataFrame({"review": reviews, "label": labels})


# ─── 2. PREPROCESARE TEXT ─────────────────────────────────────────────────────

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        # 1. Minuscule
        text = text.lower()
        # 2. Elimina tag-uri HTML
        text = re.sub(r"<[^>]+>", " ", text)
        # 3. Elimina caractere non-alfabetice
        text = re.sub(r"[^a-z\s]", " ", text)
        # 4. Elimina spatii multiple
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_lemmatize(self, text: str) -> str:
        tokens = text.split()
        tokens = [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 2
        ]
        return " ".join(tokens)

    def preprocess(self, text: str) -> str:
        return self.tokenize_and_lemmatize(self.clean(text))

    def preprocess_series(self, series: pd.Series) -> pd.Series:
        print(f"[INFO] Preprocesare {len(series)} texte...")
        return series.apply(self.preprocess)


# ─── 3. VECTORIZARE TF-IDF ────────────────────────────────────────────────────

def build_tfidf_vectorizer(max_features: int = 10000):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),   # unigrame + bigrame
        sublinear_tf=True,
        min_df=2,
    )


# ─── 4. MODELE DE CLASIFICARE ─────────────────────────────────────────────────

def build_classifier(name: str):
    name = name.lower().replace(" ", "_")
    if name == "naive_bayes":
        return MultinomialNB(alpha=0.1)
    elif name == "logistic_regression":
        return LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs",
                                  random_state=42)
    elif name == "neural_network":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            max_iter=50,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False,
        )
    else:
        raise ValueError(f"Classifier necunoscut: {name}. "
                         f"Alege din: naive_bayes, logistic_regression, neural_network")


# ─── 5. PIPELINE COMPLET ──────────────────────────────────────────────────────

class SentimentPipeline:
    """
    Pipeline complet: preprocesare → vectorizare → clasificare.
    
    Parametri:
        vocab_size       : int   – dimensiunea vocabularului TF-IDF
        classifier_name  : str   – 'naive_bayes' | 'logistic_regression' | 'neural_network'
        dataset_fraction : float – fractiunea din dataset (0.0-1.0)
    """

    def __init__(
        self,
        vocab_size: int = 10000,
        classifier_name: str = "logistic_regression",
        dataset_fraction: float = 1.0,
    ):
        self.vocab_size = vocab_size
        self.classifier_name = classifier_name
        self.dataset_fraction = dataset_fraction

        self.preprocessor = TextPreprocessor()
        self.vectorizer    = build_tfidf_vectorizer(max_features=vocab_size)
        self.classifier    = build_classifier(classifier_name)
        
        self.metrics = {}
        self.is_trained = False

    def fit(self, df: pd.DataFrame):
        """Antreneaza pipeline-ul pe un DataFrame cu coloanele 'review' si 'label'."""
        print(f"\n{'='*60}")
        print(f"  ANTRENARE – {self.classifier_name.upper()}")
        print(f"  Vocab: {self.vocab_size} | Fractiune: {self.dataset_fraction*100:.0f}%")
        print(f"{'='*60}")

        # Preprocesare
        t0 = time.time()
        df["processed"] = self.preprocessor.preprocess_series(df["review"])
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            df["processed"], df["label"],
            test_size=0.2, random_state=42, stratify=df["label"]
        )

        # Vectorizare
        print("[INFO] Vectorizare TF-IDF...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec  = self.vectorizer.transform(X_test)

        # Antrenare
        print(f"[INFO] Antrenare {self.classifier_name}...")
        t_train = time.time()
        self.classifier.fit(X_train_vec, y_train)
        train_time = time.time() - t_train

        # Evaluare
        y_pred = self.classifier.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        cm  = confusion_matrix(y_test, y_pred)

        self.metrics = {
            "accuracy":   acc,
            "f1_score":   f1,
            "conf_matrix": cm,
            "train_time": train_time,
            "total_time": time.time() - t0,
            "n_train":    len(X_train),
            "n_test":     len(X_test),
            "y_test":     y_test,
            "y_pred":     y_pred,
        }
        self.is_trained = True

        print(f"\n[REZULTATE]")
        print(f"  Acuratete : {acc:.4f} ({acc*100:.2f}%)")
        print(f"  F1-Score  : {f1:.4f}")
        print(f"  Timp antrenare: {train_time:.2f}s")
        print(classification_report(y_test, y_pred,
                                    target_names=["Negativ", "Pozitiv"]))
        return self

    def predict(self, text: str):
        """Prezice sentimentul unui text nou. Returneaza (label, probability)."""
        if not self.is_trained:
            raise RuntimeError("Pipeline-ul nu este antrenat! Ruleaza .fit() mai intai.")
        processed = self.preprocessor.preprocess(text)
        vec = self.vectorizer.transform([processed])
        label = self.classifier.predict(vec)[0]
        if hasattr(self.classifier, "predict_proba"):
            prob = self.classifier.predict_proba(vec)[0][label]
        else:
            prob = 1.0
        return ("pozitiv" if label == 1 else "negativ"), float(prob)

    def save(self, path: str = "pilon3_model.pkl"):
        """Salveaza pipeline-ul antrenat."""
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[INFO] Model salvat in: {path}")

    @staticmethod
    def load(path: str = "pilon3_model.pkl"):
        """Incarca un pipeline salvat."""
        with open(path, "rb") as f:
            return pickle.load(f)


# ─── 6. VIZUALIZARI ───────────────────────────────────────────────────────────

def plot_confusion_matrix(pipeline: SentimentPipeline, save_path: str = None):
    """Deseneaza matricea de confuzie."""
    cm = pipeline.metrics["conf_matrix"]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Negativ", "Pozitiv"],
        yticklabels=["Negativ", "Pozitiv"],
        ax=ax
    )
    ax.set_xlabel("Prezis", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    ax.set_title(
        f"Matrice de Confuzie – {pipeline.classifier_name.replace('_', ' ').title()}",
        fontsize=13
    )
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Grafic salvat: {save_path}")
    return fig


def plot_comparison(results: list, save_path: str = None):
    """
    Grafic comparativ al performantei pentru mai multi clasificatori.
    results = [{"name": str, "accuracy": float, "f1": float, "time": float}, ...]
    """
    names = [r["name"].replace("_", "\n") for r in results]
    accs  = [r["accuracy"] for r in results]
    f1s   = [r["f1"] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Comparatie Algoritmi NLP – IMDb Sentiment", fontsize=14, fontweight="bold")

    # Acuratete
    axes[0].bar(x - width/2, accs, width, label="Acuratete", color="#4C72B0")
    axes[0].bar(x + width/2, f1s,  width, label="F1-Score",  color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=10)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].set_ylabel("Scor")
    axes[0].set_title("Acuratete si F1-Score")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    for i, (a, f) in enumerate(zip(accs, f1s)):
        axes[0].text(i - width/2, a + 0.005, f"{a:.3f}", ha="center", fontsize=8)
        axes[0].text(i + width/2, f + 0.005, f"{f:.3f}", ha="center", fontsize=8)

    # Timp
    times = [r["time"] for r in results]
    bars = axes[1].bar(names, times, color=["#4C72B0", "#DD8452", "#55A868"])
    axes[1].set_ylabel("Timp (secunde)")
    axes[1].set_title("Timp de Antrenare")
    axes[1].grid(axis="y", alpha=0.3)
    for bar, t in zip(bars, times):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.1,
                     f"{t:.1f}s", ha="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Grafic comparativ salvat: {save_path}")
    return fig


# ─── 7. MAIN (rulare standalone) ──────────────────────────────────────────────

if __name__ == "__main__":
    # ── Parametri configurabili ──
    VOCAB_SIZE       = 15000
    CLASSIFIER       = "logistic_regression"   # naive_bayes / logistic_regression / neural_network
    DATASET_FRACTION = 0.5                     # 10%, 50% sau 100%

    # Incarcare dataset
    df = load_imdb_dataset(fraction=DATASET_FRACTION)

    # Antrenare
    pipeline = SentimentPipeline(
        vocab_size=VOCAB_SIZE,
        classifier_name=CLASSIFIER,
        dataset_fraction=DATASET_FRACTION,
    ).fit(df)

    # Vizualizare
    os.makedirs("outputs", exist_ok=True)
    plot_confusion_matrix(pipeline, save_path="outputs/confusion_matrix.png")
    plt.show()

    # Salvare model
    pipeline.save("pilon3_model.pkl")

    # Test rapid
    test_reviews = [
        "This movie was absolutely fantastic! I loved every second of it.",
        "Terrible film, complete waste of time. I hated every minute.",
        "It was okay, not great but not bad either.",
    ]
    print("\n[TEST PREDICTII]")
    for review in test_reviews:
        label, prob = pipeline.predict(review)
        print(f"  '{review[:60]}...' → {label.upper()} ({prob*100:.1f}%)")