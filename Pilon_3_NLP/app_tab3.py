"""
app_tab3.py
===========
Componenta Streamlit pentru Tab-ul 3 – NLP Sentiment Analysis.
Poate fi importata in app_main.py sau rulata standalone cu:
    streamlit run app_tab3.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Asigura ca putem importa din acelasi folder
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

MODEL_PATH = os.path.join(_DIR, "pilon3_model.pkl")
OUTPUT_DIR = os.path.join(_DIR, "outputs")


# ─── Incarcare model (cu cache Streamlit) ─────────────────────────────────────

@st.cache_resource(show_spinner="Se incarca modelul NLP...")
def load_model():
    """Incarca modelul salvat. Daca nu exista, antreneaza unul rapid."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    else:
        # Antrenare rapida de urgenta pe 10% din date
        st.warning("⚠️ Modelul salvat nu a fost gasit. "
                   "Se antreneaza un model de urgenta pe 10% din date...")
        from nlp_pipeline import load_imdb_dataset, SentimentPipeline
        df = load_imdb_dataset(fraction=0.1)
        pipeline = SentimentPipeline(
            vocab_size=10000,
            classifier_name="logistic_regression",
            dataset_fraction=0.1,
        ).fit(df)
        pipeline.save(MODEL_PATH)
        return pipeline


# ─── Functii ajutatoare ───────────────────────────────────────────────────────

def sentiment_badge(label: str, prob: float) -> str:
    """Returneaza HTML pentru badge sentiment."""
    if label == "pozitiv":
        color = "#22c55e"
        emoji = "😊"
        text  = "POZITIV"
    else:
        color = "#ef4444"
        emoji = "😞"
        text  = "NEGATIV"
    
    return f"""
    <div style="
        display: inline-block;
        background: {color};
        color: white;
        padding: 12px 24px;
        border-radius: 50px;
        font-size: 1.4rem;
        font-weight: bold;
        letter-spacing: 2px;
        box-shadow: 0 4px 15px {color}60;
        margin: 10px 0;
    ">
        {emoji} {text}
    </div>
    <div style="
        font-size: 1rem;
        color: #666;
        margin-top: 8px;
    ">
        Probabilitate: <strong style="color:{color}">{prob*100:.1f}%</strong>
    </div>
    """


def draw_probability_bar(prob: float, label: str):
    """Deseneaza bara de probabilitate cu matplotlib."""
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.patch.set_facecolor("none")
    ax.set_facecolor("none")
    
    pos_prob = prob if label == "pozitiv" else 1 - prob
    neg_prob = 1 - pos_prob

    ax.barh(0, neg_prob, color="#ef4444", height=0.5, label="Negativ")
    ax.barh(0, pos_prob, left=neg_prob, color="#22c55e", height=0.5, label="Pozitiv")
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8, frameon=False,
              bbox_to_anchor=(1, 1.5), ncol=2)
    ax.text(neg_prob / 2, 0, f"{neg_prob*100:.1f}%",
            ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    ax.text(neg_prob + pos_prob / 2, 0, f"{pos_prob*100:.1f}%",
            ha="center", va="center", color="white", fontweight="bold", fontsize=9)
    
    plt.tight_layout()
    return fig


# ─── COMPONENTA PRINCIPALA ────────────────────────────────────────────────────

def render_tab3():
    """
    Randeaza continutul Tab-ului 3 in aplicatia Streamlit.
    Apeleaza aceasta functie din app_main.py in contextul tab-ului 3.
    """
    st.markdown("## 🎬 Analiza Sentiment – Recenzii de Film")
    st.markdown(
        "Model antrenat pe **IMDb Movie Reviews** (50.000 recenzii). "
        "Scrie o recenzie in engleza si afla instant daca e **pozitiva** sau **negativa**."
    )

    # ── Incarcare model ──
    pipeline = load_model()

    # ── Informatii model ──
    clf_name = pipeline.classifier_name.replace("_", " ").title()
    vocab    = pipeline.vocab_size
    frac     = int(pipeline.dataset_fraction * 100)

    with st.expander("ℹ️ Informatii despre modelul curent", expanded=False):
        col1, col2, col3 = st.columns(3)
        col1.metric("Algoritm", clf_name)
        col2.metric("Vocabular TF-IDF", f"{vocab:,} cuvinte")
        col3.metric("Date antrenare", f"{frac}% din dataset")

        if pipeline.metrics:
            c1, c2, c3 = st.columns(3)
            c1.metric("Acuratete", f"{pipeline.metrics['accuracy']*100:.2f}%")
            c2.metric("F1-Score",  f"{pipeline.metrics['f1_score']:.4f}")
            c3.metric("Timp antrenare", f"{pipeline.metrics['train_time']:.1f}s")

    st.markdown("---")

    # ── Zona de predictie ──
    st.markdown("### ✍️ Scrie o recenzie de film")
    
    # Exemple rapide
    st.markdown("**Exemple rapide:**")
    col_ex1, col_ex2, col_ex3 = st.columns(3)
    
    example_pos = "This movie was an absolute masterpiece. The acting was brilliant, the story was compelling, and I was on the edge of my seat the entire time. Highly recommended!"
    example_neg = "What a terrible waste of time. The plot made no sense, the characters were flat and boring, and the special effects looked cheap. I want my two hours back."
    example_neu = "The movie had some good moments but overall it was just average. Nothing special, nothing terrible. The acting was decent but the story felt rushed."

    if col_ex1.button("😊 Recenzie pozitiva", use_container_width=True):
        st.session_state["nlp_review_text"] = example_pos
    if col_ex2.button("😞 Recenzie negativa", use_container_width=True):
        st.session_state["nlp_review_text"] = example_neg
    if col_ex3.button("😐 Recenzie neutra", use_container_width=True):
        st.session_state["nlp_review_text"] = example_neu

    # Textarea
    review_text = st.text_area(
        label="Recenzie (in engleza):",
        value=st.session_state.get("nlp_review_text", ""),
        height=150,
        placeholder="Write your movie review here in English...",
        key="nlp_textarea",
    )

    # Buton analiza
    analyze_btn = st.button("🔍 Analizeaza Sentimentul", type="primary",
                            use_container_width=True)

    if analyze_btn:
        if not review_text or len(review_text.strip()) < 10:
            st.warning("⚠️ Te rog introdu o recenzie mai lunga (minim 10 caractere).")
        else:
            with st.spinner("Se analizeaza..."):
                label, prob = pipeline.predict(review_text)

            st.markdown("### 📊 Rezultat")
            
            # Badge sentiment
            st.markdown(sentiment_badge(label, prob), unsafe_allow_html=True)
            
            # Bara probabilitate
            fig = draw_probability_bar(prob, label)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            # Detalii preprocesare
            with st.expander("🔬 Cum a fost procesat textul?"):
                processed = pipeline.preprocessor.preprocess(review_text)
                st.markdown("**Text original:**")
                st.text(review_text[:300] + ("..." if len(review_text) > 300 else ""))
                st.markdown("**Dupa preprocesare (curatat + lemmatizat):**")
                st.text(processed[:300] + ("..." if len(processed) > 300 else ""))
                
                # Top cuvinte cheie
                words = processed.split()
                if words:
                    st.markdown(f"**Cuvinte cheie identificate ({len(words)} total):**")
                    st.markdown(" ".join([f"`{w}`" for w in words[:20]]))

    # ─── Sectiunea de grafice si comparatii ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Grafice si Comparatii")

    # Grafic comparativ daca exista
    comparison_img = os.path.join(OUTPUT_DIR, "comparison.png")
    if os.path.exists(comparison_img):
        st.image(comparison_img,
                 caption="Comparatie algoritmi – Acuratete, F1-Score si Timp antrenare",
                 use_container_width=True)
    
    # Matrice de confuzie daca exista
    cm_imgs = [
        os.path.join(OUTPUT_DIR, f"confusion_{clf}.png")
        for clf in ["naive_bayes", "logistic_regression", "neural_network"]
    ]
    cm_imgs_exist = [p for p in cm_imgs if os.path.exists(p)]

    if cm_imgs_exist:
        st.markdown("**Matrici de Confuzie per Algoritm:**")
        cols = st.columns(len(cm_imgs_exist))
        labels_map = {
            "naive_bayes": "Naive Bayes",
            "logistic_regression": "Logistic Regression",
            "neural_network": "Neural Network",
        }
        for col, img_path in zip(cols, cm_imgs_exist):
            clf_key = os.path.basename(img_path).replace("confusion_", "").replace(".png", "")
            col.image(img_path, caption=labels_map.get(clf_key, clf_key),
                      use_container_width=True)
    
    # Matrice de confuzie a modelului curent (calculata live)
    if pipeline.metrics and "conf_matrix" in pipeline.metrics:
        if not cm_imgs_exist:  # arata doar daca nu exista imagini salvate
            st.markdown("**Matrice de Confuzie – Model Curent:**")
            cm = pipeline.metrics["conf_matrix"]
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Negativ", "Pozitiv"],
                        yticklabels=["Negativ", "Pozitiv"], ax=ax)
            ax.set_xlabel("Prezis")
            ax.set_ylabel("Real")
            ax.set_title(f"Confusion Matrix – {clf_name}")
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Tabel rezultate daca exista
    csv_path = os.path.join(OUTPUT_DIR, "results_table.csv")
    if os.path.exists(csv_path):
        st.markdown("**Tabel Comparativ Final:**")
        df_res = pd.read_csv(csv_path)
        st.dataframe(df_res, use_container_width=True, hide_index=True)

    # ─── Sectiunea de re-antrenare cu parametri custom ───────────────────────
    st.markdown("---")
    st.markdown("### ⚙️ Experimenteaza cu Parametri")
    st.markdown("Modifica parametrii si reantreneaza modelul direct din interfata.")

    with st.form("retrain_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        
        new_vocab = col_p1.selectbox(
            "Vocabular TF-IDF",
            options=[5000, 10000, 15000, 20000],
            index=1,
        )
        new_clf = col_p2.selectbox(
            "Algoritm",
            options=["naive_bayes", "logistic_regression", "neural_network"],
            format_func=lambda x: x.replace("_", " ").title(),
            index=1,
        )
        new_frac = col_p3.select_slider(
            "% din Dataset",
            options=[10, 25, 50, 75, 100],
            value=50,
        )
        
        retrain_btn = st.form_submit_button(
            "🚀 Reantreneaza Modelul",
            type="primary",
            use_container_width=True,
        )

    if retrain_btn:
        from nlp_pipeline import load_imdb_dataset, SentimentPipeline

        progress_bar = st.progress(0, text="Se incarca dataset-ul...")
        df = load_imdb_dataset(fraction=new_frac / 100)
        progress_bar.progress(30, text="Se antreneaza modelul...")
        
        new_pipeline = SentimentPipeline(
            vocab_size=new_vocab,
            classifier_name=new_clf,
            dataset_fraction=new_frac / 100,
        ).fit(df)
        
        progress_bar.progress(90, text="Se salveaza modelul...")
        new_pipeline.save(MODEL_PATH)
        
        # Invalideaza cache-ul
        load_model.clear()
        
        progress_bar.progress(100, text="Gata!")
        
        acc = new_pipeline.metrics["accuracy"]
        f1  = new_pipeline.metrics["f1_score"]
        t   = new_pipeline.metrics["train_time"]
        
        st.success(
            f"✅ Model reantrenat cu succes!\n\n"
            f"**{new_clf.replace('_', ' ').title()}** | "
            f"Vocab: {new_vocab:,} | "
            f"Date: {new_frac}%\n\n"
            f"📊 Acuratete: **{acc*100:.2f}%** | F1: **{f1:.4f}** | Timp: **{t:.1f}s**"
        )
        st.rerun()


# ─── Rulare standalone ────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="NLP Sentiment – Pilon 3",
        page_icon="🎬",
        layout="wide",
    )
    render_tab3()