"""
app_main.py
===========
Aplicatia Streamlit principala cu 3 tab-uri:
  Tab 1 – Robotica / RL (Fata 1)
  Tab 2 – Optimizare TSP (Fata 2)
  Tab 3 – NLP Sentiment Analysis (Fata 3)

Rulare:
    streamlit run app_main.py

IMPORTANT: Inainte de prima rulare, antreneaza modelul NLP:
    cd Pilon_3_NLP && python train_and_save.py
"""

import sys
import os
import streamlit as st

# ─── Configurare pagina (TREBUIE sa fie primul apel Streamlit) ─────────────────
st.set_page_config(
    page_title="Proiect IA – Echipa",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS global
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 1rem; }
    
    .pilon-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin-bottom: 8px;
    }
    .badge-1 { background: #dbeafe; color: #1d4ed8; }
    .badge-2 { background: #dcfce7; color: #166534; }
    .badge-3 { background: #fef3c7; color: #92400e; }
    
    /* Ascunde meniu Streamlit */
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🤖 Proiect Inteligenta Artificiala</h1>
    <p>Robotica cu RL · Optimizare TSP · NLP Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🤖 Tab 1 – Robotica RL",
    "🗺️ Tab 2 – Optimizare TSP",
    "🎬 Tab 3 – NLP Sentiment",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 – Robotica / RL (Fata 1)
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="pilon-badge badge-1">PILON 1 – ROBOTICA</span>', 
                unsafe_allow_html=True)
    st.markdown("## 🤖 Robot Mobil in CoppeliaSim cu Reinforcement Learning")
    
    st.info(
        "📋 **codul din `Pilon_1_RL/` aici.\n\n"
        "Poti importa functii din folderul tau sau afisa grafice/rezultate direct."
    )
    
    # Placeholder – Fata 1 va completa aceasta sectiune
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Configurare Antrenament RL")
        algorithm  = st.selectbox("Algoritm RL", ["PPO", "DQN"])
        n_steps    = st.slider("Numar pasi antrenament", 
                               min_value=1000, max_value=100000, value=50000, step=1000)
        start_btn  = st.button("▶️ Porneste Simularea", type="primary", 
                               use_container_width=True)
        if start_btn:
            st.warning("⚠️ Conecteaza CoppeliaSim si adauga codul RL din Pilon_1_RL/ !")
    
    with col2:
        st.markdown("### Grafic Convergenta")
        import numpy as np
        import matplotlib.pyplot as plt

        # Grafic demo
        fig, ax = plt.subplots(figsize=(6, 3))
        x = np.linspace(0, 50000, 500)
        y = -80 * np.exp(-x / 15000) + 80 + np.random.normal(0, 5, 500)
        ax.plot(x, y, color="#4C72B0", alpha=0.8, linewidth=1.5)
        ax.fill_between(x, y - 10, y + 10, alpha=0.1, color="#4C72B0")
        ax.set_xlabel("Pasi antrenament")
        ax.set_ylabel("Reward total per episod")
        ax.set_title(f"Convergenta {algorithm} (demo)")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("*Grafic de demonstratie – va fi inlocuit cu date reale din CoppeliaSim*")


with tab2:
    st.markdown('<span class="pilon-badge badge-2">PILON 2 – OPTIMIZARE</span>', 
                unsafe_allow_html=True)
    st.markdown("## 🗺️ Problema Comis-Voiajorului (TSP)")
    
    st.info(
        "📋 ** codul din `Pilon_2_Optim/` aici.\n\n"
        "Importa algoritmii tai (BKT, HC, SA, GA, NN) si afiseaza vizualizarile."
    )

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Configurare TSP")
        n_cities   = st.slider("Numar orase", min_value=5, max_value=20, value=12)
        tsp_algo   = st.selectbox("Algoritm", 
                                  ["Genetic Algorithm (GA)", 
                                   "Simulated Annealing (SA)",
                                   "Hill Climbing (HC)",
                                   "Backtracking (BKT)",
                                   "Neural Network (NN)"])
        solve_btn  = st.button("🔍 Rezolva TSP", type="primary", 
                               use_container_width=True)
        if solve_btn:
            st.warning("⚠️ Adauga algoritmii TSP din Pilon_2_Optim/ !")
    
    with col2:
        st.markdown("### Vizualizare Ruta Optima")
        import numpy as np, matplotlib.pyplot as plt
        
        rng = np.random.RandomState(42)
        x_cities = rng.rand(n_cities)
        y_cities = rng.rand(n_cities)
        route_idx = list(range(n_cities)) + [0]
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(x_cities[route_idx], y_cities[route_idx], 
                "b-o", linewidth=1.5, markersize=8, alpha=0.7)
        for i, (xi, yi) in enumerate(zip(x_cities, y_cities)):
            ax.annotate(f" C{i}", (xi, yi), fontsize=8)
        ax.set_title(f"Ruta TSP – {n_cities} orase (demo)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("*Ruta aleatoare de demonstratie – va fi inlocuita cu solutia algoritmului*")


with tab3:
    st.markdown('<span class="pilon-badge badge-3">PILON 3 – NLP</span>', 
                unsafe_allow_html=True)
    
    # Adauga folderul Pilon_3_NLP in path
    pilon3_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pilon_3_NLP")
    if os.path.isdir(pilon3_dir) and pilon3_dir not in sys.path:
        sys.path.insert(0, pilon3_dir)
    
    try:
        from Pilon_3_NLP.app_tab3 import render_tab3
        render_tab3()
    except ImportError as e:
        st.error(
            f"⚠️ Nu s-a putut importa componenta NLP: {e}\n\n"
            "Asigura-te ca `Pilon_3_NLP/app_tab3.py` si `Pilon_3_NLP/nlp_pipeline.py` "
            "exista in acelasi folder cu `app_main.py`."
        )
    except Exception as e:
        st.error(f"Eroare la incarcarea componentei NLP: {e}")
        st.exception(e)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "Proiect Inteligenta Artificiala · "
    "Pilon 1: Robotica RL · Pilon 2: Optimizare TSP · Pilon 3: NLP"
    "</div>",
    unsafe_allow_html=True,
)