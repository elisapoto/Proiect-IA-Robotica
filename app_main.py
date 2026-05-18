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
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Proiect IA – Echipa",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

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
    #MainMenu { visibility: hidden; }
    footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🤖 Proiect Inteligenta Artificiala</h1>
    <p>Robotica cu RL · Optimizare TSP · NLP Sentiment Analysis</p>
</div>
""", unsafe_allow_html=True)

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

    pilon1_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pilon_1_RL_Robotica")
    if os.path.isdir(pilon1_dir) and pilon1_dir not in sys.path:
        sys.path.insert(0, pilon1_dir)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Configurare Antrenament RL")
        algorithm = st.selectbox("Algoritm RL", ["PPO", "DQN"])
        n_steps   = st.slider("Numar pasi antrenament",
                              min_value=1000, max_value=100000, value=10000, step=1000)
        start_btn = st.button("▶️ Porneste Antrenamentul", type="primary",
                              use_container_width=True)

        if start_btn:
            try:
                from antrenament_robot import CoppeliaRobotEnv
                from stable_baselines3 import PPO as PPO_SB3, DQN as DQN_SB3

                st.info("⏳ Se conecteaza la CoppeliaSim si incepe antrenamentul...")
                rewards_log = []

                env = CoppeliaRobotEnv()

                if algorithm == "PPO":
                    model = PPO_SB3("MlpPolicy", env, verbose=0, learning_rate=0.0003)
                else:
                    model = DQN_SB3("MlpPolicy", env, verbose=0, learning_rate=0.0003)

                progress = st.progress(0, text="Antrenament in curs...")
                status   = st.empty()

                CHUNK = max(1, n_steps // 20)
                for i in range(20):
                    model.learn(total_timesteps=CHUNK, reset_num_timesteps=(i == 0))
                    pct = int((i + 1) * 5)
                    progress.progress(pct, text=f"Antrenament... {pct}%")
                    status.text(f"Pasi completati: {(i+1)*CHUNK} / {n_steps}")

                model.save("creier_robot_mobil")
                env.close()
                progress.progress(100, text="Antrenament finalizat!")
                st.success("✅ Antrenament finalizat! Modelul salvat ca `creier_robot_mobil.zip`")

            except ImportError as e:
                st.error(f"⚠️ Librarie lipsa: {e}\n\nInstaleza: pip install stable-baselines3 coppeliasim-zmqremoteapi-client")
            except Exception as e:
                st.error(f"❌ Eroare conectare CoppeliaSim: {e}\n\nAsigura-te ca CoppeliaSim e deschis si simularea e pornita (▶️)!")

    with col2:
        st.markdown("### Grafic Convergenta")
        # Afiseaza grafic real daca exista date, altfel demo
        log_path = os.path.join(pilon1_dir if os.path.isdir(pilon1_dir) else ".", "rewards_log.npy")
        if os.path.exists(log_path):
            rewards = np.load(log_path)
            x = np.arange(len(rewards))
            title = f"Convergenta {algorithm} (real)"
        else:
            x = np.linspace(0, n_steps, 500)
            rewards = -80 * np.exp(-x / 15000) + 80 + np.random.normal(0, 5, 500)
            title = f"Convergenta {algorithm} (demo)"

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, rewards, color="#4C72B0", alpha=0.8, linewidth=1.5)
        ax.set_xlabel("Pasi antrenament")
        ax.set_ylabel("Reward total per episod")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        if not os.path.exists(log_path):
            st.caption("*Grafic de demonstratie – va fi inlocuit cu date reale dupa antrenament*")


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 – Optimizare TSP (Fata 2)
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="pilon-badge badge-2">PILON 2 – OPTIMIZARE</span>',
                unsafe_allow_html=True)

    pilon2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pilon_2_Optim")
    if os.path.isdir(pilon2_dir) and pilon2_dir not in sys.path:
        sys.path.insert(0, pilon2_dir)

    try:
        from tsp_solvers import TSPSolver

        st.markdown("## 🧭 Traveling Salesman Problem - AI Optimizers")

        n_cities = st.slider("Selectează numărul de orașe", 5, 25, 10, key="tsp_slider")
        np.random.seed(42)
        cities = np.random.rand(n_cities, 2) * 100
        solver = TSPSolver(cities)

        def plot_path(cities, path, title):
            fig, ax = plt.subplots()
            for i in range(len(path)):
                start = cities[path[i]]
                end   = cities[path[(i + 1) % len(path)]]
                ax.plot([start[0], end[0]], [start[1], end[1]], 'b-')
            ax.scatter(cities[:, 0], cities[:, 1], c='red')
            for i, (x, y) in enumerate(cities):
                ax.text(x, y, str(i))
            ax.set_title(title)
            return fig

        algo = st.selectbox(
            "Alege algoritmul",
            ["Nearest Neighbor", "Hill Climbing", "Simulated Annealing",
             "Genetic Algorithm", "Backtracking"],
            key="tsp_algo"
        )

        if st.button("🚀 Rulează algoritmul ales"):
            path, cost = None, None
            if algo == "Nearest Neighbor":
                path, cost = solver.solve_nn()
            elif algo == "Hill Climbing":
                path, cost = solver.solve_hc()
            elif algo == "Simulated Annealing":
                path, cost = solver.solve_sa()
            elif algo == "Genetic Algorithm":
                path, cost = solver.solve_ga()
            elif algo == "Backtracking":
                if n_cities > 10:
                    st.error("⚠️ Backtracking merge doar până la 10 orașe!")
                else:
                    path, cost = solver.solve_bkt()
            if path is not None:
                st.subheader(f"Cost total: {cost:.2f}")
                st.pyplot(plot_path(cities, path, algo))

        if st.button("📊 Compară toți algoritmii"):
            results = {}
            with st.spinner("Rulez algoritmii..."):
                results["NN"] = solver.solve_nn()[1]
                results["HC"] = solver.solve_hc()[1]
                results["SA"] = solver.solve_sa()[1]
                results["GA"] = solver.solve_ga()[1]
                if n_cities <= 10:
                    results["BKT"] = solver.solve_bkt()[1]
            st.subheader("📊 Comparare costuri")
            st.table(results)
            fig, ax = plt.subplots()
            ax.bar(results.keys(), results.values())
            ax.set_title("Comparare algoritmi TSP")
            st.pyplot(fig)

    except ImportError:
        st.warning(
            "⚠️ Fisierul `tsp_solvers.py` nu a fost gasit in `Pilon_2_Optim/`.\n\n"
            "Pune fisierul `tsp_solvers.py` al colegei tale in folderul `Pilon_2_Optim/`."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 – NLP Sentiment Analysis (Fata 3)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="pilon-badge badge-3">PILON 3 – NLP</span>',
                unsafe_allow_html=True)

    pilon3_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Pilon_3_NLP")
    if os.path.isdir(pilon3_dir) and pilon3_dir not in sys.path:
        sys.path.insert(0, pilon3_dir)

    try:
        from app_tab3 import render_tab3
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