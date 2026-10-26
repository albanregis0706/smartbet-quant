import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.stats import poisson

st.set_page_config(page_title="Arbitrage Engine v6", layout="centered")
st.title("📊 Arbitrage Engine v6")
st.subheader("Modélisation Quantitative Football")

st.sidebar.header("💰 Capital & Risque")
bankroll = st.sidebar.number_input("Bankroll totale (€)", min_value=10.0, value=1000.0, step=50.0)
risk_co = st.sidebar.slider("Coefficient de Kelly (Sécurité)", 0.1, 1.0, 0.25, step=0.05)

col1, col2 = st.columns(2)
with col1:
    st.header("🏠 Domicile")
    team_h = st.text_input("Nom de l'équipe 1", "PSG")
    xg_h = st.slider("Indice Attaque (xG)", 0.5, 3.5, 2.1, step=0.1)
    fatigue_h = st.slider("Fatigue Équipe 1", 0.8, 1.0, 0.95, step=0.05)

with col2:
    st.header("🚀 Extérieur")
    team_a = st.text_input("Nom de l'équipe 2", "Lyon")
    xg_a = st.slider("Indice Attaque (xG)", 0.5, 3.5, 1.4, step=0.1)
    fatigue_a = st.slider("Fatigue Équipe 2", 0.8, 1.0, 1.0, step=0.05)

st.header("🎰 Cotes du Marché")
cc1, cc2, cc3 = st.columns(3)
with cc1: cote_h = st.number_input(f"Cote {team_h}", min_value=1.01, value=1.55, step=0.05)
with cc2: cote_n = st.number_input("Cote Nul", min_value=1.01, value=4.40, step=0.05)
with cc3: cote_a = st.number_input(f"Cote {team_a}", min_value=1.01, value=5.80, step=0.05)

lambda_h = xg_h * fatigue_h
lambda_a = xg_a * fatigue_a

max_goals = 6
matrix = np.zeros((max_goals, max_goals))

for i in range(max_goals):
    for j in range(max_goals):
        matrix[i, j] = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)

prob_h = np.sum(np.tril(matrix, -1))
prob_n = np.sum(np.diag(matrix))
prob_a = np.sum(np.triu(matrix, 1))

total_prob = prob_h + prob_n + prob_a
prob_h /= total_prob
prob_n /= total_prob
prob_a /= total_prob

st.header("📈 Analyse des Probabilités")
mc1, mc2, mc3 = st.columns(3)
mc1.metric(f"Victoire {team_h}", f"{prob_h*100:.1f}%")
mc2.metric("Match Nul", f"{prob_n*100:.1f}%")
mc3.metric(f"Victoire {team_a}", f"{prob_a*100:.1f}%")

st.header("🎯 Opportunités d'Arbitrage")

def check_value(prob, cote, nom_pari):
    prob_imp = 1 / cote
    alpha = prob - prob_imp
    if alpha > 0:
        f_kelly = (prob * cote - 1) / (cote - 1)
        mise_suggeree = max(0.0, f_kelly * bankroll * risk_co)
        mise_suggeree = min(mise_suggeree, bankroll * 0.05)
        st.success(f"🔥 VALUE DETECTÉE sur [{nom_pari}] ! Probabilité Réelle ({prob*100:.1f}%) > Implicite ({prob_imp*100:.1f}%). Espérance positive : +{alpha*100:.1f}%. Mise suggérée : {mise_suggeree:.2f} €")
    else:
        st.info(f"Pas de value sur [{nom_pari}]. Le marché est efficient.")

check_value(prob_h, cote_h, f"Victoire {team_h}")
check_value(prob_n, cote_n, "Match Nul")
check_value(prob_a, cote_a, f"Victoire {team_a}")

st.header("🗺️ Matrice des Scores Exacts")
fig = go.Figure(data=go.Heatmap(
    z=matrix,
    x=[str(i) for i in range(max_goals)],
    y=[str(i) for i in range(max_goals)],
    colorscale='Viridis',
    hovertemplate=f'{team_h}: %{{x}}<br>{team_a}: %{{y}}<br>Probabilité: %{{z:.2%}}<extra></extra>'
))
fig.update_layout(xaxis_title=f"Buts {team_h}", yaxis_title=f"Buts {team_a}")
st.plotly_chart(fig, use_container_width=True)
