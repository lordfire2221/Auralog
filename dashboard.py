"""
AuraLog — dashboard.py
Dashboard Streamlit pour la visualisation des prédictions ML.

Onglets :
  Vue d'ensemble  — KPIs, distribution des risques, timeline
  Alertes         — table filtrée des événements critiques
  Heatmap         — risque par heure et par jour
  Services        — classement des services à risque
  Modèles ML      — métriques IF + XGBoost

Usage :
  streamlit run dashboard.py
  streamlit run dashboard.py --server.port 8501
"""

import os
import pickle
import glob
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Config page ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AuraLog — Dashboard",
    page_icon   = "🔮",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Constantes ────────────────────────────────────────────────────────────────
PREDICTIONS_DIR = Path(os.getenv("AURALOG_PREDICTIONS_DIR", "./predictions"))
MODEL_DIR       = Path(os.getenv("AURALOG_MODEL_DIR",       "./models"))

RISK_COLORS = {
    "🔴 Critique": "#ef4444",
    "🟠 Élevé":    "#f97316",
    "🟡 Modéré":   "#eab308",
    "🟢 Normal":   "#22c55e",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)
def load_latest_predictions() -> pd.DataFrame:
    """Charge le fichier de prédictions le plus récent."""
    files = sorted(PREDICTIONS_DIR.glob("predictions_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame()
    df = pd.read_csv(files[0], low_memory=False)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].astype(str)
    return df


@st.cache_data(ttl=300)
def load_all_predictions() -> tuple[pd.DataFrame, list[str]]:
    """Charge tous les fichiers de prédictions disponibles."""
    files = sorted(PREDICTIONS_DIR.glob("predictions_*.csv"), reverse=True)
    if not files:
        return pd.DataFrame(), []
    labels = [f.stem.replace("predictions_", "") for f in files]
    return files, labels


def load_model_meta() -> dict:
    """Charge les métadonnées des modèles depuis les fichiers pkl."""
    meta = {}
    for name in ["isolation_forest", "xgboost"]:
        candidates = [
            p for p in MODEL_DIR.glob(f"{name}_*.pkl")
            if "_test_" not in p.name
        ]
        if candidates:
            latest = max(candidates, key=lambda p: p.stat().st_mtime)
            meta[name] = {
                "path":    latest.name,
                "size_kb": round(latest.stat().st_size / 1024, 1),
                "date":    datetime.fromtimestamp(latest.stat().st_mtime)
                           .strftime("%Y-%m-%d %H:%M"),
            }
    return meta


# ═══════════════════════════════════════════════════════════════════════════════
#  COMPOSANTS UI RÉUTILISABLES
# ═══════════════════════════════════════════════════════════════════════════════

def kpi_card(col, label: str, value, delta=None, color: str = "#6366f1"):
    with col:
        st.markdown(
            f"""<div style="background:#1e1e2e;border-left:4px solid {color};
                padding:16px;border-radius:8px;margin-bottom:8px">
                <div style="color:#94a3b8;font-size:13px">{label}</div>
                <div style="color:#f1f5f9;font-size:28px;font-weight:700">{value}</div>
                {"<div style='color:#94a3b8;font-size:12px'>"+str(delta)+"</div>" if delta else ""}
            </div>""",
            unsafe_allow_html=True,
        )


def risk_gauge(score: float) -> go.Figure:
    """Jauge du score de risque global."""
    color = (
        "#ef4444" if score >= 75 else
        "#f97316" if score >= 50 else
        "#eab308" if score >= 25 else
        "#22c55e"
    )
    fig = go.Figure(go.Indicator(
        mode  = "gauge+number",
        value = score,
        title = {"text": "Score de risque moyen", "font": {"color": "#94a3b8"}},
        number= {"font": {"color": color, "size": 40}},
        gauge = {
            "axis":       {"range": [0, 100], "tickcolor": "#64748b"},
            "bar":        {"color": color},
            "bgcolor":    "#1e1e2e",
            "bordercolor":"#334155",
            "steps": [
                {"range": [0,  25], "color": "#14532d"},
                {"range": [25, 50], "color": "#713f12"},
                {"range": [50, 75], "color": "#7c2d12"},
                {"range": [75,100], "color": "#450a0a"},
            ],
        },
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", font_color="#94a3b8",
        height=250, margin=dict(t=40, b=10, l=20, r=20),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.image(
        "https://img.shields.io/badge/AuraLog-ML%20Dashboard-6366f1?style=for-the-badge",
        use_column_width=True,
    )
    st.sidebar.markdown("---")

    # Sélecteur de fichier
    files, labels = load_all_predictions()
    if labels:
        selected = st.sidebar.selectbox(
            "📁 Fichier de prédictions",
            options=range(len(labels)),
            format_func=lambda i: labels[i],
        )
        if selected > 0:
            df = pd.read_csv(files[selected], low_memory=False)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            if "risk_level" in df.columns:
                df["risk_level"] = df["risk_level"].astype(str)

    st.sidebar.markdown("### 🎛️ Filtres")

    # Filtre niveau de risque
    levels = ["🔴 Critique", "🟠 Élevé", "🟡 Modéré", "🟢 Normal"]
    selected_levels = st.sidebar.multiselect(
        "Niveau de risque",
        options  = levels,
        default  = levels,
    )

    # Filtre service / processus
    if "process" in df.columns:
        all_procs = sorted(df["process"].dropna().unique().tolist())
        selected_procs = st.sidebar.multiselect(
            "Processus",
            options = all_procs,
            default = [],
            placeholder = "Tous les processus",
        )
    else:
        selected_procs = []

    # Filtre plage horaire
    if "timestamp" in df.columns:
        min_dt = df["timestamp"].min()
        max_dt = df["timestamp"].max()
        if pd.notna(min_dt) and pd.notna(max_dt):
            st.sidebar.markdown("**Plage de dates**")
            st.sidebar.text(f"{min_dt.strftime('%Y-%m-%d')} → {max_dt.strftime('%Y-%m-%d')}")

    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Rafraîchir les données"):
        st.cache_data.clear()
        st.rerun()

    # Appliquer les filtres
    filtered = df.copy()
    if selected_levels and "risk_level" in filtered.columns:
        filtered = filtered[filtered["risk_level"].isin(selected_levels)]
    if selected_procs and "process" in filtered.columns:
        filtered = filtered[filtered["process"].isin(selected_procs)]

    st.sidebar.markdown(f"**{len(filtered):,}** logs affichés / {len(df):,}")
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
#  ONGLET 1 — VUE D'ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════════════

def tab_overview(df: pd.DataFrame):
    # ── KPIs
    total       = len(df)
    n_critical  = int((df["risk_score"] >= 75).sum()) if "risk_score" in df.columns else 0
    n_high      = int(((df["risk_score"] >= 50) & (df["risk_score"] < 75)).sum()) if "risk_score" in df.columns else 0
    n_anomalies = int(df["anomaly_flag"].sum()) if "anomaly_flag" in df.columns else 0
    avg_risk    = round(df["risk_score"].mean(), 1) if "risk_score" in df.columns else 0
    error_rate  = round(df["log_level"].eq("ERROR").mean() * 100, 2) if "log_level" in df.columns else 0

    cols = st.columns(5)
    kpi_card(cols[0], "Total logs",      f"{total:,}",       color="#6366f1")
    kpi_card(cols[1], "🔴 Critiques",    f"{n_critical:,}",  color="#ef4444")
    kpi_card(cols[2], "🟠 Élevés",       f"{n_high:,}",      color="#f97316")
    kpi_card(cols[3], "Anomalies IF",    f"{n_anomalies:,}", color="#a855f7")
    kpi_card(cols[4], "Taux d'erreur",   f"{error_rate}%",   color="#06b6d4")

    st.markdown("---")

    # ── Jauge + Distribution
    col_gauge, col_dist = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(risk_gauge(avg_risk), use_container_width=True)

    with col_dist:
        if "risk_level" in df.columns:
            counts = df["risk_level"].value_counts().reset_index()
            counts.columns = ["level", "count"]
            order  = ["🔴 Critique", "🟠 Élevé", "🟡 Modéré", "🟢 Normal"]
            counts["level"] = pd.Categorical(counts["level"], categories=order, ordered=True)
            counts = counts.sort_values("level")
            colors = [RISK_COLORS.get(l, "#6366f1") for l in counts["level"]]

            fig = go.Figure(go.Bar(
                x=counts["count"], y=counts["level"],
                orientation="h",
                marker_color=colors,
                text=[f"{c:,}" for c in counts["count"]],
                textposition="outside",
            ))
            fig.update_layout(
                title="Distribution des niveaux de risque",
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#94a3b8", height=220,
                margin=dict(t=40, b=10, l=10, r=60),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Timeline du score de risque
    if "timestamp" in df.columns and "risk_score" in df.columns:
        st.markdown("#### 📈 Évolution du score de risque")
        df_time = df.copy()
        df_time["hour"] = df_time["timestamp"].dt.floor("1h")
        timeline = df_time.groupby("hour").agg(
            risk_mean   = ("risk_score", "mean"),
            risk_max    = ("risk_score", "max"),
            n_anomalies = ("anomaly_flag", "sum") if "anomaly_flag" in df_time.columns else ("risk_score", "count"),
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=timeline["hour"], y=timeline["risk_mean"].round(1),
            name="Risque moyen", line=dict(color="#6366f1", width=2),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=timeline["hour"], y=timeline["risk_max"].round(1),
            name="Risque max", line=dict(color="#ef4444", width=1, dash="dot"),
        ), secondary_y=False)
        fig.add_trace(go.Bar(
            x=timeline["hour"], y=timeline["n_anomalies"],
            name="Anomalies", marker_color="rgba(168,85,247,0.4)",
        ), secondary_y=True)

        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=300,
            margin=dict(t=20, b=40, l=20, r=20),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b", title="Score de risque"),
        )
        fig.update_yaxes(title_text="Nb anomalies", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ONGLET 2 — ALERTES
# ═══════════════════════════════════════════════════════════════════════════════

def tab_alerts(df: pd.DataFrame):
    # Filtrer les alertes significatives
    if "risk_score" not in df.columns:
        st.warning("Pas de données de score de risque disponibles.")
        return

    threshold = st.slider("Seuil de risque minimum", 0, 100, 50, step=5)
    alerts = df[df["risk_score"] >= threshold].copy()

    col1, col2, col3 = st.columns(3)
    col1.metric("Alertes filtrées",   f"{len(alerts):,}")
    col2.metric("Score max",           f"{alerts['risk_score'].max():.1f}" if len(alerts) else "N/A")
    col3.metric("Score moyen",         f"{alerts['risk_score'].mean():.1f}" if len(alerts) else "N/A")

    if alerts.empty:
        st.success("✅ Aucune alerte au-dessus du seuil.")
        return

    # Table des alertes
    display_cols = [c for c in [
        "timestamp", "risk_level", "risk_score",
        "process", "host_agent", "log_level",
        "failure_proba", "anomaly_flag", "message"
    ] if c in alerts.columns]

    alerts_display = alerts[display_cols].sort_values(
        "risk_score", ascending=False
    ).head(200).reset_index(drop=True)

    # Formatage
    if "timestamp" in alerts_display.columns:
        alerts_display["timestamp"] = alerts_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    if "risk_score" in alerts_display.columns:
        alerts_display["risk_score"] = alerts_display["risk_score"].round(1)
    if "failure_proba" in alerts_display.columns:
        alerts_display["failure_proba"] = alerts_display["failure_proba"].round(3)
    if "message" in alerts_display.columns:
        alerts_display["message"] = alerts_display["message"].str[:80]

    st.dataframe(
        alerts_display,
        use_container_width = True,
        height              = 450,
        column_config       = {
            "risk_score":    st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
            "failure_proba": st.column_config.ProgressColumn("Proba panne", min_value=0, max_value=1),
            "anomaly_flag":  st.column_config.CheckboxColumn("Anomalie"),
        },
    )

    # Export CSV
    csv = alerts_display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Exporter les alertes (CSV)",
        data=csv, file_name=f"alertes_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ONGLET 3 — HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════

def tab_heatmap(df: pd.DataFrame):
    if "timestamp" not in df.columns or "risk_score" not in df.columns:
        st.warning("Données temporelles non disponibles.")
        return

    df_h = df.copy()
    df_h["hour"] = df_h["timestamp"].dt.hour
    df_h["date"] = df_h["timestamp"].dt.date
    df_h["dow"]  = df_h["timestamp"].dt.day_name()

    col1, col2 = st.columns(2)

    # ── Heatmap heure x jour de semaine
    with col1:
        st.markdown("#### 🗓️ Risque : heure × jour de semaine")
        pivot = df_h.pivot_table(
            values  = "risk_score",
            index   = "dow",
            columns = "hour",
            aggfunc = "mean",
        ).reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

        fig = px.imshow(
            pivot,
            color_continuous_scale = "RdYlGn_r",
            labels = dict(x="Heure", y="Jour", color="Risque moyen"),
            aspect = "auto",
        )
        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=320,
            margin=dict(t=20, b=40, l=80, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Distribution horaire
    with col2:
        st.markdown("#### ⏰ Score de risque par heure")
        hourly = df_h.groupby("hour").agg(
            risk_mean   = ("risk_score", "mean"),
            n_anomalies = ("anomaly_flag", "sum") if "anomaly_flag" in df_h.columns else ("risk_score", "count"),
        ).reset_index()

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=hourly["hour"], y=hourly["risk_mean"].round(1),
            marker=dict(
                color=hourly["risk_mean"],
                colorscale="RdYlGn_r",
                cmin=0, cmax=100,
            ),
            text=hourly["risk_mean"].round(1),
            textposition="outside",
        ))
        fig2.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=320,
            margin=dict(t=20, b=40, l=20, r=20),
            xaxis=dict(title="Heure", gridcolor="#1e293b", dtick=2),
            yaxis=dict(title="Score moyen", gridcolor="#1e293b"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Timeline par date
    if df_h["date"].nunique() > 1:
        st.markdown("#### 📅 Évolution quotidienne")
        daily = df_h.groupby("date").agg(
            risk_mean   = ("risk_score", "mean"),
            n_logs      = ("risk_score", "count"),
            n_critical  = ("risk_score", lambda x: (x >= 75).sum()),
        ).reset_index()

        fig3 = make_subplots(specs=[[{"secondary_y": True}]])
        fig3.add_trace(go.Scatter(
            x=daily["date"], y=daily["risk_mean"].round(1),
            name="Risque moyen", fill="tozeroy",
            line=dict(color="#6366f1"),
            fillcolor="rgba(99,102,241,0.15)",
        ), secondary_y=False)
        fig3.add_trace(go.Bar(
            x=daily["date"], y=daily["n_critical"],
            name="Alertes critiques", marker_color="rgba(239,68,68,0.6)",
        ), secondary_y=True)
        fig3.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=280,
            margin=dict(t=20, b=40, l=20, r=20),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  ONGLET 4 — SERVICES
# ═══════════════════════════════════════════════════════════════════════════════

def tab_services(df: pd.DataFrame):
    if "process" not in df.columns or "risk_score" not in df.columns:
        st.warning("Données de service non disponibles.")
        return

    # Agrégation par processus
    agg = df.groupby("process").agg(
        n_logs      = ("risk_score", "count"),
        risk_max    = ("risk_score", "max"),
        risk_mean   = ("risk_score", "mean"),
        n_anomalies = ("anomaly_flag", "sum") if "anomaly_flag" in df.columns else ("risk_score", "count"),
        n_critical  = ("risk_score", lambda x: (x >= 75).sum()),
    ).reset_index().sort_values("risk_max", ascending=False)

    top_n = st.slider("Nombre de services à afficher", 5, 30, 10)
    agg_top = agg.head(top_n)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🏆 Top services par risque max")
        fig = go.Figure(go.Bar(
            y=agg_top["process"],
            x=agg_top["risk_max"].round(1),
            orientation="h",
            marker=dict(
                color=agg_top["risk_max"],
                colorscale="RdYlGn_r", cmin=0, cmax=100,
            ),
            text=agg_top["risk_max"].round(1),
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=max(300, top_n * 28),
            margin=dict(t=20, b=20, l=20, r=60),
            xaxis=dict(range=[0, 110], showgrid=False),
            yaxis=dict(showgrid=False, autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📊 Anomalies vs volume")
        fig2 = px.scatter(
            agg_top,
            x           = "n_logs",
            y           = "n_anomalies",
            size        = "risk_max",
            color       = "risk_mean",
            color_continuous_scale = "RdYlGn_r",
            hover_name  = "process",
            labels      = dict(n_logs="Volume de logs",
                               n_anomalies="Nb anomalies",
                               risk_mean="Risque moyen"),
        )
        fig2.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=380,
            margin=dict(t=20, b=40, l=20, r=20),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Table détaillée
    st.markdown("#### 📋 Détail par service")
    agg_display = agg_top.copy()
    agg_display["risk_max"]  = agg_display["risk_max"].round(1)
    agg_display["risk_mean"] = agg_display["risk_mean"].round(1)
    st.dataframe(
        agg_display,
        use_container_width = True,
        column_config = {
            "risk_max":  st.column_config.ProgressColumn("Risque max",  min_value=0, max_value=100),
            "risk_mean": st.column_config.ProgressColumn("Risque moyen",min_value=0, max_value=100),
            "n_logs":    st.column_config.NumberColumn("Logs",    format="%d"),
            "n_anomalies":st.column_config.NumberColumn("Anomalies", format="%d"),
            "n_critical": st.column_config.NumberColumn("Critiques", format="%d"),
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  ONGLET 5 — MODÈLES ML
# ═══════════════════════════════════════════════════════════════════════════════

def tab_incidents(df: pd.DataFrame):
    """Onglet Incidents & Prédictions préventives."""
    try:
        from alert_report import detect_incidents, detect_predicted_incidents, PREDICTION_HORIZON_MIN
    except Exception:
        st.error("Module alert_report.py introuvable — copiez-le dans le dossier.")
        return

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    incidents   = detect_incidents(df)
    predictions = detect_predicted_incidents(df, threshold=0.5)

    # ── KPIs
    n_critical = sum(1 for i in incidents if i["is_critical"])
    cols = st.columns(4)
    kpi_card(cols[0], "Incidents totaux",   f"{len(incidents)}",   color="#6366f1")
    kpi_card(cols[1], "🔴 Critiques",       f"{n_critical}",       color="#ef4444")
    kpi_card(cols[2], f"🔮 Préventif +{PREDICTION_HORIZON_MIN}min", f"{len(predictions)}", color="#f97316")
    kpi_card(cols[3], "Services impactés", f"{len(set(i['service'] for i in incidents))}", color="#06b6d4")

    st.markdown("---")

    # ── Prédictions préventives
    st.markdown(f"#### 🔮 Prédictions préventives — erreur probable dans les {PREDICTION_HORIZON_MIN} min")
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df["timestamp"]     = pred_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        pred_df["failure_proba"] = pred_df["failure_proba"].round(3)
        st.dataframe(
            pred_df, use_container_width=True,
            column_config={
                "failure_proba": st.column_config.ProgressColumn(
                    "Probabilité", min_value=0, max_value=1),
                "current_risk": st.column_config.ProgressColumn(
                    "Risque actuel", min_value=0, max_value=100),
            },
        )
    else:
        st.success(f"✅ Aucune erreur prédite dans les {PREDICTION_HORIZON_MIN} prochaines minutes.")

    st.markdown("---")

    # ── Timeline des incidents
    st.markdown("#### 🚨 Timeline des incidents détectés")
    if incidents:
        inc_df = pd.DataFrame(incidents)

        fig = go.Figure()
        for _, inc in inc_df.iterrows():
            color = "#ef4444" if inc["is_critical"] else "#f97316"
            fig.add_trace(go.Scatter(
                x=[inc["start"], inc["end"]],
                y=[inc["service"], inc["service"]],
                mode="lines+markers",
                line=dict(color=color, width=8),
                marker=dict(size=10, color=color),
                name=inc["service"],
                showlegend=False,
                hovertemplate=(
                    f"<b>{inc['service']}</b><br>"
                    f"Risque max: {inc['max_risk']:.1f}<br>"
                    f"Erreurs: {inc['total_errors']:,}<br>"
                    f"Durée: {inc['duration_min']:.0f} min<extra></extra>"
                ),
            ))

        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=max(250, len(inc_df["service"].unique()) * 60),
            margin=dict(t=20, b=40, l=20, r=20),
            xaxis=dict(title="Temps", gridcolor="#1e293b"),
            yaxis=dict(title="Service", gridcolor="#1e293b"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table détaillée
        st.markdown("#### 📋 Détail des incidents")
        display = inc_df.copy()
        display["start"] = display["start"].dt.strftime("%Y-%m-%d %H:%M")
        display["end"]   = display["end"].dt.strftime("%Y-%m-%d %H:%M")
        display["is_critical"] = display["is_critical"].map({True: "🔴 Oui", False: "🟠 Non"})
        display = display.rename(columns={
            "service": "Service", "start": "Début", "end": "Fin",
            "duration_min": "Durée (min)", "max_risk": "Risque max",
            "n_buckets": "Buckets", "is_critical": "Critique",
            "total_errors": "Erreurs", "total_logs": "Logs", "max_proba": "Proba max",
        })
        st.dataframe(
            display.sort_values("Risque max", ascending=False),
            use_container_width=True,
            column_config={"Risque max": st.column_config.ProgressColumn(
                "Risque max", min_value=0, max_value=100)},
        )

        # Export
        csv = display.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Exporter les incidents (CSV)", data=csv,
                          file_name=f"incidents_{datetime.now():%Y%m%d_%H%M}.csv",
                          mime="text/csv")
    else:
        st.success("✅ Aucun incident détecté sur la période.")


def tab_models(df: pd.DataFrame):
    meta = load_model_meta()

    col1, col2 = st.columns(2)

    # ── Isolation Forest
    with col1:
        st.markdown("#### 🌲 Isolation Forest")
        if "isolation_forest" in meta:
            m = meta["isolation_forest"]
            st.info(f"📦 `{m['path']}`\n\n📅 {m['date']} — {m['size_kb']} KB")

        if "anomaly_flag" in df.columns and "anomaly_score" in df.columns:
            n_total   = len(df)
            n_anom    = int(df["anomaly_flag"].sum())
            pct       = round(n_anom / n_total * 100, 2)

            kpi_col1, kpi_col2 = st.columns(2)
            kpi_col1.metric("Anomalies détectées", f"{n_anom:,}")
            kpi_col2.metric("Taux d'anomalie",     f"{pct}%")

            # Distribution des scores
            fig = px.histogram(
                df, x="anomaly_score", nbins=50,
                color_discrete_sequence=["#a855f7"],
                labels={"anomaly_score": "Score d'anomalie"},
                title="Distribution des scores Isolation Forest",
            )
            fig.add_vline(x=0.5, line_dash="dash", line_color="#ef4444",
                         annotation_text="Seuil")
            fig.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#94a3b8", height=280,
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── XGBoost
    with col2:
        st.markdown("#### 🚀 XGBoost (prédictif +5min)")
        if "xgboost" in meta:
            m = meta["xgboost"]
            st.info(f"📦 `{m['path']}`\n\n📅 {m['date']} — {m['size_kb']} KB")

        if "failure_proba" in df.columns:
            n_high_risk = int((df["failure_proba"] >= 0.9).sum())
            avg_proba   = round(df["failure_proba"].mean() * 100, 2)

            kpi_col1, kpi_col2 = st.columns(2)
            kpi_col1.metric("Proba ≥ 90%",    f"{n_high_risk:,}")
            kpi_col2.metric("Proba moyenne", f"{avg_proba}%")

            # Distribution des probabilités
            fig2 = px.histogram(
                df, x="failure_proba", nbins=50,
                color_discrete_sequence=["#f97316"],
                labels={"failure_proba": "Probabilité de panne"},
                title="Distribution des probabilités XGBoost",
            )
            fig2.add_vline(x=0.9, line_dash="dash", line_color="#ef4444",
                          annotation_text="Seuil 0.90")
            fig2.update_layout(
                paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
                font_color="#94a3b8", height=280,
                margin=dict(t=40, b=20, l=20, r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Corrélation IF × XGBoost
    if "anomaly_score" in df.columns and "failure_proba" in df.columns:
        st.markdown("#### 🔗 Corrélation Isolation Forest × XGBoost")
        sample = df.sample(min(5000, len(df)), random_state=42)
        fig3 = px.scatter(
            sample,
            x           = "anomaly_score",
            y           = "failure_proba",
            color       = "risk_score" if "risk_score" in sample.columns else None,
            color_continuous_scale = "RdYlGn_r",
            opacity     = 0.4,
            labels      = dict(anomaly_score="Score IF",
                               failure_proba="Proba XGBoost",
                               risk_score="Score risque"),
        )
        fig3.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=350,
            margin=dict(t=20, b=40, l=20, r=20),
            xaxis=dict(gridcolor="#1e293b"),
            yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Header
    st.markdown(
        """<h1 style='color:#f1f5f9;font-size:2rem;margin-bottom:0'>
        🔮 AuraLog <span style='color:#6366f1'>Dashboard</span></h1>
        <p style='color:#64748b;margin-top:4px'>
        Surveillance prédictive des logs infrastructure · ML · Syslog</p>""",
        unsafe_allow_html=True,
    )

    # Chargement des données
    df_raw = load_latest_predictions()

    if df_raw.empty:
        st.error(
            "⚠️ Aucune prédiction disponible. "
            "Lancez d'abord : `python3 main.py --mode predict`"
        )
        st.code("python3 main.py --mode predict --no-wandb", language="bash")
        return

    # Info barre
    if "timestamp" in df_raw.columns:
        min_ts = df_raw["timestamp"].min()
        max_ts = df_raw["timestamp"].max()
        st.caption(
            f"📅 Données : {min_ts.strftime('%Y-%m-%d %H:%M')} → "
            f"{max_ts.strftime('%Y-%m-%d %H:%M')} UTC  |  "
            f"**{len(df_raw):,}** logs  |  "
            f"Dernière mise à jour : {datetime.utcnow().strftime('%H:%M')} UTC"
        )

    # Sidebar avec filtres
    df = render_sidebar(df_raw)

    # Onglets
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "📊 Vue d'ensemble",
        "🚨 Alertes",
        "🌡️ Heatmap",
        "⚙️ Services",
        "🔮 Incidents & Prédictions",
        "🤖 Modèles ML",
    ])

    with t1: tab_overview(df)
    with t2: tab_alerts(df)
    with t3: tab_heatmap(df)
    with t4: tab_services(df)
    with t5: tab_incidents(df)
    with t6: tab_models(df)


if __name__ == "__main__":
    main()