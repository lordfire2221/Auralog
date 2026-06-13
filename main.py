"""
AuraLog — main.py
Point d'entrée unique du pipeline complet.

Modes :
  full    → extraction + entraînement + prédiction + alertes  (défaut)
  train   → ré-entraîne les modèles uniquement
  predict → prédit avec les modèles existants (pas de ré-entraînement)
  health  → vérifie ELK + état des modèles sauvegardés

Usage :
  python3 main.py                        # pipeline complet
  python3 main.py --mode predict         # prédiction seule
  python3 main.py --mode train           # ré-entraînement seul
  python3 main.py --mode health          # diagnostic
  python3 main.py --mode full --days 60  # fenêtre étendue
"""

import argparse
import logging
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from connect_elk import ELKConnector
from extract_data import LogExtractor
from train_models import (
    FEATURE_COLS,
    PREDICTION_HORIZON_MIN,
    run_training,
)

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, os.getenv("AURALOG_LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("auralog.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("auralog.main")

# ── Constantes ────────────────────────────────────────────────────────────────
MODEL_DIR     = Path(os.getenv("AURALOG_MODEL_DIR", "./models"))
LOOKBACK_DAYS = int(os.getenv("AURALOG_LOOKBACK_DAYS", "30"))
RISK_CRITICAL = float(os.getenv("AURALOG_RISK_CRITICAL", "75"))
RISK_HIGH     = float(os.getenv("AURALOG_RISK_HIGH", "50"))
OUTPUT_DIR    = Path("./predictions")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CHARGEMENT DES MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════

def load_latest_model(name: str):
    """
    Charge le modèle le plus récent correspondant au nom donné.
    Ignore les modèles de test (contenant '_test_' dans le nom).
    Trie par date de modification (le plus récent en premier).
    """
    candidates = [
        p for p in MODEL_DIR.glob(f"{name}_*.pkl")
        if "_test_" not in p.name
    ]
    if not candidates:
        return None
    # Tri par date de modification — le plus récent en premier
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    with open(candidates[0], "rb") as f:
        model = pickle.load(f)
    logger.info("📦  Modèle chargé : %s", candidates[0].name)
    return model


def load_models() -> dict:
    """Charge Isolation Forest et XGBoost depuis ./models/."""
    models = {
        "isolation_forest": load_latest_model("isolation_forest"),
        "xgboost":          load_latest_model("xgboost"),
    }
    missing = [k for k, v in models.items() if v is None]
    if missing:
        logger.warning("⚠️  Modèles manquants : %s", missing)
    return models


# ═══════════════════════════════════════════════════════════════════════════════
#  PRÉDICTION ET SCORE DE RISQUE
# ═══════════════════════════════════════════════════════════════════════════════

class AuraLogPredictor:
    """
    Orchestre les prédictions combinées IF + XGBoost.

    Score de risque composite (0–100) :
      40% score d'anomalie Isolation Forest
      60% probabilité de panne XGBoost
    """

    def __init__(self, models: dict):
        self.iso = models.get("isolation_forest")
        self.xgb = models.get("xgboost")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcule le score de risque pour chaque ligne du DataFrame.

        Colonnes ajoutées :
          anomaly_score   Score IF normalisé (0–1)
          anomaly_flag    1 si détecté comme anomalie par IF
          failure_proba   Probabilité de panne prédite par XGBoost
          risk_score      Score composite (0–100)
          risk_level      🟢 Normal / 🟡 Modéré / 🟠 Élevé / 🔴 Critique
        """
        if df.empty:
            return df

        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            logger.warning("Features manquantes pour la prédiction : %s", missing)
            for col in missing:
                df[col] = 0

        X = df[FEATURE_COLS].fillna(0).values
        df = df.copy()

        # ── Isolation Forest
        if self.iso is not None:
            raw          = self.iso.decision_function(X)
            min_, max_   = raw.min(), raw.max()
            df["anomaly_score"] = 1 - (raw - min_) / (max_ - min_ + 1e-9)
            df["anomaly_flag"]  = (self.iso.predict(X) == -1).astype(int)
        else:
            df["anomaly_score"] = 0.0
            df["anomaly_flag"]  = 0

        # ── XGBoost
        if self.xgb is not None:
            proba               = self.xgb.predict_proba(X)[:, 1]
            df["failure_proba"] = proba.round(4)
        else:
            df["failure_proba"] = 0.0

        # ── Score de risque composite
        risk = (df["anomaly_score"] * 0.40 + df["failure_proba"] * 0.60) * 100
        df["risk_score"] = np.clip(risk, 0, 100).round(1)
        df["risk_level"] = pd.cut(
            df["risk_score"],
            bins   = [-0.1, 25, 50, 75, 100.1],
            labels = ["🟢 Normal", "🟡 Modéré", "🟠 Élevé", "🔴 Critique"],
        )
        return df

    def top_alerts(self, df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Retourne les N alertes les plus critiques."""
        cols = [c for c in [
            "timestamp", "process", "host_agent",
            "log_level", "message", "risk_score", "risk_level",
            "anomaly_flag", "failure_proba"
        ] if c in df.columns]
        return (df[df["risk_score"] >= RISK_HIGH][cols]
                .nlargest(n, "risk_score")
                .reset_index(drop=True))

    def aggregate_by_service(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrège le score de risque par service (processus + hôte)."""
        if "service" not in df.columns:
            return pd.DataFrame()
        return (df.groupby("service")
                  .agg(
                      n_logs       = ("risk_score", "count"),
                      risk_max     = ("risk_score", "max"),
                      risk_mean    = ("risk_score", "mean"),
                      n_anomalies  = ("anomaly_flag", "sum"),
                      n_critical   = ("risk_score", lambda x: (x >= RISK_CRITICAL).sum()),
                  )
                  .sort_values("risk_max", ascending=False)
                  .reset_index())

    def aggregate_by_hour(self, df: pd.DataFrame) -> pd.DataFrame:
        """Agrège le score de risque par heure (pour heatmap)."""
        if "timestamp" not in df.columns:
            return pd.DataFrame()
        df = df.copy()
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
        return (df.groupby("hour")
                  .agg(
                      n_logs      = ("risk_score", "count"),
                      risk_mean   = ("risk_score", "mean"),
                      n_anomalies = ("anomaly_flag", "sum"),
                  )
                  .reset_index())


# ═══════════════════════════════════════════════════════════════════════════════
#  AFFICHAGE DES RÉSULTATS
# ═══════════════════════════════════════════════════════════════════════════════

def print_banner():
    print("╔" + "═" * 58 + "╗")
    print("║        AuraLog — Système de prédiction d'erreurs         ║")
    print("║        Infrastructure · Syslog · ML · W&B                ║")
    print("╚" + "═" * 58 + "╝")
    print(f"  {datetime.utcnow():%Y-%m-%d %H:%M} UTC\n")


def print_risk_summary(df: pd.DataFrame):
    """Affiche la distribution des niveaux de risque."""
    if "risk_level" not in df.columns:
        return
    counts = df["risk_level"].value_counts()
    total  = len(df)

    print("\n┌─────────────────────────────────────────────┐")
    print("│            Résumé des niveaux de risque      │")
    print("├──────────────────┬───────────┬───────────────┤")
    print(f"│ {'Niveau':<16} │ {'Logs':>9} │ {'%':>13} │")
    print("├──────────────────┼───────────┼───────────────┤")
    for level in ["🔴 Critique", "🟠 Élevé", "🟡 Modéré", "🟢 Normal"]:
        count = counts.get(level, 0)
        pct   = count / total * 100
        bar   = "█" * min(int(pct / 5), 10)
        print(f"│ {str(level):<16} │ {count:>9,} │ {pct:>6.1f}% {bar:<5} │")
    print("└──────────────────┴───────────┴───────────────┘")


def print_top_alerts(alerts: pd.DataFrame):
    """Affiche les alertes les plus critiques."""
    if alerts.empty:
        print("\n✅  Aucune alerte significative détectée.")
        return

    print(f"\n🚨  Top alertes ({len(alerts)}) :\n")
    for _, row in alerts.iterrows():
        ts      = str(row.get("timestamp", ""))[:19]
        process = str(row.get("process", "?"))[:15]
        host    = str(row.get("host_agent", "?"))[:12]
        score   = row.get("risk_score", 0)
        level   = str(row.get("risk_level", ""))
        msg     = str(row.get("message", ""))[:60]
        print(f"  {level} [{score:5.1f}] {ts} | {host}/{process}")
        print(f"           └─ {msg}")


def print_service_risks(agg: pd.DataFrame, top: int = 8):
    """Affiche les services les plus à risque."""
    if agg.empty:
        return
    print(f"\n📊  Services les plus à risque (top {top}) :\n")
    print(f"  {'Service':<35} {'Logs':>6} {'RiskMax':>8} {'Anomalies':>10} {'Critiques':>10}")
    print("  " + "─" * 75)
    for _, row in agg.head(top).iterrows():
        svc  = str(row["service"])[:34]
        bar  = "█" * min(int(row["risk_max"] / 10), 10)
        print(f"  {svc:<35} {int(row['n_logs']):>6,} {row['risk_max']:>7.1f}  "
              f"{int(row['n_anomalies']):>9} {int(row['n_critical']):>10}  {bar}")


def print_hourly_heatmap(hourly: pd.DataFrame):
    """Affiche une heatmap ASCII du risque par heure."""
    if hourly.empty:
        return
    max_risk = hourly["risk_mean"].max() or 1
    print("\n⏰  Heatmap risque par heure :\n")
    for _, row in hourly.iterrows():
        h      = int(row["hour"])
        risk   = row["risk_mean"]
        filled = int(risk / max_risk * 20)
        bar    = "█" * filled + "░" * (20 - filled)
        anom   = int(row["n_anomalies"])
        print(f"  {h:02d}h  {bar}  {risk:5.1f}  ({anom} anomalies)")


# ═══════════════════════════════════════════════════════════════════════════════
#  MODES DU PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def mode_health(args):
    """Vérifie ELK et l'état des modèles sauvegardés."""
    print("─" * 50)
    print("  Mode : HEALTH CHECK")
    print("─" * 50)

    # ELK
    with ELKConnector() as elk:
        health = elk.health_check()
        status = {"green": "✅", "yellow": "⚠️ ", "red": "❌"}.get(
            health["status"], "?"
        )
        print(f"\n{status}  Elasticsearch : {health['cluster_name']} "
              f"v{health['version']} | {health['nodes']} nœud(s) | "
              f"statut {health['status'].upper()}")
        print(f"     Shards actifs : {health['active_shards']}")
        if health["indices"]:
            print(f"     Indices logs : {len(health['indices'])}")
            for idx in health["indices"][:5]:
                print(f"       — {idx['name']} ({idx['docs']} docs, {idx['size']})")

    # Modèles
    print("\n📦  Modèles disponibles :")
    found = False
    for pkl in sorted(MODEL_DIR.glob("*.pkl"), reverse=True)[:6]:
        size = pkl.stat().st_size / 1024
        print(f"  ✅  {pkl.name:<45} {size:>8.1f} KB")
        found = True
    if not found:
        print("  ⚠️   Aucun modèle trouvé dans ./models/")
        print("       → Lancez : python3 main.py --mode train")

    print("\n✅  Health check terminé.")


def mode_train(args):
    """Entraîne ou ré-entraîne les modèles."""
    print("─" * 50)
    print("  Mode : TRAIN")
    print("─" * 50)
    run_training(args)


def mode_predict(args):
    """Charge les modèles et prédit sur les données récentes."""
    print("─" * 50)
    print("  Mode : PREDICT")
    print("─" * 50)

    models = load_models()
    if all(v is None for v in models.values()):
        logger.error("❌  Aucun modèle disponible. Lancez --mode train d'abord.")
        sys.exit(1)

    predictor = AuraLogPredictor(models)

    with ELKConnector() as elk:
        extractor = LogExtractor(elk)
        if args.index:
            extractor.index = args.index

        # ── Agrégations (scalable à millions de logs)
        logger.info("📊  Extraction via agrégations ELK (scalable)...")
        df_raw = extractor.fetch_aggregated(days=args.days)

    if df_raw.empty:
        logger.error("❌  Aucune donnée extraite.")
        sys.exit(1)

    df_feat = extractor.build_features(df_raw)

    logger.info("🔮  Calcul des scores de risque…")
    df_pred = predictor.predict(df_feat)

    print_risk_summary(df_pred)
    print_top_alerts(predictor.top_alerts(df_pred, n=10))
    print_service_risks(predictor.aggregate_by_service(df_pred))
    print_hourly_heatmap(predictor.aggregate_by_hour(df_pred))

    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"predictions_{ts}.csv"
    df_pred.to_csv(out_path, index=False)
    logger.info("💾  Prédictions sauvegardées : %s", out_path)

    return df_pred


def mode_alert(args):
    """Vérifie les incidents/prédictions et envoie des alertes Discord."""
    print("─" * 50)
    print("  Mode : ALERT")
    print("─" * 50)
    from alert_report import run_analysis, detect_incidents, detect_predicted_incidents, alert_on_incidents

    df_pred, incidents, predictions = run_analysis(args.days)
    print(f"\n📊  {len(df_pred):,} buckets analysés")
    print(f"🚨  {len(incidents)} incident(s) détecté(s)")
    print(f"🔮  {len(predictions)} prédiction(s) préventive(s) (+{os.getenv('AURALOG_PRED_HORIZON_MIN','30')}min)")
    sent = alert_on_incidents(incidents, predictions)
    print(f"📨  {sent} alerte(s) envoyée(s)")


def mode_report(args):
    """Génère un rapport Markdown (résumé, incidents, prédictions)."""
    print("─" * 50)
    print("  Mode : REPORT")
    print("─" * 50)
    from alert_report import run_analysis, detect_incidents, detect_predicted_incidents, generate_report, REPORT_DIR

    df_pred, incidents, predictions = run_analysis(args.days)
    report   = generate_report(df_pred, incidents, predictions, args.days)
    ts       = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = REPORT_DIR / f"report_{ts}.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n💾  Rapport sauvegardé : {out_path}")
    print("\n" + report)


def mode_full(args):
    """Pipeline complet : train + predict + alert."""
    print("─" * 50)
    print("  Mode : FULL PIPELINE")
    print("─" * 50)
    mode_train(args)
    mode_predict(args)

    # ── Alerting automatique en fin de pipeline
    try:
        from alert_report import run_analysis, alert_on_incidents
        print("\n" + "─" * 50)
        print("  Sous-étape : ALERTING")
        print("─" * 50)
        df_pred, incidents, predictions = run_analysis(min(args.days, 1))
        sent = alert_on_incidents(incidents, predictions)
        print(f"📨  {sent} alerte(s) envoyée(s)")
    except Exception as e:
        logger.warning("⚠️  Alerting ignoré : %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="AuraLog — Pipeline de prédiction d'erreurs infrastructure",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["full", "train", "predict", "health", "alert", "report"],
        default="full",
        help=(
            "full    : extraction + entraînement + prédiction (défaut)\n"
            "train   : ré-entraîne les modèles\n"
            "predict : prédit avec les modèles existants\n"
            "health  : vérifie ELK et les modèles"
        ),
    )
    parser.add_argument("--days",     type=int, default=LOOKBACK_DAYS,
                        help=f"Fenêtre d'extraction en jours (défaut: {LOOKBACK_DAYS})")
    parser.add_argument("--index",    type=str, default=None,
                        help="Pattern d'index ELK (ex: logs-allsystem-system)")
    parser.add_argument("--model",    choices=["iso", "xgb", "both"],
                        default="both", help="Modèle à entraîner (défaut: both)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Désactive le tracking W&B")
    return parser.parse_args()


def main():
    print_banner()
    args = parse_args()

    logger.info("Mode : %s | Fenêtre : %dj | W&B : %s",
                args.mode, args.days, "off" if args.no_wandb else "on")

    dispatch = {
        "health":  mode_health,
        "train":   mode_train,
        "predict": mode_predict,
        "full":    mode_full,
        "alert":   mode_alert,
        "report":  mode_report,
    }
    dispatch[args.mode](args)
    print(f"\n{'═' * 60}")
    print(f"  ✅  AuraLog terminé — {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()