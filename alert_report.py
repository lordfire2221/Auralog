"""
AuraLog — alert_report.py
Alerting (Discord) + Génération de rapport (Markdown/texte).

Fonctions :
  - Détecte les incidents (clusters de buckets à risque consécutifs)
  - Envoie des alertes Discord pour les incidents critiques
  - Génère un rapport (résumé période, top incidents, services à risque)
  - Génère un rapport hebdomadaire complet

Usage :
  python3 alert_report.py --mode alert    # vérifie et alerte (à scheduler)
  python3 alert_report.py --mode report   # génère le rapport texte/markdown
  python3 alert_report.py --mode weekly    # rapport hebdomadaire complet
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from connect_elk import ELKConnector
from extract_data import LogExtractor
from main import AuraLogPredictor, load_models, RISK_CRITICAL, RISK_HIGH

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("auralog.alert")

# ── Constantes ────────────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
REPORT_DIR          = Path("./reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTION_HORIZON_MIN = int(os.getenv("AURALOG_PRED_HORIZON_MIN", "30"))
ALERT_STATE_FILE       = Path("./predictions/.last_alert_state.json")


# ═══════════════════════════════════════════════════════════════════════════════
#  DÉTECTION D'INCIDENTS (clusters de buckets à risque)
# ═══════════════════════════════════════════════════════════════════════════════

def detect_incidents(df: pd.DataFrame, gap_minutes: int = 5) -> list[dict]:
    """
    Regroupe les buckets à risque élevé/critique en incidents.
    Un incident = série de buckets consécutifs (écart < gap_minutes)
    pour un même service, avec risk_score >= RISK_HIGH.

    Returns:
        Liste de dicts décrivant chaque incident :
          service, start, end, duration_min, max_risk, n_buckets, predicted
    """
    if df.empty or "risk_score" not in df.columns:
        return []

    flagged = df[df["risk_score"] >= RISK_HIGH].copy()
    if flagged.empty:
        return []

    flagged = flagged.sort_values(["service", "timestamp"])
    incidents = []

    for service, grp in flagged.groupby("service"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        grp["gap"] = grp["timestamp"].diff().dt.total_seconds().div(60).fillna(0)
        grp["cluster"] = (grp["gap"] > gap_minutes).cumsum()

        for _, cl in grp.groupby("cluster"):
            incidents.append({
                "service":       service,
                "start":         cl["timestamp"].min(),
                "end":           cl["timestamp"].max(),
                "duration_min":  round((cl["timestamp"].max() - cl["timestamp"].min())
                                        .total_seconds() / 60, 1) + 1,
                "max_risk":      cl["risk_score"].max(),
                "n_buckets":     len(cl),
                "is_critical":   bool((cl["risk_score"] >= RISK_CRITICAL).any()),
                "total_errors":  int(cl["error_count"].sum()) if "error_count" in cl.columns else 0,
                "total_logs":    int(cl["total_logs"].sum()) if "total_logs" in cl.columns else 0,
                "max_proba":     float(cl["failure_proba"].max()) if "failure_proba" in cl.columns else 0,
            })

    return sorted(incidents, key=lambda x: x["max_risk"], reverse=True)


def detect_predicted_incidents(df: pd.DataFrame, threshold: float = 0.5) -> list[dict]:
    """
    Détecte les buckets RÉCENTS où XGBoost prédit une erreur dans les
    PREDICTION_HORIZON_MIN prochaines minutes — alertes "préventives".

    Ne regarde que les buckets des dernières PREDICTION_HORIZON_MIN minutes
    (pour ne pas re-alerter sur du passé).
    """
    if df.empty or "failure_proba" not in df.columns:
        return []

    now      = df["timestamp"].max()
    cutoff   = now - pd.Timedelta(minutes=PREDICTION_HORIZON_MIN)
    recent   = df[df["timestamp"] >= cutoff]
    flagged  = recent[recent["failure_proba"] >= threshold].copy()

    if flagged.empty:
        return []

    flagged = flagged.sort_values("failure_proba", ascending=False)
    predictions = []
    for _, row in flagged.iterrows():
        predictions.append({
            "service":      row["service"],
            "timestamp":    row["timestamp"],
            "failure_proba":round(float(row["failure_proba"]), 3),
            "current_risk": round(float(row["risk_score"]), 1),
            "error_count":  int(row.get("error_count", 0)),
            "total_logs":   int(row.get("total_logs", 0)),
        })
    return predictions


# ═══════════════════════════════════════════════════════════════════════════════
#  ÉTAT PERSISTANT (anti-spam)
# ═══════════════════════════════════════════════════════════════════════════════

def _incident_key(inc: dict) -> str:
    """Clé unique pour identifier un incident (service + heure de début)."""
    return f"{inc['service']}|{inc['start']:%Y%m%d%H%M}"


def load_alert_state() -> dict:
    """Charge l'état des dernières alertes envoyées."""
    if ALERT_STATE_FILE.exists():
        try:
            return json.loads(ALERT_STATE_FILE.read_text())
        except Exception:
            return {}
    return {}


def save_alert_state(state: dict):
    """Sauvegarde l'état, en ne gardant que les 500 dernières entrées."""
    ALERT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    if len(state) > 500:
        # Garde les plus récentes
        items = sorted(state.items(), key=lambda kv: kv[1], reverse=True)[:500]
        state = dict(items)
    ALERT_STATE_FILE.write_text(json.dumps(state))


def filter_new_incidents(incidents: list[dict], state: dict) -> list[dict]:
    """Retourne uniquement les incidents pas encore alertés."""
    now_iso = datetime.utcnow().isoformat()
    new = []
    for inc in incidents:
        key = _incident_key(inc)
        if key not in state:
            new.append(inc)
            state[key] = now_iso
    return new


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERTING DISCORD
# ═══════════════════════════════════════════════════════════════════════════════

def send_discord_alert(title: str, description: str, color: int = 0xFF0000,
                        fields: list[dict] | None = None) -> bool:
    """Envoie une alerte vers Discord via webhook."""
    if not DISCORD_WEBHOOK_URL:
        logger.warning("⚠️  DISCORD_WEBHOOK_URL non configuré — alerte non envoyée")
        logger.info("📋  [SIMULATION ALERTE]\n%s\n%s", title, description)
        return False

    embed = {
        "title":       title,
        "description": description,
        "color":       color,
        "timestamp":   datetime.utcnow().isoformat(),
        "footer":      {"text": "AuraLog — Système prédictif d'erreurs"},
    }
    if fields:
        embed["fields"] = fields

    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            json={"embeds": [embed]},
            timeout=10,
        )
        resp.raise_for_status()
        logger.info("📨  Alerte Discord envoyée : %s", title)
        return True
    except Exception as e:
        logger.error("❌  Erreur envoi Discord : %s", e)
        return False


def alert_on_incidents(incidents: list[dict], predictions: list[dict]):
    """Envoie des alertes Discord pour incidents critiques et prédictions (anti-spam via état persistant)."""
    sent  = 0
    state = load_alert_state()

    # ── Incidents critiques en cours — uniquement les NOUVEAUX
    critical_incidents = [i for i in incidents if i["is_critical"]]
    new_critical = filter_new_incidents(critical_incidents, state)

    for inc in new_critical[:5]:  # max 5 pour éviter le spam
        title = f"🔴 Incident critique — {inc['service']}"
        desc  = (
            f"**Risque max :** {inc['max_risk']:.1f}/100\n"
            f"**Durée :** {inc['duration_min']:.0f} min\n"
            f"**Période :** {inc['start']:%H:%M} → {inc['end']:%H:%M}"
        )
        fields = [
            {"name": "Erreurs",  "value": f"{inc['total_errors']:,}", "inline": True},
            {"name": "Logs",     "value": f"{inc['total_logs']:,}",   "inline": True},
            {"name": "Buckets",  "value": str(inc["n_buckets"]),      "inline": True},
        ]
        if send_discord_alert(title, desc, color=0xEF4444, fields=fields):
            sent += 1

    # ── Prédictions préventives (+N min) — toujours envoyées (signal temps réel)
    if predictions:
        top_preds = predictions[:5]
        desc_lines = []
        for p in top_preds:
            desc_lines.append(
                f"**{p['service']}** — proba: {p['failure_proba']:.0%} "
                f"(à {p['timestamp']:%H:%M}, risque actuel: {p['current_risk']:.0f})"
            )
        title = f"🟠 Prédiction préventive — +{PREDICTION_HORIZON_MIN}min"
        desc  = (
            f"⚠️ {len(predictions)} bucket(s) avec forte probabilité d'erreur "
            f"dans les {PREDICTION_HORIZON_MIN} prochaines minutes :\n\n"
            + "\n".join(desc_lines)
        )
        if send_discord_alert(title, desc, color=0xF97316):
            sent += 1

    # ── Sauvegarde l'état mis à jour
    save_alert_state(state)

    if sent == 0:
        if critical_incidents and not new_critical:
            logger.info("✅  %d incident(s) critique(s) déjà notifié(s) — pas de nouvelle alerte.",
                        len(critical_incidents))
        else:
            logger.info("✅  Aucune alerte à envoyer — situation normale.")

    return sent


# ═══════════════════════════════════════════════════════════════════════════════
#  GÉNÉRATION DE RAPPORT
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(df: pd.DataFrame, incidents: list[dict],
                     predictions: list[dict], days: int) -> str:
    """Génère un rapport Markdown complet."""
    now = datetime.utcnow()

    total       = len(df)
    n_critical  = int((df["risk_score"] >= RISK_CRITICAL).sum())
    n_high      = int(((df["risk_score"] >= RISK_HIGH) & (df["risk_score"] < RISK_CRITICAL)).sum())
    avg_risk    = round(df["risk_score"].mean(), 1)
    total_errors= int(df["error_count"].sum()) if "error_count" in df.columns else 0
    total_logs  = int(df["total_logs"].sum())  if "total_logs"  in df.columns else 0

    lines = []
    lines.append(f"# 🔮 Rapport AuraLog — {now:%Y-%m-%d %H:%M} UTC")
    lines.append("")
    lines.append(f"**Période analysée :** {df['timestamp'].min():%Y-%m-%d %H:%M} → "
                  f"{df['timestamp'].max():%Y-%m-%d %H:%M} UTC ({days} jours)")
    lines.append("")

    # ── Résumé exécutif
    lines.append("## 📊 Résumé exécutif")
    lines.append("")
    lines.append(f"| Métrique | Valeur |")
    lines.append(f"|---|---|")
    lines.append(f"| Buckets analysés | {total:,} |")
    lines.append(f"| Logs traités | {total_logs:,} |")
    lines.append(f"| Erreurs détectées | {total_errors:,} |")
    lines.append(f"| Score de risque moyen | {avg_risk}/100 |")
    lines.append(f"| 🔴 Buckets critiques | {n_critical} |")
    lines.append(f"| 🟠 Buckets à risque élevé | {n_high} |")
    lines.append(f"| Incidents détectés | {len(incidents)} |")
    lines.append("")

    # ── Incidents
    lines.append("## 🚨 Incidents détectés")
    lines.append("")
    if incidents:
        lines.append("| Service | Début | Durée | Risque max | Erreurs | Critique |")
        lines.append("|---|---|---|---|---|---|")
        for inc in incidents[:15]:
            crit = "🔴 Oui" if inc["is_critical"] else "🟠 Non"
            lines.append(
                f"| {inc['service']} | {inc['start']:%Y-%m-%d %H:%M} | "
                f"{inc['duration_min']:.0f} min | {inc['max_risk']:.1f} | "
                f"{inc['total_errors']:,} | {crit} |"
            )
    else:
        lines.append("✅ Aucun incident détecté sur la période.")
    lines.append("")

    # ── Prédictions préventives
    lines.append(f"## 🔮 Prédictions préventives (+{PREDICTION_HORIZON_MIN} min)")
    lines.append("")
    if predictions:
        lines.append(f"⚠️ **{len(predictions)} bucket(s) à surveiller** — "
                      f"probabilité d'erreur dans les {PREDICTION_HORIZON_MIN} prochaines minutes :")
        lines.append("")
        lines.append("| Service | Heure | Probabilité | Risque actuel |")
        lines.append("|---|---|---|---|")
        for p in predictions[:10]:
            lines.append(
                f"| {p['service']} | {p['timestamp']:%H:%M} | "
                f"{p['failure_proba']:.0%} | {p['current_risk']:.1f} |"
            )
    else:
        lines.append("✅ Aucune erreur prédite dans les prochaines "
                      f"{PREDICTION_HORIZON_MIN} minutes.")
    lines.append("")

    # ── Services les plus à risque
    lines.append("## ⚙️ Services les plus à risque")
    lines.append("")
    if "service" in df.columns:
        agg = (df.groupby("service")
                 .agg(risk_max=("risk_score", "max"),
                      risk_mean=("risk_score", "mean"),
                      n_critical=("risk_score", lambda x: (x >= RISK_CRITICAL).sum()),
                      total_errors=("error_count", "sum") if "error_count" in df.columns else ("risk_score", "count"))
                 .sort_values("risk_max", ascending=False)
                 .head(10))
        lines.append("| Service | Risque max | Risque moyen | Critiques | Erreurs totales |")
        lines.append("|---|---|---|---|---|")
        for svc, row in agg.iterrows():
            lines.append(
                f"| {svc} | {row['risk_max']:.1f} | {row['risk_mean']:.1f} | "
                f"{int(row['n_critical'])} | {int(row['total_errors']):,} |"
            )
    lines.append("")

    # ── Heatmap horaire (texte)
    lines.append("## ⏰ Distribution horaire du risque")
    lines.append("")
    df_h = df.copy()
    df_h["hour"] = df_h["timestamp"].dt.hour
    hourly = df_h.groupby("hour")["risk_score"].mean().round(1)
    lines.append("| Heure | Risque moyen |")
    lines.append("|---|---|")
    for h, val in hourly.items():
        bar = "█" * int(val / 5)
        lines.append(f"| {h:02d}h | {val} {bar} |")
    lines.append("")

    # ── Footer
    lines.append("---")
    lines.append(f"*Rapport généré automatiquement par AuraLog le {now:%Y-%m-%d à %H:%M} UTC*")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(days: int):
    """Extrait, prédit, et retourne (df_pred, incidents, predictions)."""
    models = load_models()
    if all(v is None for v in models.values()):
        logger.error("❌  Aucun modèle disponible. Lancez train-model.py d'abord.")
        sys.exit(1)

    predictor = AuraLogPredictor(models)

    with ELKConnector() as elk:
        extractor = LogExtractor(elk)
        df_raw = extractor.fetch_aggregated(days=days)

    if df_raw.empty:
        logger.error("❌  Aucune donnée extraite.")
        sys.exit(1)

    df_feat = extractor.build_features(df_raw)
    df_pred = predictor.predict(df_feat)

    incidents   = detect_incidents(df_pred)
    predictions = detect_predicted_incidents(df_pred, threshold=0.5)

    return df_pred, incidents, predictions


def main():
    parser = argparse.ArgumentParser(description="AuraLog — Alertes & Rapports")
    parser.add_argument("--mode", choices=["alert", "report", "weekly"],
                        default="alert")
    parser.add_argument("--days", type=int, default=1,
                        help="Fenêtre d'analyse en jours (défaut: 1)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Seuil de probabilité pour alertes préventives")
    args = parser.parse_args()

    days = 7 if args.mode == "weekly" else args.days

    print("=" * 60)
    print(f"  AuraLog — Mode: {args.mode} | Fenêtre: {days}j | Horizon: +{PREDICTION_HORIZON_MIN}min")
    print("=" * 60)

    df_pred, incidents, predictions = run_analysis(days)

    print(f"\n📊  {len(df_pred):,} buckets analysés")
    print(f"🚨  {len(incidents)} incident(s) détecté(s)")
    print(f"🔮  {len(predictions)} prédiction(s) préventive(s)")

    if args.mode == "alert":
        sent = alert_on_incidents(incidents, predictions)
        print(f"\n📨  {sent} alerte(s) envoyée(s)")

    elif args.mode in ("report", "weekly"):
        report = generate_report(df_pred, incidents, predictions, days)
        ts        = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        prefix    = "weekly" if args.mode == "weekly" else "report"
        out_path  = REPORT_DIR / f"{prefix}_{ts}.md"
        out_path.write_text(report, encoding="utf-8")
        print(f"\n💾  Rapport sauvegardé : {out_path}")
        print("\n" + "─" * 60)
        print(report)

        # Envoie aussi un résumé sur Discord
        if args.mode == "weekly" and DISCORD_WEBHOOK_URL:
            n_crit = sum(1 for i in incidents if i["is_critical"])
            send_discord_alert(
                title=f"📊 Rapport hebdomadaire AuraLog",
                description=(
                    f"**{len(df_pred):,}** buckets analysés sur **{days}** jours\n"
                    f"**{len(incidents)}** incidents ({n_crit} critiques)\n"
                    f"**{len(predictions)}** prédictions préventives actives"
                ),
                color=0x6366F1,
            )

    print(f"\n{'='*60}")
    print(f"  ✅  Terminé — {datetime.utcnow():%Y-%m-%d %H:%M} UTC")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()