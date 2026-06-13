#!/bin/bash
# AuraLog — Configuration des tâches planifiées (cron)
#
# Usage : bash setup_cron.sh
#
# Met en place :
#   - Alerting toutes les 10 min (incidents + prédictions +30min)
#   - Prédiction complète (mise à jour CSV pour dashboard) toutes les heures
#   - Ré-entraînement quotidien à 3h du matin
#   - Rapport hebdomadaire chaque lundi à 8h

set -e

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${APP_DIR}/.venv/bin/python3"

if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="$(which python3)"
fi

echo "📁  Dossier AuraLog : $APP_DIR"
echo "🐍  Python utilisé  : $PYTHON_BIN"
echo ""

# Construction du contenu crontab
CRON_BLOCK=$(cat << CRON
# ═══════════════ AuraLog — Tâches planifiées ═══════════════
# Alerting : incidents + prédictions préventives (+30min)
*/10 * * * * cd $APP_DIR && $PYTHON_BIN main.py --mode alert --days 1 --no-wandb >> $APP_DIR/cron_alert.log 2>&1

# Prédiction : met à jour le CSV pour le dashboard
0 * * * * cd $APP_DIR && $PYTHON_BIN main.py --mode predict --days 7 --no-wandb >> $APP_DIR/cron_predict.log 2>&1

# Ré-entraînement quotidien (3h du matin)
0 3 * * * cd $APP_DIR && $PYTHON_BIN main.py --mode train --days 30 --no-wandb >> $APP_DIR/cron_train.log 2>&1

# Rapport hebdomadaire (lundi 8h)
0 8 * * 1 cd $APP_DIR && $PYTHON_BIN main.py --mode report --days 7 --no-wandb >> $APP_DIR/cron_report.log 2>&1
# ═════════════════════════════════════════════════════════════
CRON
)

# Vérifie si déjà installé
if crontab -l 2>/dev/null | grep -q "AuraLog — Tâches planifiées"; then
    echo "⚠️  Des tâches AuraLog existent déjà dans crontab."
    echo "    Pour les remplacer : crontab -e (puis supprimez le bloc AuraLog et relancez ce script)"
    exit 1
fi

# Ajoute au crontab existant
(crontab -l 2>/dev/null; echo ""; echo "$CRON_BLOCK") | crontab -

echo "✅  Tâches cron installées :"
echo ""
crontab -l | grep -A 10 "AuraLog"
echo ""
echo "📋  Logs disponibles dans :"
echo "    $APP_DIR/cron_alert.log"
echo "    $APP_DIR/cron_predict.log"
echo "    $APP_DIR/cron_train.log"
echo "    $APP_DIR/cron_report.log"
echo ""
echo "🔍  Vérifier l'état : crontab -l"
echo "🗑️   Supprimer       : crontab -e  (puis effacer le bloc AuraLog)"