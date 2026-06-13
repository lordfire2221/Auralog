"""
AuraLog — weight_biases.py
Gestionnaire W&B pour le suivi MLOps des modèles prédictifs.

Note : le fichier s'appelle weight_biases.py (sans &) pour
       compatibilité avec les imports Python.

Usage :
    from weight_biases import WandBManager
    mgr = WandBManager()
    run = mgr.start_run("mon_run", model_type="xgboost")
"""

import os
import logging
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any

import wandb
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("auralog.wandb")

WANDB_PROJECT = os.getenv("WANDB_PROJECT", "auralog")
WANDB_ENTITY  = os.getenv("WANDB_ENTITY", None)
MODEL_DIR     = Path(os.getenv("AURALOG_MODEL_DIR", "./models"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class WandBManager:
    """
    Gestionnaire centralisé W&B pour AuraLog.

    Regroupe :
      - Création / gestion des runs
      - Logging métriques, configs, tables
      - Sauvegarde et versioning des modèles
    """

    def __init__(self, project: str = WANDB_PROJECT, entity: str | None = WANDB_ENTITY):
        self.project  = project
        self.entity   = entity
        self._run     = None

        api_key = os.getenv("WANDB_API_KEY", "")
        if api_key:
            try:
                wandb.login(key=api_key, relogin=False)
                logger.info("✅  W&B connecté (projet: %s)", project)
            except Exception as e:
                logger.warning("⚠️  W&B login échoué : %s", e)
        else:
            logger.warning(
                "⚠️  WANDB_API_KEY absent — utilisation du mode offline.\n"
                "    Obtenez votre clé sur https://wandb.ai/authorize"
            )

    # ── Gestion du run ────────────────────────────────────────────────────────

    def start_run(
        self,
        run_name:   str | None = None,
        model_type: str        = "unknown",
        config:     dict | None = None,
        tags:       list[str] | None = None,
        notes:      str | None = None,
    ) -> wandb.run:
        default_config = {
            "project":    self.project,
            "model_type": model_type,
            "started_at": datetime.utcnow().isoformat(),
        }
        self._run = wandb.init(
            project = self.project,
            entity  = self.entity,
            name    = run_name or f"{model_type}_{datetime.utcnow():%Y%m%d_%H%M%S}",
            config  = {**default_config, **(config or {})},
            tags    = tags or [model_type, "auralog"],
            notes   = notes,
            reinit  = True,
        )
        logger.info("🚀  Run W&B : %s (id: %s)", self._run.name, self._run.id)
        return self._run

    def finish_run(self, exit_code: int = 0):
        if self._run:
            wandb.finish(exit_code=exit_code)
            logger.info("✅  Run W&B terminé.")
            self._run = None

    @property
    def run(self) -> wandb.run:
        if self._run is None:
            raise RuntimeError("Aucun run actif. Appelez start_run() d'abord.")
        return self._run

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, metrics: dict, step: int | None = None):
        """Log des métriques numériques."""
        self.run.log(metrics, step=step)

    def log_config(self, config: dict):
        """Met à jour la config du run."""
        self.run.config.update(config)

    def log_table(self, name: str, columns: list, data: list):
        """Log une table W&B."""
        self.run.log({name: wandb.Table(columns=columns, data=data)})

    # ── Sauvegarde modèles ────────────────────────────────────────────────────

    def save_model(self, model: Any, name: str, metadata: dict | None = None) -> Path:
        ts   = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = MODEL_DIR / f"{name}_{ts}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        artifact = wandb.Artifact(
            name     = name,
            type     = "model",
            metadata = {"saved_at": ts, "run_id": self.run.id, **(metadata or {})},
        )
        artifact.add_file(str(path))
        self.run.log_artifact(artifact)
        logger.info("💾  Modèle '%s' sauvegardé et versé dans W&B.", name)
        return path

    def load_model(self, name: str, version: str = "latest") -> Any:
        ref      = f"{self.entity or ''}/{self.project}/{name}:{version}".lstrip("/")
        artifact = self.run.use_artifact(ref, type="model")
        local    = artifact.download(root=str(MODEL_DIR / "downloaded"))
        pkls     = list(Path(local).glob("*.pkl"))
        if not pkls:
            raise FileNotFoundError(f"Aucun .pkl dans {local}")
        with open(max(pkls, key=lambda p: p.stat().st_mtime), "rb") as f:
            return pickle.load(f)

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.finish_run(exit_code=1 if exc_type else 0)
        return False


# ── Test standalone ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import IsolationForest

    print("=" * 55)
    print("  AuraLog — Test W&B")
    print("=" * 55)

    mgr = WandBManager()
    run = mgr.start_run(
        run_name   = "test_connexion_wandb",
        model_type = "isolation_forest",
        config     = {"contamination": 0.01, "n_estimators": 10, "test": True},
        tags       = ["test"],
        notes      = "Test de connexion W&B depuis AuraLog",
    )

    # Données fictives
    X = np.random.randn(200, 6)

    # Entraîne un mini modèle
    model = IsolationForest(contamination=0.01, n_estimators=10, random_state=42)
    model.fit(X)
    scores = model.predict(X)
    n_anom = int((scores == -1).sum())

    # Log quelques métriques
    mgr.log({"iso/n_anomalies": n_anom, "iso/anomaly_rate": n_anom / len(X)})
    mgr.log_table(
        "iso/sample_scores",
        columns=["index", "score", "label"],
        data=[[i, float(model.decision_function([X[i]])[0]), int(scores[i])]
              for i in range(5)]
    )

    path = mgr.save_model(model, "isolation_forest_test", {"test": True})
    print(f"\n  Modèle sauvegardé : {path}")
    print(f"  Anomalies test    : {n_anom}")

    mgr.finish_run()
    print("\n✅  Test W&B terminé.")
    print("   → Vérifiez votre dashboard : https://wandb.ai")
