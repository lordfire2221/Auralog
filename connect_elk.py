"""
AuraLog — connect_elk.py
Gestionnaire de connexion à Elasticsearch 8.x.

Stack cible :
  - Docker Swarm + Traefik reverse proxy
  - TLS Let's Encrypt sur https://elastic.fares.top
  - xpack.security.enabled=false  →  aucune auth requise
  - Elasticsearch accessible en HTTP en interne,
    mais exposé en HTTPS publiquement via Traefik

Supporte aussi :
  - Authentification API key ou user/password (si activée)
  - Mode dev sans vérification SSL (ES_VERIFY_SSL=false)
  - Context manager (with ELKConnector() as elk: ...)
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import (
    ConnectionError as ESConnectionError,
    AuthenticationException,
    NotFoundError,
    TransportError,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("auralog.elk")


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASSE PRINCIPALE
# ═══════════════════════════════════════════════════════════════════════════════

class ELKConnector:
    """
    Connexion à Elasticsearch 8.x exposé via Traefik (Let's Encrypt).

    Variables d'environnement (voir .env.example) :

        ES_HOST         URL publique Traefik   ex: https://elastic.fares.top
        ES_VERIFY_SSL   true (défaut) — vérifie le cert Let's Encrypt standard

        # Optionnel — si vous réactivez xpack.security un jour :
        ES_USERNAME     Utilisateur            ex: elastic
        ES_PASSWORD     Mot de passe
        ES_API_KEY      Clé API                ex: id:valeur
        ES_CA_CERT      Chemin cert custom     (inutile avec Let's Encrypt)
    """

    def __init__(self):
        self.host       = os.getenv("ES_HOST", "https://elastic.fares.top")
        self.username   = os.getenv("ES_USERNAME", "")
        self.password   = os.getenv("ES_PASSWORD", "")
        self.api_key    = os.getenv("ES_API_KEY", "")
        self.ca_cert    = os.getenv("ES_CA_CERT", "")
        self.verify_ssl = os.getenv("ES_VERIFY_SSL", "true").lower() == "true"
        self._client: Optional[Elasticsearch] = None

    # ── Connexion ─────────────────────────────────────────────────────────────

    def connect(self) -> "ELKConnector":
        """Établit la connexion et valide qu'Elasticsearch répond."""
        try:
            kwargs = self._build_kwargs()
            self._client = Elasticsearch(**kwargs)
            self._client.info()  # ping réel
            logger.info("✅  Connecté à Elasticsearch — %s", self.host)
            return self
        except AuthenticationException:
            logger.error(
                "❌  Authentification refusée. "
                "Vérifiez ES_USERNAME / ES_PASSWORD ou ES_API_KEY dans votre .env"
            )
            raise
        except ESConnectionError as exc:
            logger.error("❌  Impossible de joindre %s : %s", self.host, exc)
            raise
        except Exception as exc:
            logger.error("❌  Erreur inattendue lors de la connexion : %s", exc)
            raise

    def _build_kwargs(self) -> dict:
        """
        Construit les paramètres du client Elasticsearch.

        Cas courant (votre stack) :
          - HTTPS via Traefik Let's Encrypt → verify_certs=True, pas de CA custom
          - xpack.security.enabled=false    → pas d'auth
        """
        kwargs: dict = {"hosts": [self.host]}

        # ── Authentification (optionnelle — sécurité désactivée dans votre stack)
        if self.api_key:
            try:
                key_id, key_value = self.api_key.split(":", 1)
                kwargs["api_key"] = (key_id, key_value)
                logger.debug("Auth : API key")
            except ValueError:
                raise ValueError("ES_API_KEY doit être au format 'id:valeur'")

        elif self.username and self.password:
            kwargs["basic_auth"] = (self.username, self.password)
            logger.debug("Auth : basic (user/password)")

        else:
            # Votre cas : xpack.security.enabled=false → aucune auth
            logger.debug("Auth : aucune (xpack.security.enabled=false)")

        # ── SSL
        if self.verify_ssl:
            if self.ca_cert:
                # Cert custom (local sans Traefik)
                kwargs["ca_certs"] = self.ca_cert
                logger.debug("SSL : certificat CA custom (%s)", self.ca_cert)
            else:
                # Traefik + Let's Encrypt → CA publique reconnue nativement
                logger.debug("SSL : Let's Encrypt via Traefik (CA publique)")
        else:
            # Mode dev — désactive la vérification SSL
            kwargs["verify_certs"]  = False
            kwargs["ssl_show_warn"] = False
            logger.warning("⚠️  Vérification SSL désactivée — mode DEV uniquement")

        return kwargs

    # ── Propriété client ──────────────────────────────────────────────────────

    @property
    def client(self) -> Elasticsearch:
        """Retourne le client, le crée si nécessaire (lazy connect)."""
        if self._client is None:
            self.connect()
        return self._client

    # ── Santé du cluster ──────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """
        Retourne un rapport de santé complet :
          cluster, version, statut, nœuds, shards, indices de logs.
        """
        info   = self.client.info()
        health = self.client.cluster.health()
        raw_indices = self.client.cat.indices(
            format="json",
            h="index,health,status,docs.count,store.size",
        )

        # Filtre les indices système (commençant par ".")
        indices = [
            {
                "name":   i["index"],
                "health": i["health"],
                "status": i["status"],
                "docs":   i.get("docs.count", "0"),
                "size":   i.get("store.size", "0b"),
            }
            for i in raw_indices
            if not i["index"].startswith(".")
        ]

        report = {
            "cluster_name":  info["cluster_name"],
            "version":       info["version"]["number"],
            "status":        health["status"],   # green / yellow / red
            "nodes":         health["number_of_nodes"],
            "active_shards": health["active_shards"],
            "indices":       indices,
        }

        status_icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}.get(
            health["status"], "⚪"
        )
        logger.info(
            "%s  Cluster : %s | Version : %s | Nœuds : %s | Shards actifs : %s",
            status_icon,
            report["cluster_name"],
            report["version"],
            report["nodes"],
            report["active_shards"],
        )
        return report

    # ── Gestion des indices ───────────────────────────────────────────────────

    def list_log_indices(self, pattern: str = "logs-*") -> list[str]:
        """Liste les indices correspondant au pattern (défaut : logs-*)."""
        try:
            raw = self.client.cat.indices(index=pattern, format="json", h="index")
            indices = sorted([i["index"] for i in raw])
            logger.info("📂  %d indice(s) trouvé(s) pour '%s'", len(indices), pattern)
            return indices
        except NotFoundError:
            logger.warning("Aucun indice correspondant à '%s'", pattern)
            return []

    def index_exists(self, index: str) -> bool:
        """Vérifie si un indice existe."""
        return bool(self.client.indices.exists(index=index).body)

    def get_index_mapping(self, index: str) -> dict:
        """Retourne le mapping d'un indice (champs disponibles)."""
        return self.client.indices.get_mapping(index=index).body

    def count_documents(self, index: str, query: dict | None = None) -> int:
        """Compte les documents dans un indice avec un filtre optionnel."""
        body = {"query": query} if query else {}
        result = self.client.count(index=index, body=body)
        return result["count"]

    # ── Fermeture ─────────────────────────────────────────────────────────────

    def close(self):
        """Ferme la connexion proprement."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("🔌  Connexion Elasticsearch fermée.")

    # ── Context manager ───────────────────────────────────────────────────────

    def __enter__(self) -> "ELKConnector":
        return self.connect()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False   # propage les exceptions

    def __repr__(self) -> str:
        status = "connecté" if self._client else "déconnecté"
        return f"<ELKConnector host={self.host!r} [{status}]>"


# ═══════════════════════════════════════════════════════════════════════════════
#  POINT D'ENTRÉE — TEST RAPIDE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("  AuraLog — Test de connexion Elasticsearch")
    print("=" * 60)

    with ELKConnector() as elk:
        # 1 — Santé du cluster
        report = elk.health_check()
        print("\n📊  Rapport de santé :")
        print(json.dumps(report, indent=2, ensure_ascii=False))

        # 2 — Indices de logs disponibles
        indices = elk.list_log_indices("logs-*")
        if indices:
            print(f"\n📂  Indices de logs ({len(indices)}) :")
            for idx in indices:
                print(f"    — {idx}")
        else:
            print("\n⚠️  Aucun indice 'logs-*' trouvé.")
            print("    → Vérifiez que vos logs sont bien ingérés dans ELK.")

    print("\n✅  Test terminé.")
