"""
AuraLog — extract_data.py  (v2 — aggregation-based)
Extraction via agrégations Elasticsearch pour tenir à l'échelle.

Au lieu d'extraire 5M+ logs individuellement, on calcule les features
directement dans ES avec date_histogram + terms aggregations.

Pipeline :
  ES aggregations  →  time-series buckets  →  feature DataFrame  →  ML
"""

import os
import re
import logging
import argparse
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from connect_elk import ELKConnector

load_dotenv()
logger = logging.getLogger("auralog.extract")

LOG_LEVEL_MAP = {
    "TRACE": 0, "DEBUG": 1, "INFO": 2,
    "WARN": 3, "WARNING": 3, "ERROR": 4, "CRITICAL": 5, "FATAL": 5,
}

SYSLOG_LEVEL_PATTERNS = [
    (re.compile(r'\b(panic|emerg|emergency|fatal)\b',   re.I), "CRITICAL"),
    (re.compile(r'\b(alert|crit|critical)\b',           re.I), "CRITICAL"),
    (re.compile(r'\b(err|error|fail|failed|failure)\b', re.I), "ERROR"),
    (re.compile(r'\b(warn|warning)\b',                  re.I), "WARNING"),
    (re.compile(r'\b(debug)\b',                         re.I), "DEBUG"),
]
SYSLOG_PROCESS_RE = re.compile(
    r'^\S+\s+\S+\s+([a-zA-Z0-9_\-\.]+)(?:\[\d+\])?:'
)

SCROLL_PAGE_SIZE   = int(os.getenv("ES_SCROLL_PAGE_SIZE",    "1000"))
LOOKBACK_DAYS      = int(os.getenv("AURALOG_LOOKBACK_DAYS",  "7"))
AGG_INTERVAL       = os.getenv("AURALOG_AGG_INTERVAL",       "1m")   # bucket de 1 minute
MAX_SAMPLE_DOCS    = int(os.getenv("AURALOG_MAX_SAMPLE_DOCS","50000"))


def parse_syslog_level(message: str) -> str:
    if not message or not isinstance(message, str):
        return "INFO"
    for pattern, level in SYSLOG_LEVEL_PATTERNS:
        if pattern.search(message):
            return level
    return "INFO"


def parse_syslog_process(message: str) -> str:
    if not message or not isinstance(message, str):
        return "unknown"
    m = SYSLOG_PROCESS_RE.match(message.strip())
    return m.group(1) if m else "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  EXTRACTEUR v2 — AGGREGATION-BASED
# ═══════════════════════════════════════════════════════════════════════════════

class LogExtractor:
    """
    Extrait les features depuis ELK via agrégations (scalable à l'infini).

    Modes d'extraction :
      fetch_aggregated()  → agrégations date_histogram (RECOMMANDÉ, scalable)
      fetch_sample()      → échantillon de logs bruts (pour analyse ponctuelle)
      fetch_logs()        → scroll complet (uniquement pour petits volumes <100k)
    """

    def __init__(self, connector: ELKConnector):
        self.elk          = connector
        self.index        = os.getenv("ES_LOG_INDEX_PATTERN", "logs-*")
        self._service_map: dict = {}

    # ── MODE 1 : Agrégations (production, scalable) ───────────────────────────

    def fetch_aggregated(
        self,
        days:     int = LOOKBACK_DAYS,
        interval: str = AGG_INTERVAL,
    ) -> pd.DataFrame:
        """
        Extrait des features pré-agrégées depuis Elasticsearch.

        Calcule directement dans ES, par bucket de N minutes :
          - Nombre total de logs
          - Nombre d'erreurs (détectées par regex dans le message)
          - Processus dominant (top service)

        Scalable à n'importe quel volume de logs.
        Retourne un DataFrame de ~buckets de temps, pas de logs individuels.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        logger.info(
            "📊  Extraction agrégée — fenêtre : %d j | intervalle : %s | depuis %s",
            days, interval, since.strftime("%Y-%m-%d %H:%M UTC")
        )

        query = {
            "size": 0,
            "query": {
                "range": {"@timestamp": {"gte": since.isoformat(), "lte": "now"}}
            },
            "aggs": {
                # Buckets temporels de N minutes
                "by_time": {
                    "date_histogram": {
                        "field":              "@timestamp",
                        "fixed_interval":     interval,
                        "min_doc_count":      1,
                    },
                    "aggs": {
                        # Top processus dans ce bucket
                        "top_service": {
                            "terms": {
                                "field": "agent.name",
                                "size":  1,
                            }
                        },
                        # Filtre : logs contenant des mots-clés d'erreur
                        "error_count": {
                            "filter": {
                                "bool": {
                                    "should": [
                                        {"match_phrase": {"message": "error"}},
                                        {"match_phrase": {"message": "failed"}},
                                        {"match_phrase": {"message": "failure"}},
                                        {"match_phrase": {"message": "critical"}},
                                        {"match_phrase": {"message": "fatal"}},
                                        {"match_phrase": {"message": "panic"}},
                                    ]
                                }
                            }
                        },
                        # Filtre : logs contenant des mots-clés d'avertissement
                        "warn_count": {
                            "filter": {
                                "bool": {
                                    "should": [
                                        {"match_phrase": {"message": "warn"}},
                                        {"match_phrase": {"message": "warning"}},
                                    ]
                                }
                            }
                        },
                    }
                }
            }
        }

        resp    = self.elk.client.search(index=self.index, **query)
        buckets = resp["aggregations"]["by_time"]["buckets"]

        if not buckets:
            logger.warning("⚠️  Aucun bucket trouvé pour la période.")
            return pd.DataFrame()

        rows = []
        for b in buckets:
            ts          = pd.Timestamp(b["key_as_string"], tz="UTC")
            total       = b["doc_count"]
            n_errors    = b["error_count"]["doc_count"]
            n_warns     = b["warn_count"]["doc_count"]
            top_svc_bkts= b["top_service"]["buckets"]
            top_service = top_svc_bkts[0]["key"] if top_svc_bkts else "unknown"

            rows.append({
                "timestamp":     ts,
                "total_logs":    total,
                "error_count":   n_errors,
                "warn_count":    n_warns,
                "service":       top_service,
                "log_level":     "ERROR" if n_errors > 0 else ("WARNING" if n_warns > 0 else "INFO"),
            })

        df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
        logger.info(
            "✅  %d buckets extraits (%s → %s)",
            len(df),
            df["timestamp"].min().strftime("%Y-%m-%d %H:%M"),
            df["timestamp"].max().strftime("%Y-%m-%d %H:%M"),
        )
        return df

    # ── MODE 2 : Échantillon intelligent ──────────────────────────────────────

    def fetch_sample(
        self,
        days:       int = LOOKBACK_DAYS,
        max_docs:   int = MAX_SAMPLE_DOCS,
    ) -> pd.DataFrame:
        """
        Extrait un échantillon représentatif :
          - Tous les logs ERROR/WARN/CRITICAL
          - Un échantillon aléatoire des logs INFO

        Utile pour l'analyse qualitative et le debug.
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        logger.info(
            "🎯  Extraction échantillon — fenêtre : %d j | max : %d docs",
            days, max_docs
        )

        # 1 — Tous les logs d'erreur
        error_query = {
            "query": {
                "bool": {
                    "must": [
                        {"range": {"@timestamp": {"gte": since.isoformat()}}},
                    ],
                    "should": [
                        {"match_phrase": {"message": "error"}},
                        {"match_phrase": {"message": "failed"}},
                        {"match_phrase": {"message": "critical"}},
                        {"match_phrase": {"message": "warn"}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "sort": [{"@timestamp": "asc"}],
            "_source": ["@timestamp", "message", "agent", "host", "fileset",
                        "data_stream", "event", "log"],
        }
        error_records = list(self._scroll(error_query))
        logger.info("  → %d logs d'erreur/warning trouvés", len(error_records))

        # 2 — Échantillon INFO (pour contexte)
        remaining    = max(0, max_docs - len(error_records))
        info_records = []
        if remaining > 0:
            info_query = {
                "query": {
                    "bool": {
                        "must": [
                            {"range": {"@timestamp": {"gte": since.isoformat()}}},
                        ],
                        "must_not": [
                            {"match_phrase": {"message": "error"}},
                            {"match_phrase": {"message": "failed"}},
                            {"match_phrase": {"message": "warn"}},
                        ],
                    }
                },
                "sort":    [{"@timestamp": "asc"}],
                "_source": ["@timestamp", "message", "agent", "host",
                            "fileset", "data_stream", "event", "log"],
            }
            # Utilise random_score pour un vrai échantillon aléatoire
            info_query["query"] = {
                "function_score": {
                    "query":      info_query["query"],
                    "functions":  [{"random_score": {}}],
                    "boost_mode": "replace",
                }
            }
            info_records = list(self._scroll_limited(info_query, limit=remaining))
            logger.info("  → %d logs INFO échantillonnés", len(info_records))

        all_records = error_records + info_records
        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = self._normalize_columns(df)
        logger.info("✅  %d logs extraits (échantillon)", len(df))
        return df

    # ── MODE 3 : Scroll complet (petits volumes seulement) ───────────────────

    def fetch_logs(
        self,
        days:          int = LOOKBACK_DAYS,
        level_filter:  list[str] | None = None,
        hosts:         list[str] | None = None,
        extra_filters: dict | None = None,
    ) -> pd.DataFrame:
        """
        Scroll complet — À N'UTILISER QUE pour de petits volumes (<100k logs).
        Pour de gros volumes, utilisez fetch_aggregated() ou fetch_sample().
        """
        since = datetime.now(timezone.utc) - timedelta(days=days)
        query = self._build_query(since, level_filter, hosts, extra_filters)

        logger.info(
            "🔍  Extraction complète — fenêtre : %d jour(s) depuis %s",
            days, since.strftime("%Y-%m-%d %H:%M UTC"),
        )
        records = list(self._scroll(query))

        if not records:
            logger.warning("⚠️  Aucun log trouvé pour la période demandée.")
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df = self._normalize_columns(df)
        logger.info("✅  %d logs extraits.", len(df))
        return df

    # ── Feature Engineering sur données agrégées ─────────────────────────────

    def build_features(
        self,
        df:             pd.DataFrame,
        window_minutes: int = 5,
    ) -> pd.DataFrame:
        """
        Construit les features ML à partir d'un DataFrame agrégé ou brut.
        Compatible avec la sortie de fetch_aggregated() et fetch_logs().
        """
        if df.empty:
            logger.warning("⚠️  DataFrame vide.")
            return df

        df = df.copy()

        # Détecte si c'est un DataFrame agrégé ou brut
        is_aggregated = "total_logs" in df.columns

        if is_aggregated:
            df = self._build_features_aggregated(df, window_minutes)
        else:
            df = self._build_features_raw(df, window_minutes)

        logger.info(
            "✅  Features construites — %d lignes x %d colonnes",
            len(df), len(df.columns)
        )
        return df

    def _build_features_aggregated(self, df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
        """Features pour DataFrame agrégé (issu de fetch_aggregated)."""
        df = df.set_index("timestamp")

        # Taux d'erreur et ratio
        df["error_rate_1m"]   = df["error_count"].rolling("1min",  min_periods=1).sum()
        df["error_rate_5m"]   = df["error_count"].rolling(f"{window_minutes}min", min_periods=1).sum()
        df["total_logs_5m"]   = df["total_logs"].rolling(f"{window_minutes}min", min_periods=1).sum()
        df["error_ratio_5m"]  = df["error_rate_5m"] / df["total_logs_5m"].replace(0, 1)
        df["error_velocity"]  = df["error_rate_1m"].diff().fillna(0)
        df["log_level_num"]   = df["log_level"].map(LOG_LEVEL_MAP).fillna(2)
        df["is_error"]        = df["log_level"].isin(["ERROR", "CRITICAL"]).astype(int)

        # Encodage cyclique temporel
        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)

        # Répétition d'erreurs
        df["repeat_error"] = (df["error_rate_5m"] > 1).astype(int)

        # Encodage service
        cat              = df["service"].fillna("unknown").astype("category")
        df["service_id"] = cat.cat.codes
        self._service_map = dict(enumerate(cat.cat.categories))

        # Message synthétique pour compatibilité
        # Message synthétique informatif
        df["process"]      = df["service"]
        df["host_agent"]   = df["service"]
        df["message"] = (
            "Bucket " + df.index.strftime("%H:%M") +
            " — " + df["total_logs"].astype(str) + " logs, " +
            df["error_count"].astype(str) + " erreurs, " +
            df["warn_count"].astype(str) + " warnings"
        )

        return df.reset_index()

    def _build_features_raw(self, df: pd.DataFrame, window_minutes: int) -> pd.DataFrame:
        """Features pour DataFrame de logs bruts (issu de fetch_logs/fetch_sample)."""
        df = df.set_index("timestamp")

        level_col = self._detect_level_column(df)
        svc_col   = self._detect_service_column(df)

        if level_col:
            df["log_level_num"] = df[level_col].str.upper().map(LOG_LEVEL_MAP).fillna(2)
            df["is_error"]      = df[level_col].str.upper().isin(
                ["ERROR", "CRITICAL", "FATAL"]).astype(int)
            if level_col != "log_level":
                df["log_level"] = df[level_col].str.upper()
        else:
            df["log_level_num"] = 2
            df["is_error"]      = 0
            df["log_level"]     = "UNKNOWN"

        df["error_rate_1m"]  = df["is_error"].rolling("1min",  min_periods=1).sum()
        df["error_rate_5m"]  = df["is_error"].rolling(f"{window_minutes}min", min_periods=1).sum()
        df["total_logs_5m"]  = df["is_error"].rolling(f"{window_minutes}min", min_periods=1).count()
        df["error_ratio_5m"] = df["error_rate_5m"] / df["total_logs_5m"].replace(0, 1)
        df["error_velocity"] = df["error_rate_1m"].diff().fillna(0)

        df["hour_sin"] = np.sin(2 * np.pi * df.index.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df.index.hour / 24)
        df["dow_sin"]  = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df["dow_cos"]  = np.cos(2 * np.pi * df.index.dayofweek / 7)

        if "message" in df.columns:
            df["repeat_error"] = (
                df.groupby("message")["is_error"]
                .transform(lambda s: s.rolling("5min", min_periods=1).sum()) > 1
            ).astype(int)
        else:
            df["repeat_error"] = 0

        if svc_col:
            cat              = df[svc_col].fillna("unknown").astype("category")
            df["service_id"] = cat.cat.codes
            self._service_map = dict(enumerate(cat.cat.categories))
            if svc_col != "service":
                df["service"] = df[svc_col]
        else:
            df["service_id"]  = 0
            df["service"]     = "unknown"
            self._service_map = {}

        return df.reset_index()

    # ── Helpers internes ──────────────────────────────────────────────────────

    def _build_query(self, since, level_filter, hosts, extra_filters) -> dict:
        must = [{"range": {"@timestamp": {"gte": since.isoformat(), "lte": "now"}}}]
        if level_filter:
            must.append({"bool": {"should": [
                {"terms": {"log.level": [v.lower() for v in level_filter]}},
                {"terms": {"log.syslog.severity.name": [v.lower() for v in level_filter]}},
            ]}})
        if hosts:
            must.append({"terms": {"agent.name": hosts}})
        if extra_filters:
            must.append(extra_filters)
        return {
            "query":   {"bool": {"must": must}},
            "sort":    [{"@timestamp": {"order": "asc"}}],
            "_source": [
                "@timestamp", "message", "event.original",
                "log.level", "log.syslog.severity.name",
                "agent", "host", "fileset", "data_stream",
                "event", "log", "process", "service",
            ],
        }

    def _scroll(self, query: dict) -> Iterator[dict]:
        pit    = self.elk.client.open_point_in_time(index=self.index, keep_alive="5m")
        pit_id = pit["id"]
        base   = {k: v for k, v in query.items() if k != "sort"}
        sa     = None
        total  = 0
        try:
            while True:
                kw = {
                    **base,
                    "size":             SCROLL_PAGE_SIZE,
                    "sort":             [{"@timestamp": "asc"}, {"_shard_doc": "asc"}],
                    "pit":              {"id": pit_id, "keep_alive": "5m"},
                    "track_total_hits": False,
                }
                if sa:
                    kw["search_after"] = sa
                resp   = self.elk.client.search(**kw)
                pit_id = resp.get("pit_id", pit_id)
                hits   = resp["hits"]["hits"]
                if not hits:
                    break
                for h in hits:
                    yield h.get("_source", {})
                total += len(hits)
                sa     = hits[-1]["sort"]
                if total % 10_000 == 0:
                    logger.info("  … %d documents extraits", total)
                if len(hits) < SCROLL_PAGE_SIZE:
                    break
        finally:
            try:
                self.elk.client.close_point_in_time(id=pit_id)
            except Exception:
                pass

    def _scroll_limited(self, query: dict, limit: int) -> Iterator[dict]:
        """Scroll avec limite de documents."""
        count = 0
        for doc in self._scroll(query):
            if count >= limit:
                break
            yield doc
            count += 1

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        def get_nested(series, *keys):
            def _get(val):
                for k in keys:
                    if isinstance(val, dict):
                        val = val.get(k)
                    else:
                        return None
                return val
            return series.apply(_get)

        if "agent" in df.columns:
            df["host_agent"] = get_nested(df["agent"], "name").fillna("unknown")
            df["agent_type"] = get_nested(df["agent"], "type").fillna("unknown")
            df.drop(columns=["agent"], inplace=True)
        if "host" in df.columns:
            df["host_name"]  = get_nested(df["host"], "name").fillna("unknown")
            df.drop(columns=["host"], inplace=True)
        if "fileset" in df.columns:
            df["fileset"]    = get_nested(df["fileset"], "name").fillna("unknown")
        if "log" in df.columns:
            df["log_file"]   = get_nested(df["log"], "file", "path").fillna("unknown")
            df.drop(columns=["log"], inplace=True)
        if "data_stream" in df.columns:
            df["ds_dataset"] = get_nested(df["data_stream"], "dataset").fillna("unknown")
            df.drop(columns=["data_stream"], inplace=True)
        if "event" in df.columns:
            df["event_dataset"] = get_nested(df["event"], "dataset").fillna("unknown")
            if "message" not in df.columns:
                df["message"] = get_nested(df["event"], "original").fillna("")
            df.drop(columns=["event"], inplace=True)

        ts_col = "@timestamp" if "@timestamp" in df.columns else "timestamp"
        df["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
        if ts_col != "timestamp" and ts_col in df.columns:
            df.drop(columns=[ts_col], inplace=True)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

        if "message" not in df.columns:
            df["message"] = ""

        df["process"] = df["message"].apply(parse_syslog_process)

        for cand in ["log.level", "log.syslog.severity.name"]:
            if cand in df.columns and df[cand].notna().any():
                df["log_level"] = df[cand].astype(str).str.upper().replace("NAN", "UNKNOWN")
                break
        else:
            df["log_level"] = df["message"].apply(parse_syslog_level)

        if "host_agent" in df.columns:
            df["service"] = df["host_agent"] + "/" + df["process"]
        else:
            df["service"] = "unknown/" + df["process"]

        return df

    def _detect_level_column(self, df):
        for col in ["log_level", "level", "log.level", "severity"]:
            if col in df.columns:
                return col
        return None

    def _detect_service_column(self, df):
        for col in ["service", "host_agent", "agent_name", "process"]:
            if col in df.columns:
                return col
        return None

    def inspect_fields(self, df, n=2):
        print("\n" + "─" * 70)
        print(f"  {len(df.columns)} colonnes disponibles :")
        print("─" * 70)
        for col in sorted(df.columns):
            vals = df[col].dropna().head(n).tolist()
            print(f"  {col:<38} {str(vals)[:60]}")
        print("─" * 70 + "\n")

    def summary(self, df: pd.DataFrame) -> dict:
        if df.empty:
            return {"status": "empty"}
        level_counts   = df["log_level"].value_counts().to_dict() if "log_level" in df.columns else {}
        service_counts = df["service"].value_counts().head(10).to_dict() if "service" in df.columns else {}
        return {
            "total_logs":     len(df),
            "period_start":   str(df["timestamp"].min()),
            "period_end":     str(df["timestamp"].max()),
            "levels":         level_counts,
            "top_services":   service_counts,
            "error_rate_pct": round(
                df["is_error"].mean() * 100, 2
            ) if "is_error" in df.columns else 0,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="AuraLog — Extraction v2")
    parser.add_argument("--days",   type=int, default=LOOKBACK_DAYS)
    parser.add_argument("--mode",   choices=["agg", "sample", "full"], default="agg",
                        help="agg=agrégations (défaut), sample=échantillon, full=scroll complet")
    parser.add_argument("--index",  type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print(f"  AuraLog — Extraction [{args.mode}]")
    print("=" * 60)

    with ELKConnector() as elk:
        extractor = LogExtractor(elk)
        if args.index:
            extractor.index = args.index

        if args.mode == "agg":
            df_raw = extractor.fetch_aggregated(days=args.days)
        elif args.mode == "sample":
            df_raw = extractor.fetch_sample(days=args.days)
        else:
            df_raw = extractor.fetch_logs(days=args.days)

        if df_raw.empty:
            print("⚠️  Aucune donnée.")
        else:
            df_feat = extractor.build_features(df_raw)
            stats   = extractor.summary(df_feat)
            print("\n📊  Résumé :")
            print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
            print(f"\n✅  {len(df_feat)} buckets/lignes × {len(df_feat.columns)} colonnes")

    print("\n✅  Test terminé.")
