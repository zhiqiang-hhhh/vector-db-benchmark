import logging
import mysql.connector

logger = logging.getLogger(__name__)

DORIS_QUERY_PORT = 9030
DORIS_HTTP_PORT = 8030
DORIS_DB = "test"
DORIS_USER = "root"
DORIS_PASSWORD = ""

DORIS_QUERY_PORT = 9030
DORIS_HTTP_PORT = 8030
DORIS_DB = "test"
DORIS_USER = "root"
DORIS_PASSWORD = ""


def get_db_config(host, connection_params):
    cfg = {
        "host": host or "localhost",
        "query_port": DORIS_QUERY_PORT,
        "http_port": DORIS_HTTP_PORT,
        "database": DORIS_DB,
        "user": DORIS_USER,
        "password": DORIS_PASSWORD,
        **(connection_params or {}),
    }
    # Avoid logging sensitive fields like password.
    logger.debug(
        "Constructed Doris DB config host=%s query_port=%s http_port=%s database=%s user=%s overrides=%s",
        cfg["host"], cfg["query_port"], cfg["http_port"], cfg["database"], cfg["user"], list((connection_params or {}).keys()),
    )
    return cfg


def ensure_database_exists(cfg: dict) -> None:
    """Ensure target database exists before SDK connects to it."""
    logger.info("Ensuring Doris database exists: %s (host=%s)" , cfg.get("database"), cfg.get("host"))
    try:
        conn = mysql.connector.connect(
            host=cfg["host"], port=cfg["query_port"], user=cfg["user"], password=cfg["password"]
        )
    except Exception as e:
        logger.error("Failed to connect to Doris host=%s port=%s user=%s error=%s", cfg.get("host"), cfg.get("query_port"), cfg.get("user"), e)
        raise
    try:
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{cfg['database']}`")
        conn.commit()
        logger.info("Database ensured (created if absent): %s", cfg.get("database"))
    except Exception as e:
        logger.error("Failed ensuring database %s error=%s", cfg.get("database"), e)
        raise
    finally:
        try:
            cur.close()
        except Exception:
            pass
        conn.close()
        logger.debug("Closed Doris connection for database ensure operation")


