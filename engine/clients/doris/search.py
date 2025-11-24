from typing import List, Tuple
import logging
import numpy as np

from doris_vector_search import DorisVectorClient, AuthOptions

from dataset_reader.base_reader import Query
from engine.base_client.distances import Distance
from engine.base_client.search import BaseSearcher
from engine.clients.doris.config import get_db_config, ensure_database_exists


logger = logging.getLogger(__name__)


class DorisSearcher(BaseSearcher):
    client = None
    table = None
    metric = None

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, search_params: dict):
        # Close previous resources if reinitializing
        if cls.table is not None:
            try:
                close_fn = getattr(cls.table, "close", None)
                if callable(close_fn):
                    close_fn()
                    logger.debug("Closed previous Doris search table before reinit")
            except Exception as e:
                logger.debug("Ignoring search table close error during reinit: %s", e)
            finally:
                cls.table = None
        if cls.client is not None:
            try:
                close_fn = getattr(cls.client, "close", None)
                if callable(close_fn):
                    close_fn()
                    logger.debug("Closed previous Doris search client before reinit")
            except Exception as e:
                logger.debug("Ignoring search client close error during reinit: %s", e)
            finally:
                cls.client = None

        cfg = get_db_config(host, connection_params)
        ensure_database_exists(cfg)
        cls.client = DorisVectorClient(
            database=cfg["database"],
            auth_options=AuthOptions(
                host=cfg["host"],
                query_port=cfg["query_port"],
                http_port=cfg["http_port"],
                user=cfg["user"],
                password=cfg["password"],
            ),
        )
        table_name = search_params.get("table", connection_params.get("table", "items"))
        logger.info("Opening Doris table for search: %s", table_name)
        cls.table = cls.client.open_table(table_name)
        cls.metric = {
            Distance.L2: "l2_distance",
            Distance.COSINE: "inner_product",  # 归一化后等价 cosine
            Distance.DOT: "inner_product",
        }[distance]

    @classmethod
    def search_one(cls, query: Query, top) -> List[Tuple[int, float]]:
        vec = np.asarray(query.vector, dtype=float)
        if not np.isfinite(vec).all():
            vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

        df = (
            cls.table.search(vec.tolist(), metric_type=cls.metric)
            .limit(top)
            .select(["id"])
            .to_pandas()
        )

        return [(int(row["id"]), 0.0) for _, row in df.iterrows()]

    @classmethod
    def delete_client(cls):
        # Explicitly close resources
        try:
            if cls.table is not None:
                close_fn = getattr(cls.table, "close", None)
                if callable(close_fn):
                    close_fn()
                    logger.debug("Closed Doris search table")
        except Exception as e:
            logger.debug("Ignoring search table close error: %s", e)
        finally:
            cls.table = None
        try:
            if cls.client is not None:
                close_fn = getattr(cls.client, "close", None)
                if callable(close_fn):
                    close_fn()
                    logger.debug("Closed Doris search client")
        except Exception as e:
            logger.debug("Ignoring search client close error: %s", e)
        finally:
            cls.client = None



