from typing import List
import time
import logging

import pandas as pd

from doris_vector_search import DorisVectorClient, AuthOptions, LoadOptions

from dataset_reader.base_reader import Record
from engine.base_client.upload import BaseUploader
from engine.clients.doris.config import get_db_config, ensure_database_exists
from engine.clients.doris.configure import DorisConfigurator


logger = logging.getLogger(__name__)


class DorisUploader(BaseUploader):
    client = None
    table = None

    @classmethod
    def init_client(cls, host, distance, connection_params: dict, upload_params: dict):
        # Close previous if reinitializing to avoid socket leaks
        if cls.table is not None:
            try:
                close_fn = getattr(cls.table, "close", None)
                if callable(close_fn):
                    close_fn()
                    logger.debug("Closed previous Doris table before reinit")
            except Exception as e:
                logger.debug("Ignoring table close error during reinit: %s", e)
            finally:
                cls.table = None
        if cls.client is not None:
            try:
                close_fn = getattr(cls.client, "close", None)
                if callable(close_fn):
                    close_fn()
                    logger.debug("Closed previous Doris client before reinit")
            except Exception as e:
                logger.debug("Ignoring client close error during reinit: %s", e)
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
            load_options=LoadOptions(batch_size=upload_params.get("batch_size", 1024)),
        )
        table_name = upload_params.get("table", connection_params.get("table", "items"))
        logger.info("Opening Doris table for upload: %s", table_name)
        cls.table = cls.client.open_table(table_name)

    @classmethod
    def upload_batch(cls, batch: List[Record]):
        data = pd.DataFrame(
            [
                {"id": r.id, "embedding": r.vector}
                for r in batch
                if r.vector is not None
            ]
        )
        if not data.empty:
            cls.table.add(data)

    @classmethod
    def delete_client(cls):
        # Explicitly close resources
        try:
            if cls.table is not None:
                close_fn = getattr(cls.table, "close", None)
                if callable(close_fn):
                    close_fn()
                    logger.debug("Closed Doris table")
        except Exception as e:
            logger.debug("Ignoring table close error: %s", e)
        finally:
            cls.table = None
        try:
            if cls.client is not None:
                close_fn = getattr(cls.client, "close", None)
                if callable(close_fn):
                    close_fn()
                    logger.debug("Closed Doris client")
        except Exception as e:
            logger.debug("Ignoring client close error: %s", e)
        finally:
            cls.client = None

    @classmethod
    def post_upload(cls, _distance):
        # Align total_time semantics with Qdrant/Milvus:
        # upload_time = data submission time only;
        # total_time = upload_time + index building/optimization time.
        # Doris builds ANN index at table creation. We measured that in
        # DorisConfigurator.last_index_build_time. Emulate the same inclusion
        # by waiting that duration here so total_time reflects both phases.
        try:
            index_time = DorisConfigurator.last_index_build_time
            if index_time is not None and index_time > 0:
                time.sleep(index_time)
        except Exception:
            pass
        return {}



