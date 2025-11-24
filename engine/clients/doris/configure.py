import time
import logging
import mysql.connector

from doris_vector_search import DorisVectorClient, AuthOptions

from benchmark.dataset import Dataset
from engine.base_client.configure import BaseConfigurator
from engine.base_client.distances import Distance
from engine.clients.doris.config import get_db_config, ensure_database_exists


logger = logging.getLogger(__name__)


class DorisConfigurator(BaseConfigurator):
    DISTANCE_MAPPING = {
        Distance.L2: "l2_distance",
        Distance.COSINE: "inner_product",  # cosine 通过归一化 + 内积实现
        Distance.DOT: "inner_product",
    }

    # Used by uploader to emulate index build inclusion in total_time
    last_index_build_time: float | None = None

    def __init__(self, host, collection_params: dict, connection_params: dict):
        super().__init__(host, collection_params, connection_params)
        cfg = get_db_config(host, connection_params)
        ensure_database_exists(cfg)
        self.client = DorisVectorClient(
            database=cfg["database"],
            auth_options=AuthOptions(
                host=cfg["host"],
                query_port=cfg["query_port"],
                http_port=cfg["http_port"],
                user=cfg["user"],
                password=cfg["password"],
            ),
        )

    def clean(self):
        table = self.collection_params.get("table", "items")
        logger.info("Dropping Doris table if exists: %s", table)
        try:
            self.client.drop_table(table)
            logger.debug("Dropped table: %s", table)
        except Exception as e:
            logger.debug("Ignoring table drop error for %s: %s", table, e)

    def recreate(self, dataset: Dataset, collection_params):
        table = collection_params.get("table", "items")
        dim = dataset.config.vector_size
        metric = self.DISTANCE_MAPPING[dataset.config.distance]

        # HNSW index configuration
        hnsw_cfg = collection_params.get("hnsw_config", {"m": 16, "ef_construct": 200})
        m = hnsw_cfg.get("m", 16)
        ef_construction = hnsw_cfg.get("ef_construction", hnsw_cfg.get("ef_construct", 200))

        logger.info(
            "Creating Doris table=%s dim=%s metric=%s hnsw(m=%s,ef=%s) via native SQL",
            table,
            dim,
            metric,
            m,
            ef_construction,
        )
        
        # Use native SQL connector to avoid SDK PyArrow issues
        cfg = get_db_config(self.host, self.connection_params)
        conn = mysql.connector.connect(
            host=cfg["host"],
            port=cfg["query_port"],
            user=cfg["user"],
            password=cfg["password"],
            database=cfg["database"],
        )
        
        start = time.time()
        try:
            cur = conn.cursor()
            
            # Create table with basic schema
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS `{table}` (
                `id` INT,
                `embedding` ARRAY<FLOAT> NOT NULL
            ) ENGINE=OLAP
            DUPLICATE KEY (`id`)
            DISTRIBUTED BY HASH(`id`) BUCKETS 1
            PROPERTIES (
                "replication_num" = "1"
            )
            """
            logger.debug("Executing CREATE TABLE SQL")
            cur.execute(create_table_sql)
            
            # Create HNSW ANN index
            create_index_sql = f"""
            CREATE INDEX `idx_embedding_hnsw` 
            ON `{table}`(`embedding`) 
            USING ANN 
            PROPERTIES(
                "index_type" = "hnsw",
                "metric_type" = "{metric}",
                "dim" = "{dim}",
                "max_degree" = "{m}",
                "ef_construction" = "{ef_construction}"
            )
            """
            logger.debug("Executing CREATE INDEX SQL")
            cur.execute(create_index_sql)
            
            conn.commit()
            self.last_index_build_time = time.time() - start
            logger.info(
                "Created Doris table=%s with HNSW index time=%.3fs", table, self.last_index_build_time
            )
        except Exception as e:
            logger.error(
                "Failed to create table via SQL table=%s error=%s", table, e
            )
            raise
        finally:
            try:
                cur.close()
            except Exception:
                pass
            conn.close()

    def execution_params(self, distance, vector_size) -> dict:
        return {"normalize": distance == Distance.COSINE}

    def delete_client(self):
        pass





