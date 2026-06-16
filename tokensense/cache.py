import json
import sqlite3
import uuid
from typing import Callable, List, Optional, Tuple

class SemanticCache:
    def get(self, prompt: str) -> Optional[str]:
        raise NotImplementedError

    def set(self, prompt: str, response: str) -> None:
        raise NotImplementedError

class SQLiteVectorCache(SemanticCache):
    """
    Local Semantic Cache using sqlite-vec for vector search.
    Requires `pip install tokensense-ai[cache]`
    """
    def __init__(
        self, 
        embedding_fn: Callable[[str], List[float]], 
        dim: int,
        db_path: str = "./tokensense_cache.db",
        threshold: float = 0.95
    ):
        try:
            import sqlite_vec
        except ImportError:
            raise ImportError(
                "sqlite-vec package not found. "
                "Please install with: pip install 'tokensense-ai[cache]'"
            )
        
        self.embedding_fn = embedding_fn
        self.dim = dim
        self.db_path = db_path
        self.threshold = threshold
        self._sqlite_vec = sqlite_vec
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.enable_load_extension(True)
        self._sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_content (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL
            )
        """)
        
        # Create vector table
        cursor.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_cache USING vec0(
                id INTEGER PRIMARY KEY,
                embedding float[{self.dim}]
            )
        """)
        
        conn.commit()
        conn.close()

    def get(self, prompt: str) -> Optional[str]:
        if not prompt:
            return None
            
        try:
            import numpy as np
        except ImportError:
            raise ImportError("numpy is required for caching. pip install 'tokensense-ai[cache]'")
            
        # Get embedding for the prompt
        query_embedding = self.embedding_fn(prompt)
        
        # Ensure it's a float array
        embedding_bytes = np.array(query_embedding, dtype=np.float32).tobytes()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Vector search using vec_distance_cosine (1.0 = exact match, smaller distance is better)
        # However, sqlite-vec often uses vec_distance_L2. Let's use cosine distance.
        cursor.execute("""
            SELECT 
                c.response, 
                vec_distance_cosine(v.embedding, ?) as dist
            FROM vec_cache v
            JOIN cache_content c ON v.id = c.id
            ORDER BY dist ASC
            LIMIT 1
        """, (embedding_bytes,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            response_text, dist = result
            # Cosine distance: 0 is identical. So dist < (1.0 - threshold)
            if dist < (1.0 - self.threshold):
                return response_text
                
        return None

    def set(self, prompt: str, response: str) -> None:
        if not prompt or not response:
            return
            
        try:
            import numpy as np
        except ImportError:
            return
            
        embedding = self.embedding_fn(prompt)
        embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO cache_content (prompt, response)
            VALUES (?, ?)
        """, (prompt, response))
        row_id = cursor.lastrowid
        
        cursor.execute("""
            INSERT INTO vec_cache (id, embedding)
            VALUES (?, ?)
        """, (row_id, embedding_bytes))
        
        conn.commit()
        conn.close()
