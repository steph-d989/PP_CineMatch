from pathlib import Path

# Raíz del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Carpetas principales
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Capas Medallion
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"

# Fuentes Bronze
BRONZE_MOVIELENS_DIR = BRONZE_DIR / "movielens"
BRONZE_TMDB_DIR = BRONZE_DIR / "tmdb"
BRONZE_WATCHMODE_DIR = BRONZE_DIR / "watchmode"

# Fuentes Silver
SILVER_MOVIELENS_DIR = SILVER_DIR / "movielens"
SILVER_TMDB_DIR = SILVER_DIR / "tmdb"
SILVER_WATCHMODE_DIR = SILVER_DIR / "watchmode"
SILVER_INTEGRATED_DIR = SILVER_DIR / "integrated"

# Fuentes gold
GOLD_ANALYTICS_DIR = GOLD_DIR / "analytics"
GOLD_ML_DIR = GOLD_DIR / "ml"

# Gold Analytics
GOLD_DIMENSIONAL_DIR = GOLD_ANALYTICS_DIR / "dimensional"
GOLD_DIMENSIONS_DIR = GOLD_DIMENSIONAL_DIR / "dimensions"
GOLD_FACTS_DIR = GOLD_DIMENSIONAL_DIR / "facts"
GOLD_METRICS_DIR = GOLD_ANALYTICS_DIR / "metrics"

# Gold ML
GOLD_ML_CONTENT_DIR = GOLD_ML_DIR / "content_based"
GOLD_ML_COLLABORATIVE_DIR = GOLD_ML_DIR / "collaborative"
GOLD_ML_HYBRID_DIR = GOLD_ML_DIR / "hybrid"

# Gold Serving
GOLD_SERVING_DIR = GOLD_DIR / "serving"

# Crear carpetas si no existen
for path in [
    BRONZE_MOVIELENS_DIR,
    BRONZE_TMDB_DIR,
    BRONZE_WATCHMODE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    MODELS_DIR,
    LOGS_DIR,
    SILVER_MOVIELENS_DIR,
    SILVER_TMDB_DIR,
    SILVER_WATCHMODE_DIR,
    SILVER_INTEGRATED_DIR,
    GOLD_ANALYTICS_DIR,
    GOLD_DIMENSIONAL_DIR,
    GOLD_DIMENSIONS_DIR,
    GOLD_FACTS_DIR,
    GOLD_METRICS_DIR,
    GOLD_ML_DIR,
    GOLD_ML_CONTENT_DIR,
    GOLD_ML_COLLABORATIVE_DIR,
    GOLD_ML_HYBRID_DIR,
    GOLD_SERVING_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)