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

# Crear carpetas si no existen
for path in [
    BRONZE_MOVIELENS_DIR,
    BRONZE_TMDB_DIR,
    BRONZE_WATCHMODE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    MODELS_DIR,
    LOGS_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)