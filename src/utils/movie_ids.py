import pandas as pd

from src.utils.paths import BRONZE_MOVIELENS_DIR


### FUNCION QUE LEE LINKS.CSV Y DEVUELVE TMBID VALIDOS

def load_tmdb_ids_from_movielens() -> list[int]:

    ruta_links = (
        BRONZE_MOVIELENS_DIR
        / "ml-32m"
        / "links.csv"
    )

    if not ruta_links.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {ruta_links}"
        )

    links = pd.read_csv(ruta_links)

    tmdb_ids = (
        links["tmdbId"]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    return tmdb_ids