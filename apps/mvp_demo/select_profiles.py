
from collections import Counter

import pandas as pd

from src.services.mvp_data_service import load_mvp_context


def get_genres(value):
    if not isinstance(value, str):
        return []

    return [
        genre.strip()
        for genre in value.split("|")
        if genre.strip()
        and genre.strip() != "(no genres listed)"
    ]


def user_genre_profile(user_id, train, catalog):

    positive = train.loc[
        (train["user_id"] == user_id)
        & (train["rating"] >= 4.0)
    ]

    movies = positive.merge(
        catalog[["movie_id", "genres"]],
        on="movie_id",
        how="left",
        validate="many_to_one",
    )

    counts = Counter()

    for genres in movies["genres"]:
        counts.update(get_genres(genres))

    total = sum(counts.values())

    return {
        genre: count / total
        for genre, count in counts.items()
    } if total else {}


def compare_profiles():

    context = load_mvp_context()
    train = context["train"]
    catalog = context["catalog"]

    reference_id = 1478
    reference = user_genre_profile(
        reference_id, train, catalog
    )

    if not reference:
        raise ValueError(
            "El usuario 1478 no tiene géneros positivos."
        )

    candidates = []

    for user_id, group in train.groupby("user_id"):

        user_id = int(user_id)

        if user_id == reference_id:
            continue

        positives = int((group["rating"] >= 4).sum())

        if len(group) < 15 or positives < 10:
            continue

        profile = user_genre_profile(
            user_id, train, catalog
        )

        if not profile:
            continue

        genres = set(reference) | set(profile)

        # Distancia de variación total entre
        # distribuciones de géneros.
        distance = 0.5 * sum(
            abs(
                reference.get(genre, 0)
                - profile.get(genre, 0)
            )
            for genre in genres
        )

        top_genres = sorted(
            profile.items(),
            key=lambda item: (-item[1], item[0]),
        )[:5]

        candidates.append({
            "user_id": user_id,
            "ratings": len(group),
            "positives": positives,
            "distance": distance,
            "top_genres": ", ".join(
                f"{genre} {share:.0%}"
                for genre, share in top_genres
            ),
        })

    results = pd.DataFrame(candidates).sort_values(
        ["distance", "user_id"],
        ascending=[False, True],
    )

    print("\nPERFIL DE REFERENCIA: USUARIO 1478")
    print("----------------------------------")

    for genre, share in sorted(
        reference.items(),
        key=lambda item: -item[1],
    )[:5]:
        print(f"{genre}: {share:.1%}")

    print("\nCANDIDATOS CON GUSTOS DIFERENTES")
    print("----------------------------------")

    print(
        results.head(10).to_string(index=False)
    )


if __name__ == "__main__":
    compare_profiles()
