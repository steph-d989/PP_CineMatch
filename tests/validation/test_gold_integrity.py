
import unittest
import pandas as pd

from src.validation.gold_integrity import (
    validate_fact_rating_user_references,
    validate_fact_rating_date_references,
    validate_fact_availability_date_references,
    validate_fact_availability_movie_references,
)


class GoldIntegrityTests(unittest.TestCase):

    
    def test_fact_availability_valida_referencias_dim_date(self):

        # Dimensión de calendario.
        dim_date = pd.DataFrame({
            "date_key": [
                20261110,
                20261111,
                20261112,
            ],
        })

        # Ofertas con fechas válidas.
        fact_availability = pd.DataFrame({
            "movie_id": [1, 2],
            "provider_id": [10, 20],
            "date_key": [20261110, 20261112],
        })

        # Caso correcto: ambas fechas existen.
        validate_fact_availability_date_references(
            fact_availability=fact_availability,
            dim_date=dim_date,
        )

        # Caso incorrecto: fecha inexistente.
        fact_invalida = fact_availability.copy()

        fact_invalida.loc[
            1, "date_key"
        ] = 20350101

        with self.assertRaisesRegex(
            ValueError,
            "referencias a fechas inexistentes",
        ):
            validate_fact_availability_date_references(
                fact_availability=fact_invalida,
                dim_date=dim_date,
            )

        # Caso incorrecto: fecha duplicada en dimensión.
        dim_duplicada = pd.DataFrame({
            "date_key": [
                20261110,
                20261110,
                20261112,
            ],
        })

        with self.assertRaisesRegex(
            ValueError,
            "date_key duplicados",
        ):
            validate_fact_availability_date_references(
                fact_availability=fact_availability,
                dim_date=dim_duplicada,
            )

        # Caso incorrecto: fecha nula en la FACT.
        fact_nula = fact_availability.copy()

        fact_nula["date_key"] = (
            fact_nula["date_key"].astype("Int64")
        )

        fact_nula.loc[
            0, "date_key"
        ] = pd.NA

        with self.assertRaisesRegex(
            ValueError,
            "date_key nulos",
        ):
            validate_fact_availability_date_references(
                fact_availability=fact_nula,
                dim_date=dim_date,
            )

    def test_fact_availability_valida_referencias_dim_movie(self):

        dim_movie = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        fact_availability = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "provider_id": [10, 20, 30],
        })

        # Caso válido.
        validate_fact_availability_movie_references(
            fact_availability=fact_availability,
            dim_movie=dim_movie,
        )

        # Caso inválido: película inexistente.
        fact_invalida = fact_availability.copy()
        fact_invalida.loc[2, "movie_id"] = 999

        with self.assertRaisesRegex(
            ValueError,
            "referencias a películas inexistentes",
        ):
            validate_fact_availability_movie_references(
                fact_availability=fact_invalida,
                dim_movie=dim_movie,
            )

        # Caso inválido: duplicados en la dimensión.
        dim_duplicada = pd.DataFrame({
            "movie_id": [1, 2, 2],
        })

        with self.assertRaisesRegex(
            ValueError,
            "movie_id duplicados",
        ):
            validate_fact_availability_movie_references(
                fact_availability=fact_availability,
                dim_movie=dim_duplicada,
            )

        # Caso inválido: hecho con película nula.
        fact_nula = fact_availability.copy()

        fact_nula["movie_id"] = (
            fact_nula["movie_id"].astype("Int64")
        )

        fact_nula.loc[0, "movie_id"] = pd.NA

        with self.assertRaisesRegex(
            ValueError,
            "movie_id nulos",
        ):
            validate_fact_availability_movie_references(
                fact_availability=fact_nula,
                dim_movie=dim_movie,
            )

    def test_fact_availability_valida_referencias_dim_platform(self):

        from src.validation.gold_integrity import (
            validate_fact_availability_platform_references,
        )

        dim_platform = pd.DataFrame({
            "provider_id": [10, 20, 30],
            "provider_name": [
                "Netflix",
                "Prime Video",
                "Disney+",
            ],
        })

        fact_availability = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "provider_id": [10, 20, 30],
            "region": ["PE", "PE", "PE"],
        })

        # Caso válido: todas las plataformas existen.
        validate_fact_availability_platform_references(
            fact_availability=fact_availability,
            dim_platform=dim_platform,
        )

        # Caso inválido: plataforma inexistente.
        fact_invalida = fact_availability.copy()
        fact_invalida.loc[2, "provider_id"] = 999

        with self.assertRaisesRegex(
            ValueError,
            "referencias a plataformas inexistentes",
        ):
            validate_fact_availability_platform_references(
                fact_availability=fact_invalida,
                dim_platform=dim_platform,
            )

        # Caso inválido: clave duplicada en dimensión.
        dim_duplicada = pd.DataFrame({
            "provider_id": [10, 20, 20],
        })

        with self.assertRaisesRegex(
            ValueError,
            "provider_id duplicados",
        ):
            validate_fact_availability_platform_references(
                fact_availability=fact_availability,
                dim_platform=dim_duplicada,
            )

        # Caso inválido: hecho sin identificador.
        fact_nula = fact_availability.copy()

        fact_nula["provider_id"] = (
            fact_nula["provider_id"].astype("Int64")
        )

        fact_nula.loc[0, "provider_id"] = pd.NA

        with self.assertRaisesRegex(
            ValueError,
            "provider_id nulos",
        ):
            validate_fact_availability_platform_references(
                fact_availability=fact_nula,
                dim_platform=dim_platform,
            )
    
    def test_fact_rating_valida_referencias_dim_date(self):

        dim_date = pd.DataFrame({
            "date_key": [20240101, 20240102, 20240103],
        })

        fact_rating = pd.DataFrame({
            "user_id": [10, 20],
            "movie_id": [1, 2],
            "date_key": [20240101, 20240102],
            "rating": [4.5, 3.0],
        })

        # Caso correcto.
        validate_fact_rating_date_references(
            fact_rating=fact_rating,
            dim_date=dim_date,
        )

        # Caso incorrecto: fecha que no existe.
        fact_invalida = fact_rating.copy()
        fact_invalida.loc[1, "date_key"] = 20991231

        with self.assertRaisesRegex(
            ValueError,
            "referencias a fechas inexistentes",
        ):
            validate_fact_rating_date_references(
                fact_rating=fact_invalida,
                dim_date=dim_date,
            )

        # Caso incorrecto: fecha duplicada en dimensión.
        dim_duplicada = pd.DataFrame({
            "date_key": [20240101, 20240101, 20240102],
        })

        with self.assertRaisesRegex(
            ValueError,
            "date_key duplicados",
        ):
            validate_fact_rating_date_references(
                fact_rating=fact_rating,
                dim_date=dim_duplicada,
            )

        # Caso incorrecto: fecha nula en el hecho.
        fact_nula = fact_rating.copy()
        fact_nula["date_key"] = (
            fact_nula["date_key"].astype("Int64")
        )
        fact_nula.loc[0, "date_key"] = pd.NA

        with self.assertRaisesRegex(
            ValueError,
            "date_key nulos",
        ):
            validate_fact_rating_date_references(
                fact_rating=fact_nula,
                dim_date=dim_date,
            )

    def test_fact_rating_valida_referencias_dim_movie(self):

        from src.validation.gold_integrity import (
            validate_fact_rating_movie_references,
        )

        dim_movie = pd.DataFrame({
            "movie_id": [1, 2, 3],
            "title": [
                "Pelicula A",
                "Pelicula B",
                "Pelicula C",
            ],
        })

        fact_rating = pd.DataFrame({
            "user_id": [10, 20, 30],
            "movie_id": [1, 2, 3],
            "rating": [4.5, 3.0, 5.0],
        })

        # Caso correcto: todas las películas existen.
        validate_fact_rating_movie_references(
            fact_rating=fact_rating,
            dim_movie=dim_movie,
        )

        # Caso incorrecto: película inexistente.
        fact_invalida = fact_rating.copy()
        fact_invalida.loc[2, "movie_id"] = 999

        with self.assertRaisesRegex(
            ValueError,
            "referencias a películas inexistentes",
        ):
            validate_fact_rating_movie_references(
                fact_rating=fact_invalida,
                dim_movie=dim_movie,
            )

        # Caso incorrecto: clave dimensional duplicada.
        dim_duplicada = pd.DataFrame({
            "movie_id": [1, 2, 2, 3],
        })

        with self.assertRaisesRegex(
            ValueError,
            "movie_id duplicados",
        ):
            validate_fact_rating_movie_references(
                fact_rating=fact_rating,
                dim_movie=dim_duplicada,
            )

        # Caso incorrecto: hecho sin identificador.
        fact_nula = fact_rating.copy()

        fact_nula["movie_id"] = (
            fact_nula["movie_id"].astype("Int64")
        )

        fact_nula.loc[0, "movie_id"] = pd.NA

        with self.assertRaisesRegex(
            ValueError,
            "movie_id nulos",
        ):
            validate_fact_rating_movie_references(
                fact_rating=fact_nula,
                dim_movie=dim_movie,
            )
    
    def test_fact_rating_valida_referencias_dim_user(self):

        dim_user = pd.DataFrame({
            "user_id": [10, 20, 30],
        })

        fact_rating = pd.DataFrame({
            "user_id": [10, 20, 10, 30],
            "movie_id": [1, 2, 3, 1],
            "rating": [4.5, 3.0, 5.0, 4.0],
        })

        # Escenario válido: todos los usuarios existen.
        validate_fact_rating_user_references(
            fact_rating=fact_rating,
            dim_user=dim_user,
        )

        # Escenario inválido: usuario 999 inexistente.
        fact_invalida = fact_rating.copy()
        fact_invalida.loc[0, "user_id"] = 999

        with self.assertRaisesRegex(
            ValueError,
            "referencias a usuarios inexistentes",
        ):
            validate_fact_rating_user_references(
                fact_rating=fact_invalida,
                dim_user=dim_user,
            )

        # Escenario inválido: clave dimensional repetida.
        dim_duplicada = pd.DataFrame({
            "user_id": [10, 10, 20, 30],
        })

        with self.assertRaisesRegex(
            ValueError,
            "user_id duplicados",
        ):
            validate_fact_rating_user_references(
                fact_rating=fact_rating,
                dim_user=dim_duplicada,
            )

        # Escenario inválido: hecho sin usuario.
        fact_nula = fact_rating.copy()
        fact_nula["user_id"] = (
            fact_nula["user_id"].astype("Int64")
        )
        fact_nula.loc[0, "user_id"] = pd.NA

        with self.assertRaisesRegex(
            ValueError,
            "user_id nulos",
        ):
            validate_fact_rating_user_references(
                fact_rating=fact_nula,
                dim_user=dim_user,
            )
