
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import altair as alt
import pandas as pd
import streamlit as st


# ==================================================
# CONFIGURACIÓN DEL PROYECTO
# ==================================================

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.services.mvp_data_service import (
    load_mvp_context,
    get_user_history,
)
from src.services.recommendation_service import (
    recommend_movies_for_user,
)


LOGO_PATH = ROOT / "assets" / "alternativa 1.png"
ICON_PATH = ROOT / "assets" / "alternativa 1 - icon.png"

PROFILE_CONFIG = {
    1478: {
        "name": "Clau Cárdenas",
        "photo": ROOT / "assets" / "perfil_clau.jpeg",
        "description": "Drama · Fantasía · Ciencia ficción",
    },
    52453: {
        "name": "Steph Damiani",
        "photo": ROOT / "assets" / "perfil_steph.jpeg",
        "description": "Animación · Comedia · Aventura",
    },
}

st.set_page_config(
    page_title="CineMatch",
    page_icon=str(ICON_PATH) if ICON_PATH.is_file() else "🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ==================================================
# ESTILOS CINEMATCH
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #0d111b;
        color: #f5f6fa;
    }

    .block-container {
        max-width: 1120px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    [data-testid="stMetric"] {
        background: #1b2231;
        border: 1px solid #343d50;
        border-radius: 14px;
        padding: 14px;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #f5f6fa !important;
    }

    div[data-testid="stButton"] button[kind="secondary"] {
        background: #242b3a !important;
        color: #f5f6fa !important;
        border: 1px solid #525c70 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stButton"] button[kind="primary"] {
        background: #bf3657 !important;
        color: white !important;
        border: 0 !important;
        border-radius: 10px !important;
    }

    div[data-testid="stButton"] button:hover {
        border-color: #e26b8b !important;
        color: #ffffff !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #343d50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# CARGA DE MODELOS
# ==================================================

@st.cache_resource(show_spinner="Cargando modelos...")
def get_context():
    return load_mvp_context()


try:
    context = get_context()
except Exception:
    st.error(
        "No se pudieron cargar los modelos de CineMatch. "
        "Verifica los archivos experimentales."
    )
    st.stop()


train = context["train"]
catalog = context["catalog"]


# ==================================================
# CONFIGURACIÓN TMDB
# ==================================================


def get_tmdb_credentials():
    """
    Acepta TMDb Read Access Token o API Key.
    Prioriza el token de lectura.
    """

    def read_secret(name):
        value = os.getenv(name, "").strip()

        if value:
            return value

        try:
            return str(st.secrets.get(name, "")).strip()
        except Exception:
            return ""

    token = read_secret("TMDB_READ_ACCESS_TOKEN")

    if token:
        return "token", token

    api_key = read_secret("TMDB_API_KEY")

    if api_key:
        return "api_key", api_key

    return None, None


TMDB_AUTH_MODE, TMDB_CREDENTIAL = get_tmdb_credentials()




@st.cache_data(ttl=86400, show_spinner=False)
def get_tmdb_details(tmdb_id, auth_mode, credential):

    if pd.isna(tmdb_id):
        return {"error": "Película sin ID de TMDb"}

    if not credential:
        return {"error": "No se encontró una credencial de TMDb"}

    try:
        movie_id = int(tmdb_id)

        url = (
            f"https://api.themoviedb.org/3/movie/{movie_id}"
            "?language=es-ES"
        )

        headers = {
            "Accept": "application/json",
        }

        if auth_mode == "token":
            headers["Authorization"] = (
                f"Bearer {credential}"
            )
        elif auth_mode == "api_key":
            from urllib.parse import urlencode

            url += "&" + urlencode({
                "api_key": credential
            })
        else:
            return {"error": "Autenticación no compatible"}

        request = Request(url, headers=headers)

        with urlopen(request, timeout=6) as response:
            data = json.load(response)

        poster_path = data.get("poster_path")

        return {
            "poster": (
                "https://image.tmdb.org/t/p/w342"
                + poster_path
                if poster_path else None
            ),
            "overview": data.get("overview") or "",
            "error": None,
        }

    except HTTPError as exc:
        return {"error": f"TMDb HTTP {exc.code}"}

    except (URLError, TimeoutError, OSError) as exc:
        return {
            "error": f"Error de conexión: {type(exc).__name__}"
        }

    except (ValueError, TypeError) as exc:
        return {
            "error": f"Respuesta inválida: {type(exc).__name__}"
        }


# ==================================================
# FUNCIONES VISUALES
# ==================================================

def render_logo(width=230):
    if LOGO_PATH.is_file():
        st.image(str(LOGO_PATH), width=width)
    else:
        st.markdown("## CineMatch")


def genres_as_list(value):

    if isinstance(value, (list, tuple)):
        return [
            str(genre).strip()
            for genre in value
            if str(genre).strip()
        ]

    if not isinstance(value, str):
        return []

    return [
        genre.strip()
        for genre in value.split("|")
        if genre.strip()
        and genre.strip() != "(no genres listed)"
    ]


def calculate_dna(user_id):

    positives = train.loc[
        (train["user_id"] == user_id)
        & (train["rating"] >= 4.0),
        ["movie_id"],
    ]

    joined = positives.merge(
        catalog[["movie_id", "genres"]],
        on="movie_id",
        how="left",
        validate="many_to_one",
    )

    counts = {}

    for value in joined["genres"]:
        for genre in genres_as_list(value):
            counts[genre] = counts.get(genre, 0) + 1

    total = sum(counts.values())

    if total == 0:
        return pd.DataFrame(
            columns=["genre", "percentage"]
        )

    rows = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0]),
    )

    dna = pd.DataFrame([
        {
            "genre": genre,
            "percentage": count * 100 / total,
        }
        for genre, count in rows
    ])

    top = dna.head(5).copy()
    others = float(dna.iloc[5:]["percentage"].sum())

    if others > 0:
        top = pd.concat(
            [
                top,
                pd.DataFrame([{
                    "genre": "Otros",
                    "percentage": others,
                }]),
            ],
            ignore_index=True,
        )

    return top


def render_dna(user_id):

    dna = calculate_dna(user_id)

    if dna.empty:
        st.info("Todavía no hay géneros suficientes.")
        return

    colors = [
        "#bc6bdb",
        "#a268d5",
        "#d94e83",
        "#e87d9d",
        "#eea9bb",
        "#666b80",
    ]

    dna = dna.copy()
    dna["color"] = colors[:len(dna)]
    dna["label"] = dna["percentage"].map(
        lambda value: f"{value:.1f}%"
    )

    genre_order = dna["genre"].tolist()

    bars = (
        alt.Chart(dna)
        .mark_bar(
            size=17,
            cornerRadiusEnd=8,
        )
        .encode(
            y=alt.Y(
                "genre:N",
                sort=genre_order,
                title=None,
                axis=alt.Axis(
                    labelColor="#eef0f8",
                    labelFontSize=12,
                    ticks=False,
                    domain=False,
                ),
            ),
            x=alt.X(
                "percentage:Q",
                scale=alt.Scale(domain=[0, 100]),
                title=None,
                axis=None,
            ),
            color=alt.Color(
                "color:N",
                scale=None,
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("genre:N", title="Género"),
                alt.Tooltip(
                    "percentage:Q",
                    title="Porcentaje",
                    format=".1f",
                ),
            ],
        )
    )

    labels = (
        alt.Chart(dna)
        .mark_text(
            align="left",
            baseline="middle",
            dx=7,
            color="#e8eaf4",
            fontSize=12,
        )
        .encode(
            y=alt.Y(
                "genre:N",
                sort=genre_order,
            ),
            x="percentage:Q",
            text="label:N",
        )
    )

    chart = (
        (bars + labels)
        .properties(height=210)
        .configure_view(stroke=None)
        .configure(background="#161d2c")
    )

    st.altair_chart(
        chart,
        use_container_width=True,
    )

    st.caption(
        "Porcentaje de etiquetas de género entre "
        "películas valoradas con 4 estrellas o más."
    )


def render_movie_row(movie, show_rating=False):

    details = get_tmdb_details(
        movie.get("tmdb_id"),
        TMDB_AUTH_MODE,
        TMDB_CREDENTIAL,
    )

    with st.container(border=True):

        poster_col, info_col = st.columns(
            [1, 3],
            gap="medium",
        )

        with poster_col:

            if details.get("poster"):
                st.image(
                    details["poster"],
                    use_container_width=True,
                )
            else:
                st.markdown(
                    """
                    <div style="
                        height:140px;
                        background:linear-gradient(
                            150deg,#34284e,#171d2d
                        );
                        border-radius:10px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        font-size:40px;
                    ">🎞️</div>
                    """,
                    unsafe_allow_html=True,
                )

        with info_col:

            st.markdown(
                f"**{movie['title']}**"
            )

            genres = genres_as_list(
                movie.get("genres")
            )

            st.caption(
                " · ".join(genres[:3])
                or "Géneros no disponibles"
            )

            if show_rating:
                st.write(
                    f"⭐ {float(movie['rating']):.1f} / 5"
                )

            else:
                overview = details.get("overview", "")

                if overview:
                    if len(overview) > 170:
                        overview = overview[:167] + "..."

                    st.caption(overview)
                else:
                    st.caption(
                        "Información disponible en TMDb "
                        "cuando se configure el acceso."
                    )


def render_movie_section(data, show_rating=False):

    if data.empty:
        st.info("No hay películas para mostrar.")
        return

    columns = st.columns(3, gap="medium")

    for column, (_, movie) in zip(
        columns,
        data.head(3).iterrows(),
    ):
        with column:
            render_movie_row(
                movie.to_dict(),
                show_rating=show_rating,
            )


# ==================================================
# SESIÓN SIMULADA
# ==================================================

if "active_user" not in st.session_state:
    st.session_state.active_user = None

profiles = [1478, 52453]

if not set(profiles).issubset(
    set(train["user_id"])
):
    st.error(
        "Los usuarios de demostración no existen en Train."
    )
    st.stop()


# ==================================================
# VISTA 1: SELECCIÓN DE PERFIL
# ==================================================

if st.session_state.active_user is None:

    left, center, right = st.columns([1, 2, 1])

    with center:
        render_logo(width=290)

    st.markdown(
        "<h2 style='text-align:center;'>"
        "¿Quién está viendo?"
        "</h2>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center;color:#adb6c7;'>"
        "Selecciona tu perfil y descubre "
        "tu universo cinematográfico."
        "</p>",
        unsafe_allow_html=True,
    )

    # Columnas laterales para centrar las tarjetas
    spacer_left, col_clau, col_steph, spacer_right = st.columns(
        [1, 3, 3, 1],
        gap="medium",
    )

    columns = [col_clau, col_steph]

    for column, user_id in zip(columns, profiles):

        profile = PROFILE_CONFIG[user_id]

        with column:
            with st.container(border=True):

                photo_left, photo_center, photo_right = (
                    st.columns([1, 2, 1])
                )

                with photo_center:
                    if profile["photo"].is_file():
                        st.image(
                            str(profile["photo"]),
                            use_container_width=True,
                        )
                    else:
                        st.markdown("## 👤")

                st.markdown(
                    f"### {profile['name']}"
                )

                st.caption(
                    f"MovieLens #{user_id}"
                )

                st.caption(
                    profile["description"]
                )

                if st.button(
                    "Entrar a CineMatch",
                    key=f"login_{user_id}",
                    type="primary",
                    use_container_width=True,
                ):
                    st.session_state.active_user = user_id
                    st.rerun()

    st.stop()


# ==================================================
# VISTA 2: DASHBOARD DEL USUARIO
# ==================================================

user_id = st.session_state.active_user
profile = PROFILE_CONFIG[user_id]

user_train = train.loc[
    train["user_id"] == user_id
].copy()

positive_count = int(
    (user_train["rating"] >= 4.0).sum()
)

average_rating = float(
    user_train["rating"].mean()
)


# HEADER
logo_col, button_col = st.columns(
    [4, 1],
    vertical_alignment="center",
)

with logo_col:
    render_logo(width=220)

with button_col:
    if st.button(
        "← Cambiar perfil",
        key="change_profile",
        type="secondary",
        use_container_width=True,
    ):
        st.session_state.active_user = None
        st.rerun()

st.markdown(
    f"## ¡Bienvenida, {profile['name']}!"
)

st.caption(
    f"Tu universo cinematográfico · MovieLens #{user_id}"
)


# INDICADORES
c1, c2, c3 = st.columns(3)

c1.metric(
    "Películas valoradas",
    len(user_train),
)

c2.metric(
    "Valoraciones positivas",
    positive_count,
)

c3.metric(
    "Calificación promedio",
    f"{average_rating:.2f} / 5",
)

st.divider()



# ======================================
# ADN + CARRUSEL DE FAVORITAS
# ======================================

st.subheader("🧬 Mi ADN cinematográfico")
st.caption(
    "Descubre tus géneros favoritos y "
    "las películas que más has disfrutado."
)


dna_col, favorites_col = st.columns(
    2,
    gap="medium",
    vertical_alignment="top",
    border=False,
)

# --------------------------------------
# IZQUIERDA: ADN CINEMATOGRÁFICO
# --------------------------------------

with dna_col:
    with st.container(
        border=True,
        height=320,
    ):
        st.markdown("**Tus géneros favoritos**")
        render_dna(user_id)


# --------------------------------------
# DERECHA: CARRUSEL DE 5 ESTRELLAS
# --------------------------------------

# DERECHA: CARRUSEL DE 5 ESTRELLAS
with favorites_col:
    with st.container(
        border=True,
        height=320,
    ):

        st.markdown("**⭐ Tus películas favoritas**")

        favorite_movies = get_user_history(
            user_id=user_id,
            train=train,
            catalog=catalog,
            n=len(user_train),
        )

        favorite_movies = favorite_movies.loc[
            favorite_movies["rating"] == 5.0
        ].reset_index(drop=True)

        if favorite_movies.empty:
            st.info(
                "Este perfil todavía no tiene "
                "películas calificadas con 5 estrellas."
            )
        else:
            # Reiniciar el carrusel al cambiar de usuario.
            if (
                st.session_state.get("favorite_user")
                != user_id
            ):
                st.session_state.favorite_user = user_id
                st.session_state.favorite_index = 0

            total = len(favorite_movies)

            idx = st.session_state.favorite_index % total
            movie = favorite_movies.iloc[idx]
            
            details = get_tmdb_details(
                movie["tmdb_id"],
                TMDB_AUTH_MODE,
                TMDB_CREDENTIAL,
            )

            poster_col, info_col = st.columns(
                [1, 1.5],
                vertical_alignment="center",
                gap="medium",
            )

            with poster_col:
                if details.get("poster"):
                    st.image(
                        details["poster"],
                        use_container_width=True,
                    )
                else:
                    st.markdown("### 🎞️")
                    st.caption("Póster pendiente")


            with info_col:

                # Espacio fijo para el título
                st.markdown(
                    f"""
                    <div style="
                        height:65px;
                        overflow:hidden;
                        font-weight:600;
                        font-size:14px;
                        line-height:1.4;
                        color:#f5f6fa;
                    ">
                        {movie['title']}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Espacio fijo para los géneros
                genres_text = " · ".join(
                    genres_as_list(movie["genres"])[:3]
                )

                st.markdown(
                    f"""
                    <div style="
                        height:45px;
                        overflow:hidden;
                        font-size:12px;
                        color:#aab3c5;
                    ">
                        {genres_text}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.write("⭐ 5.0 / 5")

            # Espacio fijo antes de los controles
            st.markdown(
                "<div style='height:8px;'></div>",
                unsafe_allow_html=True,
            )

            previous, counter, following = st.columns(
                [1, 2, 1],
                vertical_alignment="center",
            )

            with previous:
                if st.button(
                    "❮",
                    key="favorite_previous",
                    use_container_width=True,
                ):
                    st.session_state.favorite_index = (
                        idx - 1
                    ) % total
                    st.rerun()

            with counter:
                st.markdown(
                    f"<p style='text-align:center;"
                    f"color:#c5cbd7;'>"
                    f"{idx + 1} / {total}"
                    f"</p>",
                    unsafe_allow_html=True,
                )

            with following:
                if st.button(
                    "❯",
                    key="favorite_next",
                    use_container_width=True,
                ):
                    st.session_state.favorite_index = (
                        idx + 1
                    ) % total
                    st.rerun()

st.divider()


# HISTORIAL
st.subheader("⭐ Tus últimas valoraciones")

history = get_user_history(
    user_id=user_id,
    train=train,
    catalog=catalog,
    n=3,
)

render_movie_section(
    history,
    show_rating=True,
)

with st.expander("Ver todas mis valoraciones"):

    full_history = get_user_history(
        user_id=user_id,
        train=train,
        catalog=catalog,
        n=len(user_train),
    )

    st.dataframe(
        full_history[
            ["title", "rating", "genres"]
        ],
        hide_index=True,
        use_container_width=True,
    )

st.divider()


# RECOMENDACIONES COLABORATIVAS
st.subheader(
    "👥 A personas con gustos parecidos a los tuyos les gustó..."
)

collaborative = recommend_movies_for_user(
    user_id=user_id,
    algorithm="collaborative",
    n=3,
    train=train,
    catalog=catalog,
    content_model=context["content_model"],
    collaborative_model=context["collaborative_model"],
)

render_movie_section(collaborative)

st.divider()


# RECOMENDACIONES BASADAS EN CONTENIDO
st.subheader(
    "✨ Basado en tus preferencias cinematográficas"
)

content = recommend_movies_for_user(
    user_id=user_id,
    algorithm="content",
    n=3,
    train=train,
    catalog=catalog,
    content_model=context["content_model"],
    collaborative_model=context["collaborative_model"],
)

render_movie_section(content)

st.divider()

