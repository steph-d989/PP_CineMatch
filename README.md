## 🎬 CineMatch

**CineMatch** es una aplicación inteligente de recomendación de películas y series diseñada para ayudar a los usuarios a descubrir contenido acorde con sus gustos y preferencias.

La plataforma combina información sobre películas, géneros, actores, calificaciones, popularidad y disponibilidad en servicios de streaming con un **sistema de recomendación basado en datos**. A medida que el usuario interactúa con la aplicación —seleccionando géneros favoritos, calificando películas o registrando contenido visto— CineMatch construye un perfil de preferencias y genera recomendaciones cada vez más personalizadas.

El proyecto integra APIs externas como **TMDb** para obtener información cinematográfica y **Watchmode** para consultar las plataformas de streaming donde se encuentra disponible cada título.

### 🎯 Objetivo

Desarrollar un sistema capaz de responder una pregunta sencilla:

**“¿Qué película debería ver ahora?”**

CineMatch busca reducir el tiempo que los usuarios dedican a explorar catálogos y ofrecer recomendaciones relevantes basadas en sus intereses.

### 🚀 Funcionalidades principales

* Registro e inicio de sesión de usuarios.
* Selección inicial de géneros y preferencias.
* Exploración de películas y series.
* Búsqueda por título, género, actor o director.
* Visualización de sinopsis, reparto, calificación y popularidad.
* Resumen de peliculas (Duración max. 10 minutos)
* Consulta de plataformas de streaming disponibles.
* Registro de películas vistas.
* Sistema de favoritos y lista para ver.
* Calificación de películas.
* Recomendaciones personalizadas.
* Historial de interacción del usuario.
* Perfil de preferencias dinámico.
* Motor de recomendación basado en Machine Learning.

### 🤖 Sistema de recomendación

CineMatch utilizará un enfoque de **recomendación híbrida**, combinando:

**Content-Based Filtering:** analiza las características de las películas que le gustan al usuario, como géneros, actores, directores, palabras clave y características del contenido.

**Collaborative Filtering:** identifica usuarios con patrones de preferencias similares y utiliza sus interacciones para recomendar nuevos títulos.

La combinación de ambos métodos permitirá mejorar progresivamente la precisión de las recomendaciones.

### 🛠️ Tecnologías

El proyecto contempla el uso de tecnologías orientadas al desarrollo web, análisis de datos y Machine Learning:

* Python
* Pandas
* Scikit-learn
* FastAPI / Flask
* SQL
* PostgreSQL
* TMDb API
* Watchmode API
* Machine Learning
* Sistemas de recomendación
* API REST

### 📊 Flujo general

**Usuario → Preferencias → Historial de interacción → Motor de recomendación → Ranking de películas → Disponibilidad en streaming → Recomendaciones personalizadas**

### 💡 Visión del proyecto

CineMatch busca convertirse en un asistente inteligente para el descubrimiento de entretenimiento, ofreciendo una experiencia personalizada donde cada usuario encuentre contenido relevante sin tener que revisar interminables catálogos.

> **CineMatch — Your perfect movie match.**
