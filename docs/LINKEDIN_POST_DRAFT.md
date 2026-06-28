# LinkedIn Post Draft

He estado trabajando en un prototipo de scanner visual para cartas TCG. La idea es sencilla de explicar, pero interesante a nivel técnico: apuntar con la cámara a una carta, detectar su región, normalizar la imagen y buscarla por similitud visual contra un índice local.

El pipeline tiene dos modelos principales:

1. YOLO para localizar la carta completa en una foto o vídeo.
2. SigLIP 2 para convertir la imagen normalizada de la carta en un embedding visual.

El flujo actual es:

- importo metadatos e imágenes oficiales de Riftbound;
- preproceso cada carta a una entrada estable de `384 x 384`;
- genero embeddings con SigLIP 2;
- guardo esos vectores y metadatos en LanceDB;
- cuando llega una foto del usuario, detecto la carta, la recorto, la normalizo, genero su embedding y busco los vecinos más cercanos.

Elegí LanceDB porque para este prototipo permite tener búsqueda vectorial local, rápida y fácil de reproducir. No necesitaba empezar con una base de datos vectorial distribuida; primero quería validar precisión, latencia y ergonomía del scanner.

Una parte importante fue separar dos problemas:

- detectar dónde está la carta;
- identificar qué carta es.

Para detección entrené un modelo YOLO single-class (`card`) con un dataset universal de cartas TCG. Mezclé fuentes de MTG, Pokémon, Grand Archive y otros TCGs para que el detector aprendiera “forma de carta” y no solo un arte o set concreto. En el prototipo actual todavía falta añadir fotos reales de Riftbound al entrenamiento, pero ya funciona lo bastante bien para validar la experiencia.

En la UI hice un modo live camera: el navegador envía frames ligeros al backend, YOLO devuelve la caja y se pinta el borde sobre el vídeo. Cuando la detección se mantiene estable, se captura un frame y se lanza el pipeline visual completo.

La parte de rendimiento fue curiosa: el embedding caliente está alrededor de decenas de milisegundos en mi entorno, así que el cuello de botella no era solo el modelo visual. Algunas mejoras vinieron de:

- mantener modelos cargados;
- comprimir solo los frames de detección live;
- no bloquear el reconocimiento esperando precios;
- separar la búsqueda visual de la consulta de precio.

El prototipo también integra precios vía PriceCharting como proveedor inicial. Para un producto real, el siguiente paso sería histórico de precios, cuentas de usuario, colecciones por TCG y set, y métricas de inversión/progreso.

Todavía es un MVP, pero ya demuestra algo potente: se puede construir un scanner de cartas rápido combinando detección, embeddings visuales y búsqueda vectorial sin depender de OCR ni de reglas frágiles sobre texto impreso.

Próximo paso: limpiar el repo público, documentar bien la arquitectura y seguir evolucionando la parte de producto en privado.
