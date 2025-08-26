# Análisis de Impedancia Acústica de Quenas

Este proyecto contiene scripts de Python para analizar mediciones de impedancia acústica de instrumentos de viento, con un enfoque en quenas y sus componentes.

## `knot.py`: Análisis de Resonancia vs. Diámetro de Nudo

Este script analiza cómo una modificación física (el diámetro de un agujero o "nudo" terminal) afecta las propiedades acústicas de un tubo resonante.

-   **Entrada:** Archivos `.csv` con mediciones de impedancia, donde cada archivo corresponde a un diámetro de nudo diferente.
-   **Proceso:**
    -   Calcula la admitancia (inverso de la impedancia).
    -   Detecta las frecuencias de resonancia (picos de admitancia), identificando la fundamental (`f0`) y el primer sobretono (`f1`).
    -   Calcula la **inharmonicidad** en *cents*, que mide cuánto se desvía `f1` de ser una octava perfecta (`2 * f0`).
-   **Salida:**
    -   Gráficos individuales de admitancia para cada medición.
    -   Un gráfico de resumen que muestra la evolución de `f0`, `f1` y la inharmonicidad en función del diámetro del nudo.

## `Impedancias_quenas.py`: Análisis Comparativo de Quenas

Este es un script más completo diseñado para realizar un análisis acústico detallado y comparativo de un conjunto de quenas.

-   **Entrada:** Mediciones de impedancia para múltiples notas (`G`, `A`, `B`, etc.) de varias quenas. Soporta archivos `.dat` y `.mat`.
-   **Proceso:**
    -   Utiliza un método de detección de picos robusto (con filtro Savitzky-Golay) para encontrar las resonancias de cada nota.
    -   Calcula dos métricas clave para evaluar el rendimiento del instrumento:
        1.  **Desviación (Afinación):** La diferencia en *cents* entre la frecuencia fundamental de la nota y su valor teórico en la escala temperada.
        2.  **Inharmonicidad:** La diferencia en *cents* entre el primer sobretono y la octava perfecta de la fundamental.
-   **Salida:** Un conjunto exhaustivo de gráficos guardados en carpetas separadas (`images/Set1`, `images/Set2`), incluyendo:
    -   **Análisis Individual por Quena:** Curvas de admitancia, frecuencias de los picos y gráficos de desviación/inharmonicidad.
    -   **Análisis Comparativo:** Gráficos que superponen los resultados de todas las quenas para comparar su afinación y calidad armónica.
    -   **Análisis Estadístico:** Gráficos de media y desviación estándar para evaluar el comportamiento promedio del conjunto de instrumentos.