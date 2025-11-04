# Instrucciones para el Usuario

## ✅ Lo que ya está implementado

Se ha completado la implementación de todo el código y estructura del proyecto:

### Estructura del Proyecto
```
Trabajo_02_Quartz_Meditativo/
├── README.md                      ✅ Completo
├── requirements.txt               ✅ Completo
├── .gitignore                     ✅ Completo
├── data/
│   ├── original/                  ✅ Imágenes movidas aquí
│   │   ├── IMG01.jpg
│   │   ├── IMG02.jpg
│   │   └── IMG03.jpg
│   └── synthetic/                 ✅ Para imágenes sintéticas
├── src/                           ✅ Todos los módulos implementados
│   ├── __init__.py
│   ├── utils.py
│   ├── feature_detection.py
│   ├── matching.py
│   ├── registration.py
│   ├── measurement.py
│   └── synthetic_generator.py
├── notebooks/                     ✅ Todos los notebooks creados
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_synthetic_validation.ipynb
│   └── 03_main_pipeline.ipynb
├── results/
│   ├── figures/                   📊 Se generará al ejecutar
│   └── measurements/              📊 Se generará al ejecutar
└── tests/                         📁 Opcional

```

### Módulos Python Implementados

1. **`utils.py`** ✅
   - Funciones de carga/guardado de imágenes
   - Visualización de imágenes y keypoints
   - Dibujo de matches
   - Cálculo de errores
   - Creación de canvas para panoramas

2. **`feature_detection.py`** ✅
   - Detectores SIFT, ORB, AKAZE
   - Comparación de detectores
   - Análisis de distribución de keypoints
   - Filtrado de keypoints

3. **`matching.py`** ✅
   - Brute Force y FLANN matchers
   - Ratio test de Lowe
   - Symmetry test
   - Estadísticas de matches

4. **`registration.py`** ✅
   - Estimación de homografías con RANSAC
   - Validación de homografías
   - Warping de imágenes
   - Blending: simple, feather, multiband
   - Clase ImageStitcher completa

5. **`measurement.py`** ✅
   - Clase Calibrator
   - Herramienta interactiva de medición
   - Análisis de incertidumbre
   - Reportes visuales

6. **`synthetic_generator.py`** ✅
   - Generación de imágenes sintéticas
   - Aplicación de transformaciones conocidas
   - Visualización de datasets

### Notebooks Creados

1. **`01_exploratory_analysis.ipynb`** ✅
   - Carga de imágenes
   - Comparación de detectores
   - Análisis de correspondencias
   - Distribución de keypoints

2. **`02_synthetic_validation.ipynb`** ✅
   - Generación de dataset sintético
   - Validación del pipeline
   - Cálculo de errores (RMSE, Frobenius)
   - Análisis de precisión

3. **`03_main_pipeline.ipynb`** ✅
   - Pipeline completo de fusión
   - Calibración con objetos de referencia
   - Mediciones de elementos
   - Análisis de incertidumbre
   - Tabla de resultados

## 📋 Lo que DEBES hacer

### 1. Instalar Dependencias

```bash
# Crear entorno virtual (recomendado)
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar los Notebooks en Orden

#### Paso 1: Análisis Exploratorio
```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```
- Ejecutar todas las celdas
- Revisar los detectores y sus resultados
- Verificar que hay suficientes matches

#### Paso 2: Validación Sintética
```bash
jupyter notebook notebooks/02_synthetic_validation.ipynb
```
- Ejecutar todas las celdas
- Verificar que los errores sean bajos (RMSE < 5 píxeles)
- Confirmar que el pipeline funciona correctamente

#### Paso 3: Pipeline Principal (IMPORTANTE)
```bash
jupyter notebook notebooks/03_main_pipeline.ipynb
```

**ESTE ES EL NOTEBOOK MÁS IMPORTANTE**

Debes actualizar las coordenadas de los puntos manualmente:

##### a) Calibración (Sección 5)

**Cuadro de la Virgen de Guadalupe (117 cm de altura):**
```python
# Buscar en el panorama y actualizar estas coordenadas:
cuadro_punto_superior = (500, 200)  # ← ACTUALIZAR con coordenadas reales
cuadro_punto_inferior = (500, 800)  # ← ACTUALIZAR con coordenadas reales
```

**Mesa (161.1 cm de ancho):**
```python
# Buscar en el panorama y actualizar estas coordenadas:
mesa_punto_izquierdo = (300, 1000)  # ← ACTUALIZAR con coordenadas reales
mesa_punto_derecho = (1500, 1000)   # ← ACTUALIZAR con coordenadas reales
```

##### b) Mediciones (Sección 6)

Actualizar coordenadas para las 5 mediciones:
1. Ancho del cuadro
2. Largo de la mesa
3. Altura de ventana
4. Ancho de silla
5. Altura de planta

**Opciones para obtener coordenadas:**

**Opción A: Herramienta Interactiva (Recomendada)**
- Descomentar el código en la celda "Herramienta Interactiva de Medición"
- Ejecutar la celda
- Click izquierdo para seleccionar 2 puntos
- Las mediciones se guardan automáticamente

**Opción B: Usar visor de imágenes**
- Abrir `results/panorama_comedor.png` en Photoshop, GIMP, Paint, etc.
- Mover el cursor sobre los puntos de interés
- Anotar las coordenadas (x, y)
- Actualizar en el código

### 3. Verificar Resultados

Después de ejecutar todos los notebooks, deberías tener:

```
results/
├── panorama_comedor.png          ← Imagen panorámica fusionada
├── figures/
│   ├── 01_original_images.png
│   ├── 02_detector_comparison.png
│   ├── 03_matches_img1_to_img2.png
│   ├── 03_matches_img2_to_img3.png
│   ├── 04_keypoint_distribution.png
│   ├── 05_synthetic_dataset.png
│   ├── 06_validation_errors.png
│   ├── 07_corner_errors.png
│   ├── 08_panorama_final.png
│   └── 09_measurement_report.png
└── measurements/
    └── mediciones_comedor.csv     ← Tabla con todas las mediciones
```

### 4. Crear el Blog Post (Reporte Técnico)

Debes publicar un reporte técnico en una de estas plataformas:
- RPubs
- GitHub Pages
- Medium
- Observable
- Cualquier plataforma de blogging técnico

**Estructura del reporte (según especificaciones):**

1. **Introducción**
   - Contexto del problema
   - Motivación
   - Objetivos del trabajo

2. **Marco Teórico**
   - Detección de características (SIFT, ORB, AKAZE)
   - Emparejamiento robusto (ratio test, RANSAC)
   - Homografías y transformaciones geométricas
   - Técnicas de blending
   - Calibración de cámaras
   - **Incluir al menos 5 referencias académicas**

3. **Metodología**
   - Pipeline implementado (con diagramas)
   - Decisiones técnicas y justificación
   - Parámetros utilizados

4. **Experimentos y Resultados**
   - Validación con imágenes sintéticas (usar figuras 05, 06, 07)
   - Proceso paso a paso (usar figuras 01-04)
   - Panorama final (figura 08)
   - Tabla de mediciones (mediciones_comedor.csv)

5. **Análisis y Discusión**
   - Comparación de detectores
   - Análisis de errores
   - Limitaciones
   - Posibles mejoras

6. **Conclusiones**
   - Resumen de logros
   - Aprendizajes

7. **Referencias**
   - Mínimo 5 fuentes académicas

8. **Contribución Individual**
   - Descripción de tareas por cada integrante

### 5. Referencias Sugeridas

1. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints". *International Journal of Computer Vision*.

2. Hartley, R., & Zisserman, A. (2003). "Multiple View Geometry in Computer Vision". Cambridge University Press.

3. Brown, M., & Lowe, D. G. (2007). "Automatic Panoramic Image Stitching using Invariant Features". *International Journal of Computer Vision*.

4. Rublee, E., et al. (2011). "ORB: An efficient alternative to SIFT or SURF". *IEEE International Conference on Computer Vision*.

5. Alcantarilla, P. F., et al. (2013). "Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces". *IEEE Trans. Pattern Analysis and Machine Intelligence*.

6. Burt, P. J., & Adelson, E. H. (1983). "A Multiresolution Spline With Application to Image Mosaics". *ACM Transactions on Graphics*.

7. Fischler, M. A., & Bolles, R. C. (1981). "Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography". *Communications of the ACM*.

## 🎯 Checklist Final

- [ ] Instalación de dependencias completada
- [ ] Notebook 01 ejecutado exitosamente
- [ ] Notebook 02 ejecutado exitosamente
- [ ] Notebook 03 ejecutado con coordenadas reales actualizadas
- [ ] Panorama fusionado generado
- [ ] Todas las visualizaciones en `results/figures/`
- [ ] Tabla de mediciones en CSV
- [ ] Blog post redactado con todas las secciones
- [ ] Blog post publicado en plataforma elegida
- [ ] Sección de contribución individual completada
- [ ] Al menos 5 referencias académicas citadas
- [ ] Repositorio GitHub actualizado

## ⚠️ Notas Importantes

1. **Las coordenadas en el notebook 03 son ejemplos**. Debes actualizarlas con las coordenadas reales de tu panorama.

2. **La herramienta interactiva** es la forma más fácil de obtener mediciones precisas.

3. **Los objetos de referencia** son:
   - Cuadro Virgen de Guadalupe: 117 cm de altura
   - Mesa: 161.1 cm de ancho

4. **Debes medir al menos 5 elementos adicionales**:
   - Los 2 objetos de referencia (ancho del cuadro, largo de la mesa)
   - 3 elementos más (ventanas, sillas, plantas, puertas, etc.)

5. **El blog post es parte fundamental de la entrega**. Dedica tiempo a documentar bien el proceso y los resultados.

