# Trabajo 2: Fusión de Perspectivas - Registro de Imágenes y Medición del Mundo Real

## Descripción del Proyecto

Este proyecto implementa un sistema completo de registro de imágenes para fusionar múltiples vistas del mismo escenario en una vista panorámica coherente. Además, incluye un sistema de calibración y medición que permite estimar dimensiones reales de objetos en la escena.

### Objetivos

1. **Validar el pipeline** con imágenes sintéticas (transformaciones conocidas)
2. **Registrar y fusionar** tres imágenes del comedor en una vista unificada
3. **Calibrar el sistema** usando objetos de referencia con dimensiones conocidas
4. **Medir dimensiones** de elementos en la escena fusionada

## Estructura del Proyecto

```
proyecto-registro-imagenes/
├── README.md                   # Este archivo
├── requirements.txt            # Dependencias del proyecto
├── data/
│   ├── original/              # Imágenes originales del comedor
│   └── synthetic/             # Imágenes sintéticas para validación
├── src/
│   ├── feature_detection.py   # Detectores de características (SIFT, ORB, AKAZE)
│   ├── matching.py            # Emparejamiento robusto de características
│   ├── registration.py        # Estimación de homografías y fusión
│   ├── measurement.py         # Calibración y medición
│   └── utils.py               # Funciones auxiliares y visualización
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb    # Análisis exploratorio
│   ├── 02_synthetic_validation.ipynb    # Validación con imágenes sintéticas
│   └── 03_main_pipeline.ipynb           # Pipeline principal de fusión
├── results/
│   ├── figures/               # Gráficas y visualizaciones
│   └── measurements/          # Resultados de mediciones
└── tests/                     # Pruebas unitarias (opcional)
```

## Instalación

### Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### Pasos de Instalación

1. Clonar el repositorio:
```bash
git clone <url-del-repositorio>
cd Trabajo_02_Quartz_Meditativo
```

2. Crear un entorno virtual (recomendado):
```bash
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate
```

3. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Análisis Exploratorio

```bash
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

Este notebook permite:
- Visualizar las imágenes originales
- Explorar diferentes detectores de características
- Analizar la distribución de keypoints

### 2. Validación con Imágenes Sintéticas

```bash
jupyter notebook notebooks/02_synthetic_validation.ipynb
```

Este notebook incluye:
- Generación de imágenes sintéticas con transformaciones conocidas
- Validación del pipeline de registro
- Cálculo de métricas de error (RMSE, error angular)

### 3. Pipeline Principal de Fusión

```bash
jupyter notebook notebooks/03_main_pipeline.ipynb
```

Este notebook implementa:
- Detección de características en las tres imágenes
- Matching robusto entre pares de imágenes
- Estimación de homografías con RANSAC
- Fusión con técnicas de blending
- Calibración y medición de objetos

## Metodología

### Detección de Características

Se implementan tres detectores diferentes:
- **SIFT** (Scale-Invariant Feature Transform): Robusto a cambios de escala y rotación
- **ORB** (Oriented FAST and Rotated BRIEF): Rápido y libre de patentes
- **AKAZE** (Accelerated-KAZE): Buen balance entre velocidad y precisión

### Emparejamiento de Características

- Matching usando BFMatcher o FLANN
- Aplicación de ratio test (Lowe's ratio)
- Filtrado de outliers con RANSAC

### Estimación de Homografías

- Uso de RANSAC para estimación robusta
- Validación de la calidad de las transformaciones
- Optimización de parámetros

### Fusión de Imágenes

- Warping de imágenes según homografías estimadas
- Técnicas de blending para transiciones suaves:
  - Feathering
  - Multi-band blending (opcional)

### Calibración y Medición

Objetos de referencia con dimensiones conocidas:
- Cuadro de la Virgen de Guadalupe: 117 cm de altura
- Mesa: 161.1 cm de ancho

## Resultados

Los resultados se almacenan en el directorio `results/`:
- `figures/`: Visualizaciones del proceso paso a paso
- `measurements/`: Tabla con mediciones estimadas y análisis de incertidumbre

## Referencias

1. Lowe, D. G. (2004). "Distinctive Image Features from Scale-Invariant Keypoints". International Journal of Computer Vision.
2. Hartley, R., & Zisserman, A. (2003). "Multiple View Geometry in Computer Vision". Cambridge University Press.
3. Brown, M., & Lowe, D. G. (2007). "Automatic Panoramic Image Stitching using Invariant Features". International Journal of Computer Vision.
4. Rublee, E., et al. (2011). "ORB: An efficient alternative to SIFT or SURF". IEEE International Conference on Computer Vision.
5. Alcantarilla, P. F., et al. (2013). "Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces". IEEE Trans. Pattern Analysis and Machine Intelligence.

## Uso Avanzado

### Herramienta Interactiva de Medición

Para usar la herramienta interactiva de medición en el notebook:

```python
from src.measurement import InteractiveMeasurementTool, Calibrator

# Cargar panorama y calibrador
tool = InteractiveMeasurementTool(panorama, calibrator)
tool.launch_interactive_tool()

# Click izquierdo para seleccionar puntos (2 puntos por medición)
# Click derecho para limpiar selección
```

### Personalización de Parámetros

#### Detección de Características

```python
# Ajustar parámetros de SIFT
detector = FeatureDetector(DetectorType.SIFT, 
                          nfeatures=0,           # 0 = sin límite
                          contrastThreshold=0.04, # Umbral de contraste
                          edgeThreshold=10)       # Umbral de bordes
```

#### Matching

```python
# Ajustar ratio test de Lowe
matcher = FeatureMatcher(MatcherType.BF, descriptor_type='SIFT')
matches = matcher.match_robust(desc1, desc2, 
                               ratio=0.75,        # Típicamente 0.7-0.8
                               use_symmetry=True) # Activar symmetry test
```

#### RANSAC

```python
# Ajustar parámetros de RANSAC
H, mask = estimate_homography(points1, points2,
                              ransac_threshold=5.0,  # Umbral en píxeles
                              confidence=0.995,       # Nivel de confianza
                              max_iters=2000)         # Máximo de iteraciones
```

#### Blending

```python
# Diferentes métodos de blending
stitcher = ImageStitcher(blend_method='simple')    # Promedio simple
stitcher = ImageStitcher(blend_method='feather')   # Transiciones suaves (recomendado)
stitcher = ImageStitcher(blend_method='multiband') # Pirámide de Laplace (mejor calidad)
```

## Solución de Problemas

### Problema: No se encuentran suficientes matches

**Solución**: 
- Ajustar el ratio test (aumentar de 0.75 a 0.8)
- Cambiar de detector (probar ORB o AKAZE)
- Verificar que las imágenes tengan suficiente solapamiento

### Problema: Las homografías estimadas son incorrectas

**Solución**:
- Aumentar el umbral de RANSAC
- Verificar que las correspondencias sean correctas
- Asegurarse de que haya suficientes inliers

### Problema: El panorama tiene costuras visibles

**Solución**:
- Usar `blend_method='feather'` o `'multiband'`
- Ajustar la exposición de las imágenes antes de fusionar
- Verificar que las homografías sean precisas

### Problema: Las mediciones tienen alta incertidumbre

**Solución**:
- Usar objetos de referencia más grandes
- Agregar más objetos de referencia para calibración
- Medir objetos perpendiculares al plano de la cámara

## Estructura del Código

### Módulos Principales

- **`utils.py`**: Funciones auxiliares para carga, visualización y procesamiento de imágenes
- **`feature_detection.py`**: Implementación de detectores SIFT, ORB, AKAZE
- **`matching.py`**: Estrategias de matching robusto con ratio test y symmetry test
- **`registration.py`**: Estimación de homografías, RANSAC, y técnicas de blending
- **`measurement.py`**: Calibración y herramientas de medición interactivas
- **`synthetic_generator.py`**: Generación de imágenes sintéticas para validación

### Notebooks

1. **`01_exploratory_analysis.ipynb`**: Análisis exploratorio de las imágenes
2. **`02_synthetic_validation.ipynb`**: Validación del pipeline con imágenes sintéticas
3. **`03_main_pipeline.ipynb`**: Pipeline completo de fusión, calibración y medición

## Resultados Esperados

Después de ejecutar el pipeline completo, obtendrá:

1. **Panorama fusionado** en `results/panorama_comedor.png`
2. **Visualizaciones** en `results/figures/`:
   - Imágenes originales
   - Comparación de detectores
   - Correspondencias entre imágenes
   - Distribución de keypoints
   - Dataset sintético
   - Errores de validación
   - Panorama final
   - Reporte de mediciones
3. **Mediciones** en `results/measurements/mediciones_comedor.csv`

## Contribución Individual

- **Sebastián Palacio Betancur**: Implementación del pipeline de fusión, calibración y medición.
- **Juan Manuel Sanchez Restrepo**: Implementación del pipeline de fusión, calibración y medición.
- **Henrry Uribe Cabrera Ordonez**: Implementación del pipeline de fusión, calibración y medición.
= **Laura Sanin Colorado**: Implementación del pipeline de fusión, calibración y medición.

Este proyecto es parte del curso de Visión por Computador.
Profesor: Juan David Ospina Arango
Monitor: Andrés Mauricio Zapata


