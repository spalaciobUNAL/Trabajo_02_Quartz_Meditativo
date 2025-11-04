"""
Paquete de registro de imágenes y medición.
"""

__version__ = '1.0.0'
__author__ = 'Equipo de Visión por Computador'

# Importar módulos principales
from . import utils
from . import feature_detection
from . import matching
from . import registration
from . import measurement
from . import synthetic_generator

__all__ = [
    'utils',
    'feature_detection',
    'matching',
    'registration',
    'measurement',
    'synthetic_generator'
]


