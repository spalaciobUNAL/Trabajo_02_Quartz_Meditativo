"""
Módulo de detección de características para registro de imágenes.
Implementa diferentes detectores: SIFT, ORB, AKAZE.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from enum import Enum


class DetectorType(Enum):
    """Tipos de detectores de características disponibles."""
    SIFT = "SIFT"
    ORB = "ORB"
    AKAZE = "AKAZE"


class FeatureDetector:
    """
    Clase para detectar y describir características en imágenes.
    Soporta múltiples tipos de detectores.
    """
    
    def __init__(self, detector_type: DetectorType = DetectorType.SIFT, **kwargs):
        """
        Inicializa el detector de características.
        
        Args:
            detector_type: Tipo de detector a utilizar
            **kwargs: Parámetros específicos del detector
        """
        self.detector_type = detector_type
        self.detector = self._create_detector(**kwargs)
    
    def _create_detector(self, **kwargs):
        """
        Crea el detector según el tipo especificado.
        
        Args:
            **kwargs: Parámetros específicos del detector
            
        Returns:
            Objeto detector de OpenCV
        """
        if self.detector_type == DetectorType.SIFT:
            # SIFT: Scale-Invariant Feature Transform
            # Parámetros por defecto optimizados
            nfeatures = kwargs.get('nfeatures', 0)  # 0 = sin límite
            nOctaveLayers = kwargs.get('nOctaveLayers', 3)
            contrastThreshold = kwargs.get('contrastThreshold', 0.04)
            edgeThreshold = kwargs.get('edgeThreshold', 10)
            sigma = kwargs.get('sigma', 1.6)
            
            return cv2.SIFT_create(
                nfeatures=nfeatures,
                nOctaveLayers=nOctaveLayers,
                contrastThreshold=contrastThreshold,
                edgeThreshold=edgeThreshold,
                sigma=sigma
            )
        
        elif self.detector_type == DetectorType.ORB:
            # ORB: Oriented FAST and Rotated BRIEF
            # Parámetros por defecto optimizados
            nfeatures = kwargs.get('nfeatures', 2000)
            scaleFactor = kwargs.get('scaleFactor', 1.2)
            nlevels = kwargs.get('nlevels', 8)
            edgeThreshold = kwargs.get('edgeThreshold', 31)
            firstLevel = kwargs.get('firstLevel', 0)
            WTA_K = kwargs.get('WTA_K', 2)
            scoreType = kwargs.get('scoreType', cv2.ORB_HARRIS_SCORE)
            patchSize = kwargs.get('patchSize', 31)
            fastThreshold = kwargs.get('fastThreshold', 20)
            
            return cv2.ORB_create(
                nfeatures=nfeatures,
                scaleFactor=scaleFactor,
                nlevels=nlevels,
                edgeThreshold=edgeThreshold,
                firstLevel=firstLevel,
                WTA_K=WTA_K,
                scoreType=scoreType,
                patchSize=patchSize,
                fastThreshold=fastThreshold
            )
        
        elif self.detector_type == DetectorType.AKAZE:
            # AKAZE: Accelerated-KAZE
            # Parámetros por defecto optimizados
            descriptor_type = kwargs.get('descriptor_type', cv2.AKAZE_DESCRIPTOR_MLDB)
            descriptor_size = kwargs.get('descriptor_size', 0)
            descriptor_channels = kwargs.get('descriptor_channels', 3)
            threshold = kwargs.get('threshold', 0.001)
            nOctaves = kwargs.get('nOctaves', 4)
            nOctaveLayers = kwargs.get('nOctaveLayers', 4)
            diffusivity = kwargs.get('diffusivity', cv2.KAZE_DIFF_PM_G2)
            
            return cv2.AKAZE_create(
                descriptor_type=descriptor_type,
                descriptor_size=descriptor_size,
                descriptor_channels=descriptor_channels,
                threshold=threshold,
                nOctaves=nOctaves,
                nOctaveLayers=nOctaveLayers,
                diffusivity=diffusivity
            )
        
        else:
            raise ValueError(f"Detector tipo {self.detector_type} no soportado")
    
    def detect_and_compute(self, image: np.ndarray, 
                          mask: Optional[np.ndarray] = None) -> Tuple[List, np.ndarray]:
        """
        Detecta keypoints y calcula descriptores en una imagen.
        
        Args:
            image: Imagen de entrada (puede ser color o escala de grises)
            mask: Máscara opcional para limitar la detección
            
        Returns:
            Tupla (keypoints, descriptors)
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detectar y computar
        keypoints, descriptors = self.detector.detectAndCompute(gray, mask)
        
        return keypoints, descriptors
    
    def detect(self, image: np.ndarray, 
               mask: Optional[np.ndarray] = None) -> List:
        """
        Solo detecta keypoints sin calcular descriptores.
        
        Args:
            image: Imagen de entrada
            mask: Máscara opcional
            
        Returns:
            Lista de keypoints
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        keypoints = self.detector.detect(gray, mask)
        
        return keypoints
    
    def compute(self, image: np.ndarray, keypoints: List) -> Tuple[List, np.ndarray]:
        """
        Calcula descriptores para keypoints dados.
        
        Args:
            image: Imagen de entrada
            keypoints: Lista de keypoints
            
        Returns:
            Tupla (keypoints, descriptors)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        keypoints, descriptors = self.detector.compute(gray, keypoints)
        
        return keypoints, descriptors
    
    def get_keypoint_info(self, keypoints: List) -> Dict:
        """
        Obtiene información estadística sobre los keypoints detectados.
        
        Args:
            keypoints: Lista de keypoints
            
        Returns:
            Diccionario con información estadística
        """
        if not keypoints:
            return {
                'count': 0,
                'mean_response': 0,
                'mean_size': 0,
                'mean_angle': 0
            }
        
        responses = [kp.response for kp in keypoints]
        sizes = [kp.size for kp in keypoints]
        angles = [kp.angle for kp in keypoints]
        
        return {
            'count': len(keypoints),
            'mean_response': np.mean(responses),
            'std_response': np.std(responses),
            'mean_size': np.mean(sizes),
            'std_size': np.std(sizes),
            'mean_angle': np.mean(angles),
            'std_angle': np.std(angles)
        }


def compare_detectors(image: np.ndarray, 
                     detector_types: List[DetectorType] = None) -> Dict:
    """
    Compara diferentes detectores en la misma imagen.
    
    Args:
        image: Imagen para analizar
        detector_types: Lista de tipos de detectores a comparar
        
    Returns:
        Diccionario con resultados de cada detector
    """
    if detector_types is None:
        detector_types = [DetectorType.SIFT, DetectorType.ORB, DetectorType.AKAZE]
    
    results = {}
    
    for detector_type in detector_types:
        detector = FeatureDetector(detector_type)
        keypoints, descriptors = detector.detect_and_compute(image)
        info = detector.get_keypoint_info(keypoints)
        
        results[detector_type.value] = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'info': info
        }
    
    return results


def filter_keypoints_by_response(keypoints: List, descriptors: np.ndarray,
                                 threshold: float = 0.0,
                                 top_n: Optional[int] = None) -> Tuple[List, np.ndarray]:
    """
    Filtra keypoints basándose en su respuesta (strength).
    
    Args:
        keypoints: Lista de keypoints
        descriptors: Descriptores correspondientes
        threshold: Umbral mínimo de respuesta
        top_n: Si se especifica, retorna solo los top N keypoints
        
    Returns:
        Tupla (keypoints filtrados, descriptors filtrados)
    """
    if not keypoints or descriptors is None:
        return keypoints, descriptors
    
    # Filtrar por umbral
    filtered_pairs = [(kp, desc) for kp, desc in zip(keypoints, descriptors) 
                     if kp.response >= threshold]
    
    if not filtered_pairs:
        return [], np.array([])
    
    # Ordenar por respuesta descendente
    filtered_pairs.sort(key=lambda x: x[0].response, reverse=True)
    
    # Tomar top N si se especifica
    if top_n is not None:
        filtered_pairs = filtered_pairs[:top_n]
    
    # Separar keypoints y descriptors
    filtered_kps = [pair[0] for pair in filtered_pairs]
    filtered_descs = np.array([pair[1] for pair in filtered_pairs])
    
    return filtered_kps, filtered_descs


def create_keypoint_grid_mask(image_shape: Tuple[int, int], 
                              grid_size: int = 50,
                              keypoints: List = None) -> np.ndarray:
    """
    Crea una máscara para distribuir keypoints uniformemente en una grilla.
    Útil para evitar concentración de keypoints en ciertas áreas.
    
    Args:
        image_shape: Forma de la imagen (height, width)
        grid_size: Tamaño de cada celda de la grilla
        keypoints: Si se proporcionan, distribuye basándose en keypoints existentes
        
    Returns:
        Máscara binaria
    """
    height, width = image_shape[:2]
    mask = np.ones((height, width), dtype=np.uint8) * 255
    
    if keypoints is not None:
        # Crear grilla
        grid_h = height // grid_size + 1
        grid_w = width // grid_size + 1
        grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
        
        # Asignar keypoints a celdas de la grilla
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            grid_x = min(x // grid_size, grid_w - 1)
            grid_y = min(y // grid_size, grid_h - 1)
            grid[grid_y][grid_x].append(kp)
        
        # Marcar áreas con muchos keypoints
        max_kps_per_cell = max(len(cell) for row in grid for cell in row)
        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if len(cell) > max_kps_per_cell * 0.5:  # Si tiene más del 50% del máximo
                    y1 = i * grid_size
                    y2 = min((i + 1) * grid_size, height)
                    x1 = j * grid_size
                    x2 = min((j + 1) * grid_size, width)
                    mask[y1:y2, x1:x2] = 127  # Reducir prioridad
    
    return mask


def visualize_keypoint_distribution(image: np.ndarray, keypoints: List,
                                    grid_size: int = 50) -> np.ndarray:
    """
    Visualiza la distribución de keypoints en una grilla.
    
    Args:
        image: Imagen base
        keypoints: Lista de keypoints
        grid_size: Tamaño de la grilla
        
    Returns:
        Imagen con la visualización de distribución
    """
    img_viz = image.copy()
    if len(img_viz.shape) == 2:
        img_viz = cv2.cvtColor(img_viz, cv2.COLOR_GRAY2RGB)
    
    height, width = img_viz.shape[:2]
    
    # Dibujar grilla
    for y in range(0, height, grid_size):
        cv2.line(img_viz, (0, y), (width, y), (200, 200, 200), 1)
    for x in range(0, width, grid_size):
        cv2.line(img_viz, (x, 0), (x, height), (200, 200, 200), 1)
    
    # Contar keypoints por celda
    grid_h = height // grid_size + 1
    grid_w = width // grid_size + 1
    grid_counts = np.zeros((grid_h, grid_w), dtype=int)
    
    for kp in keypoints:
        x, y = int(kp.pt[0]), int(kp.pt[1])
        grid_x = min(x // grid_size, grid_w - 1)
        grid_y = min(y // grid_size, grid_h - 1)
        grid_counts[grid_y, grid_x] += 1
    
    # Colorear celdas según densidad
    max_count = grid_counts.max()
    if max_count > 0:
        for i in range(grid_h):
            for j in range(grid_w):
                count = grid_counts[i, j]
                if count > 0:
                    intensity = int(255 * (count / max_count))
                    y1 = i * grid_size
                    y2 = min((i + 1) * grid_size, height)
                    x1 = j * grid_size
                    x2 = min((j + 1) * grid_size, width)
                    
                    # Overlay semitransparente
                    overlay = img_viz[y1:y2, x1:x2].copy()
                    cv2.rectangle(overlay, (0, 0), (x2-x1, y2-y1), 
                                (intensity, 0, 255-intensity), -1)
                    img_viz[y1:y2, x1:x2] = cv2.addWeighted(
                        img_viz[y1:y2, x1:x2], 0.7, overlay, 0.3, 0)
                    
                    # Mostrar conteo
                    text_pos = (x1 + 5, y1 + 20)
                    cv2.putText(img_viz, str(count), text_pos,
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return img_viz


