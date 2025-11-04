"""
Módulo de emparejamiento de características para registro de imágenes.
Implementa estrategias robustas de matching.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from enum import Enum


class MatcherType(Enum):
    """Tipos de matchers disponibles."""
    BF = "BruteForce"  # Brute Force Matcher
    FLANN = "FLANN"    # Fast Library for Approximate Nearest Neighbors


class FeatureMatcher:
    """
    Clase para emparejar características entre dos imágenes.
    """
    
    def __init__(self, matcher_type: MatcherType = MatcherType.BF,
                 descriptor_type: str = 'SIFT',
                 cross_check: bool = False):
        """
        Inicializa el matcher.
        
        Args:
            matcher_type: Tipo de matcher (BF o FLANN)
            descriptor_type: Tipo de descriptor ('SIFT', 'ORB', 'AKAZE')
            cross_check: Si True, solo retorna matches consistentes en ambas direcciones
        """
        self.matcher_type = matcher_type
        self.descriptor_type = descriptor_type
        self.cross_check = cross_check
        self.matcher = self._create_matcher()
    
    def _create_matcher(self):
        """
        Crea el matcher según el tipo especificado.
        
        Returns:
            Objeto matcher de OpenCV
        """
        if self.matcher_type == MatcherType.BF:
            # Brute Force Matcher
            if self.descriptor_type in ['SIFT', 'AKAZE']:
                # Para descriptores basados en flotantes (SIFT, AKAZE con MLDB)
                norm_type = cv2.NORM_L2
            else:  # ORB
                # Para descriptores binarios (ORB, AKAZE con BRIEF)
                norm_type = cv2.NORM_HAMMING
            
            return cv2.BFMatcher(norm_type, crossCheck=self.cross_check)
        
        elif self.matcher_type == MatcherType.FLANN:
            # FLANN Matcher
            if self.descriptor_type in ['SIFT', 'AKAZE']:
                # Para descriptores basados en flotantes
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            else:  # ORB
                # Para descriptores binarios
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=6,
                    key_size=12,
                    multi_probe_level=1
                )
            
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        
        else:
            raise ValueError(f"Matcher tipo {self.matcher_type} no soportado")
    
    def match(self, descriptors1: np.ndarray, 
              descriptors2: np.ndarray,
              k: int = 2) -> List:
        """
        Empareja descriptores entre dos conjuntos.
        
        Args:
            descriptors1: Descriptores de la primera imagen
            descriptors2: Descriptores de la segunda imagen
            k: Número de mejores matches a retornar por descriptor
            
        Returns:
            Lista de matches (o lista de listas si k > 1)
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        
        # Para ORB, asegurar que los descriptors son del tipo correcto
        if self.descriptor_type == 'ORB':
            descriptors1 = descriptors1.astype(np.uint8)
            descriptors2 = descriptors2.astype(np.uint8)
        
        if self.cross_check or k == 1:
            # Match simple
            matches = self.matcher.match(descriptors1, descriptors2)
            # Ordenar por distancia
            matches = sorted(matches, key=lambda x: x.distance)
            return matches
        else:
            # KNN match
            matches = self.matcher.knnMatch(descriptors1, descriptors2, k=k)
            return matches
    
    def match_with_ratio_test(self, descriptors1: np.ndarray,
                               descriptors2: np.ndarray,
                               ratio: float = 0.75) -> List:
        """
        Empareja descriptores aplicando el ratio test de Lowe.
        
        El ratio test filtra matches ambiguos comparando la distancia del
        mejor match con el segundo mejor match.
        
        Args:
            descriptors1: Descriptores de la primera imagen
            descriptors2: Descriptores de la segunda imagen
            ratio: Umbral del ratio test (típicamente 0.7-0.8)
            
        Returns:
            Lista de buenos matches (DMatch)
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        
        # KNN match con k=2
        matches = self.match(descriptors1, descriptors2, k=2)
        
        # Aplicar ratio test
        good_matches = []
        for match_pair in matches:
            # Verificar que tenemos al menos 2 matches
            if len(match_pair) >= 2:
                m, n = match_pair[0], match_pair[1]
                # Si el mejor match es significativamente mejor que el segundo
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                # Si solo hay un match, lo aceptamos
                good_matches.append(match_pair[0])
        
        # Ordenar por distancia
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        
        return good_matches
    
    def match_with_symmetry_test(self, descriptors1: np.ndarray,
                                  descriptors2: np.ndarray) -> List:
        """
        Empareja descriptores aplicando symmetry test (cross-check).
        
        Solo retorna matches que son mutuamente los mejores matches.
        
        Args:
            descriptors1: Descriptores de la primera imagen
            descriptors2: Descriptores de la segunda imagen
            
        Returns:
            Lista de matches simétricos
        """
        if descriptors1 is None or descriptors2 is None:
            return []
        
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
        
        # Match en ambas direcciones
        matches_1to2 = self.match(descriptors1, descriptors2, k=1)
        matches_2to1 = self.match(descriptors2, descriptors1, k=1)
        
        # Encontrar matches simétricos
        symmetric_matches = []
        for m1 in matches_1to2:
            for m2 in matches_2to1:
                if m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx:
                    symmetric_matches.append(m1)
                    break
        
        return symmetric_matches
    
    def match_robust(self, descriptors1: np.ndarray,
                     descriptors2: np.ndarray,
                     ratio: float = 0.75,
                     use_symmetry: bool = False) -> List:
        """
        Emparejamiento robusto combinando múltiples estrategias.
        
        Args:
            descriptors1: Descriptores de la primera imagen
            descriptors2: Descriptores de la segunda imagen
            ratio: Umbral del ratio test
            use_symmetry: Si True, aplica también el symmetry test
            
        Returns:
            Lista de matches robustos
        """
        # Aplicar ratio test
        matches = self.match_with_ratio_test(descriptors1, descriptors2, ratio)
        
        # Aplicar symmetry test si se solicita
        if use_symmetry:
            matches_symmetric = self.match_with_symmetry_test(descriptors1, descriptors2)
            
            # Intersección de ambos conjuntos
            matches_set = set((m.queryIdx, m.trainIdx) for m in matches)
            symmetric_set = set((m.queryIdx, m.trainIdx) for m in matches_symmetric)
            
            common_set = matches_set.intersection(symmetric_set)
            
            # Filtrar matches
            matches = [m for m in matches if (m.queryIdx, m.trainIdx) in common_set]
        
        return matches


def get_matched_points(keypoints1: List, keypoints2: List,
                       matches: List) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extrae las coordenadas de los puntos emparejados.
    
    Args:
        keypoints1: Keypoints de la primera imagen
        keypoints2: Keypoints de la segunda imagen
        matches: Lista de matches
        
    Returns:
        Tupla (puntos1, puntos2) como arrays Nx2
    """
    if not matches:
        return np.array([]), np.array([])
    
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    
    return points1, points2


def filter_matches_by_distance(matches: List, 
                               max_distance: float = None,
                               top_percent: float = None) -> List:
    """
    Filtra matches basándose en la distancia.
    
    Args:
        matches: Lista de matches
        max_distance: Distancia máxima permitida
        top_percent: Si se especifica, retorna el top % de matches (0-1)
        
    Returns:
        Lista de matches filtrados
    """
    if not matches:
        return []
    
    # Ordenar por distancia
    sorted_matches = sorted(matches, key=lambda x: x.distance)
    
    # Filtrar por distancia máxima
    if max_distance is not None:
        sorted_matches = [m for m in sorted_matches if m.distance <= max_distance]
    
    # Filtrar por porcentaje
    if top_percent is not None:
        n = int(len(sorted_matches) * top_percent)
        sorted_matches = sorted_matches[:max(1, n)]
    
    return sorted_matches


def compute_match_statistics(matches: List, 
                            keypoints1: List = None,
                            keypoints2: List = None) -> Dict:
    """
    Calcula estadísticas sobre los matches.
    
    Args:
        matches: Lista de matches
        keypoints1: Keypoints de la primera imagen (opcional)
        keypoints2: Keypoints de la segunda imagen (opcional)
        
    Returns:
        Diccionario con estadísticas
    """
    if not matches:
        return {
            'count': 0,
            'mean_distance': 0,
            'std_distance': 0,
            'min_distance': 0,
            'max_distance': 0
        }
    
    distances = [m.distance for m in matches]
    
    stats = {
        'count': len(matches),
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances)
    }
    
    # Si tenemos los keypoints, calcular estadísticas geométricas
    if keypoints1 is not None and keypoints2 is not None:
        points1, points2 = get_matched_points(keypoints1, keypoints2, matches)
        
        if len(points1) > 0:
            # Distancias euclidianas entre puntos
            euclidean_dists = np.linalg.norm(points1 - points2, axis=1)
            
            stats['mean_euclidean_distance'] = np.mean(euclidean_dists)
            stats['std_euclidean_distance'] = np.std(euclidean_dists)
    
    return stats


def visualize_matches_distribution(matches: List, bins: int = 50) -> None:
    """
    Visualiza la distribución de distancias de los matches.
    
    Args:
        matches: Lista de matches
        bins: Número de bins para el histograma
    """
    import matplotlib.pyplot as plt
    
    if not matches:
        print("No hay matches para visualizar")
        return
    
    distances = [m.distance for m in matches]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histograma
    axes[0].hist(distances, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(np.mean(distances), color='red', linestyle='--', 
                    label=f'Media: {np.mean(distances):.2f}')
    axes[0].axvline(np.median(distances), color='green', linestyle='--',
                    label=f'Mediana: {np.median(distances):.2f}')
    axes[0].set_xlabel('Distancia')
    axes[0].set_ylabel('Frecuencia')
    axes[0].set_title('Distribución de Distancias de Matches')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Gráfico acumulativo
    sorted_distances = sorted(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances)
    axes[1].plot(sorted_distances, cumulative, color='steelblue', linewidth=2)
    axes[1].set_xlabel('Distancia')
    axes[1].set_ylabel('Proporción Acumulativa')
    axes[1].set_title('Distribución Acumulativa')
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def filter_matches_spatially(keypoints1: List, keypoints2: List,
                            matches: List,
                            max_spatial_distance: float = None,
                            angle_threshold: float = None) -> List:
    """
    Filtra matches basándose en criterios espaciales.
    
    Args:
        keypoints1: Keypoints de la primera imagen
        keypoints2: Keypoints de la segunda imagen
        matches: Lista de matches
        max_spatial_distance: Distancia espacial máxima permitida entre puntos
        angle_threshold: Umbral de diferencia angular (en grados)
        
    Returns:
        Lista de matches filtrados
    """
    if not matches:
        return []
    
    filtered_matches = []
    
    for m in matches:
        kp1 = keypoints1[m.queryIdx]
        kp2 = keypoints2[m.trainIdx]
        
        # Filtrar por distancia espacial
        if max_spatial_distance is not None:
            spatial_dist = np.linalg.norm(
                np.array(kp1.pt) - np.array(kp2.pt)
            )
            if spatial_dist > max_spatial_distance:
                continue
        
        # Filtrar por diferencia angular
        if angle_threshold is not None:
            angle_diff = abs(kp1.angle - kp2.angle)
            # Normalizar a [-180, 180]
            angle_diff = (angle_diff + 180) % 360 - 180
            if abs(angle_diff) > angle_threshold:
                continue
        
        filtered_matches.append(m)
    
    return filtered_matches


def match_image_pair(img1: np.ndarray, img2: np.ndarray,
                    detector_type: str = 'SIFT',
                    matcher_type: MatcherType = MatcherType.BF,
                    ratio: float = 0.75,
                    min_matches: int = 10) -> Dict:
    """
    Pipeline completo para emparejar un par de imágenes.
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        detector_type: Tipo de detector ('SIFT', 'ORB', 'AKAZE')
        matcher_type: Tipo de matcher
        ratio: Umbral del ratio test
        min_matches: Número mínimo de matches requeridos
        
    Returns:
        Diccionario con keypoints, descriptors, matches y estadísticas
    """
    from src.feature_detection import FeatureDetector, DetectorType
    
    # Detectar características
    if detector_type == 'SIFT':
        det_type = DetectorType.SIFT
    elif detector_type == 'ORB':
        det_type = DetectorType.ORB
    else:
        det_type = DetectorType.AKAZE
    
    detector = FeatureDetector(det_type)
    kp1, desc1 = detector.detect_and_compute(img1)
    kp2, desc2 = detector.detect_and_compute(img2)
    
    # Emparejar características
    matcher = FeatureMatcher(matcher_type, detector_type)
    matches = matcher.match_robust(desc1, desc2, ratio=ratio)
    
    # Verificar número mínimo de matches
    success = len(matches) >= min_matches
    
    # Estadísticas
    stats = compute_match_statistics(matches, kp1, kp2)
    
    return {
        'keypoints1': kp1,
        'keypoints2': kp2,
        'descriptors1': desc1,
        'descriptors2': desc2,
        'matches': matches,
        'statistics': stats,
        'success': success
    }


