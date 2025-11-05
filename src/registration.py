"""
Módulo de registro de imágenes.
Implementa estimación de homografías, RANSAC y técnicas de blending.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
import warnings


def estimate_homography(points1: np.ndarray, points2: np.ndarray,
                       method: int = cv2.RANSAC,
                       ransac_threshold: float = 5.0,
                       confidence: float = 0.995,
                       max_iters: int = 2000) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estima la homografía entre dos conjuntos de puntos.
    
    Args:
        points1: Puntos de la primera imagen (Nx2)
        points2: Puntos de la segunda imagen (Nx2)
        method: Método de estimación (cv2.RANSAC, cv2.LMEDS, etc.)
        ransac_threshold: Umbral para RANSAC (en píxeles)
        confidence: Nivel de confianza para RANSAC
        max_iters: Número máximo de iteraciones
        
    Returns:
        Tupla (homografía, máscara de inliers)
    """
    if len(points1) < 4 or len(points2) < 4:
        warnings.warn("Se necesitan al menos 4 puntos para estimar una homografía")
        return None, None
    
    H, mask = cv2.findHomography(
        points1, points2,
        method=method,
        ransacReprojThreshold=ransac_threshold,
        confidence=confidence,
        maxIters=max_iters
    )
    
    return H, mask


def validate_homography(H: np.ndarray, 
                       image_shape: Tuple[int, int],
                       max_skew: float = 0.5,
                       max_scale_change: float = 5.0) -> bool:
    """
    Valida si una homografía es razonable.
    
    Args:
        H: Matriz de homografía
        image_shape: Forma de la imagen (height, width)
        max_skew: Máximo skew permitido
        max_scale_change: Cambio máximo de escala permitido
        
    Returns:
        True si la homografía es válida
    """
    if H is None:
        return False
    
    # Verificar si la matriz es singular
    if np.linalg.det(H) < 1e-6:
        return False
    
    # Normalizar
    H_norm = H / H[2, 2]
    
    # Verificar cambios de escala extremos
    h, w = image_shape[:2]
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    
    try:
        transformed_corners = cv2.perspectiveTransform(corners, H_norm)
    except:
        return False
    
    # Calcular área transformada
    original_area = w * h
    transformed_area = cv2.contourArea(transformed_corners)
    
    if transformed_area <= 0:
        return False
    
    scale_change = np.sqrt(transformed_area / original_area)
    
    if scale_change > max_scale_change or scale_change < 1.0 / max_scale_change:
        return False
    
    # Verificar skew (no implementado en detalle, pero podría agregarse)
    # Por ahora solo verificamos que la transformación no sea degenerada
    
    return True


def refine_homography(H: np.ndarray, points1: np.ndarray, points2: np.ndarray,
                     mask: np.ndarray = None) -> np.ndarray:
    """
    Refina una homografía usando solo inliers.
    
    Args:
        H: Homografía inicial
        points1: Puntos de la primera imagen
        points2: Puntos de la segunda imagen
        mask: Máscara de inliers
        
    Returns:
        Homografía refinada
    """
    if mask is not None:
        # Filtrar solo inliers
        inliers1 = points1[mask.ravel() == 1]
        inliers2 = points2[mask.ravel() == 1]
    else:
        inliers1 = points1
        inliers2 = points2
    
    if len(inliers1) < 4:
        return H
    
    # Re-estimar con todos los inliers (sin RANSAC)
    H_refined, _ = cv2.findHomography(inliers1, inliers2, method=0)
    
    return H_refined if H_refined is not None else H


def warp_image(image: np.ndarray, H: np.ndarray,
               output_shape: Tuple[int, int] = None,
               offset: Tuple[int, int] = (0, 0)) -> np.ndarray:
    """
    Aplica una transformación homográfica a una imagen.
    
    Args:
        image: Imagen a transformar
        H: Matriz de homografía
        output_shape: Forma de salida (height, width). Si None, usa forma de entrada
        offset: Offset para centrar la imagen (x, y)
        
    Returns:
        Imagen transformada
    """
    if output_shape is None:
        output_shape = (image.shape[0], image.shape[1])
    
    # Crear matriz de traslación para el offset
    T = np.array([
        [1, 0, offset[0]],
        [0, 1, offset[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Combinar homografía con traslación
    H_with_offset = T @ H
    
    # Aplicar transformación
    warped = cv2.warpPerspective(
        image, H_with_offset,
        (output_shape[1], output_shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return warped


def create_mask(image_shape: Tuple[int, int], H: np.ndarray = None,
                offset: Tuple[int, int] = (0, 0),
                output_shape: Tuple[int, int] = None) -> np.ndarray:
    """
    Crea una máscara para una imagen transformada.
    
    Args:
        image_shape: Forma de la imagen original
        H: Homografía (si None, usa identidad)
        offset: Offset de la imagen
        output_shape: Forma de salida
        
    Returns:
        Máscara binaria
    """
    mask = np.ones((image_shape[0], image_shape[1]), dtype=np.uint8) * 255
    
    if H is None:
        H = np.eye(3)
    
    if output_shape is None:
        output_shape = image_shape
    
    warped_mask = warp_image(mask, H, output_shape, offset)
    
    return warped_mask


def simple_blend(img1: np.ndarray, img2: np.ndarray,
                mask1: np.ndarray = None, mask2: np.ndarray = None,
                weight1: float = 1.0, weight2: float = 1.0) -> np.ndarray:
    """
    Fusión simple de dos imágenes usando promedio ponderado.
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        mask1: Máscara de la primera imagen
        mask2: Máscara de la segunda imagen
        weight1: Peso relativo de la primera imagen (default: 1.0)
        weight2: Peso relativo de la segunda imagen (default: 1.0)
        
    Returns:
        Imagen fusionada
    """
    if mask1 is None:
        mask1 = (img1 > 0).any(axis=2 if len(img1.shape) == 3 else None).astype(np.uint8) * 255
    if mask2 is None:
        mask2 = (img2 > 0).any(axis=2 if len(img2.shape) == 3 else None).astype(np.uint8) * 255
    
    # Normalizar pesos
    total_weight = weight1 + weight2
    if total_weight == 0:
        weight1, weight2 = 1.0, 1.0
        total_weight = 2.0
    w1 = weight1 / total_weight
    w2 = weight2 / total_weight
    
    # Crear resultado
    result = np.zeros_like(img1)
    
    # Áreas con solo img1
    only_img1 = (mask1 > 0) & (mask2 == 0)
    result[only_img1] = img1[only_img1]
    
    # Áreas con solo img2
    only_img2 = (mask1 == 0) & (mask2 > 0)
    result[only_img2] = img2[only_img2]
    
    # Áreas de solapamiento - promedio ponderado
    overlap = (mask1 > 0) & (mask2 > 0)
    if len(img1.shape) == 3:
        for c in range(img1.shape[2]):
            result[overlap, c] = (img1[overlap, c].astype(float) * w1 + 
                                 img2[overlap, c].astype(float) * w2)
    else:
        result[overlap] = (img1[overlap].astype(float) * w1 + 
                          img2[overlap].astype(float) * w2)
    
    return result.astype(img1.dtype)


def feather_blend(img1: np.ndarray, img2: np.ndarray,
                 mask1: np.ndarray = None, mask2: np.ndarray = None,
                 feather_amount: int = 50,
                 weight1: float = 1.0, weight2: float = 1.0) -> np.ndarray:
    """
    Fusión con feathering (transición suave en bordes).
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        mask1: Máscara de la primera imagen
        mask2: Máscara de la segunda imagen
        feather_amount: Cantidad de píxeles para el feathering
        weight1: Peso relativo de la primera imagen (default: 1.0)
        weight2: Peso relativo de la segunda imagen (default: 1.0)
        
    Returns:
        Imagen fusionada
    """
    if mask1 is None:
        mask1 = (img1 > 0).any(axis=2 if len(img1.shape) == 3 else None).astype(np.uint8) * 255
    if mask2 is None:
        mask2 = (img2 > 0).any(axis=2 if len(img2.shape) == 3 else None).astype(np.uint8) * 255
    
    # Aplicar distance transform para crear gradientes suaves
    dist1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5)
    dist2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5)
    
    # Normalizar y aplicar feathering
    dist1 = np.clip(dist1 / feather_amount, 0, 1)
    dist2 = np.clip(dist2 / feather_amount, 0, 1)
    
    # Aplicar pesos relativos a las distancias
    dist1_weighted = dist1 * weight1
    dist2_weighted = dist2 * weight2
    
    # Calcular pesos normalizados
    total_dist = dist1_weighted + dist2_weighted
    # Evitar división por cero
    total_dist = np.where(total_dist == 0, 1, total_dist)
    
    w1 = dist1_weighted / total_dist
    w2 = dist2_weighted / total_dist
    
    # Expandir pesos a las dimensiones de la imagen
    if len(img1.shape) == 3:
        w1 = np.expand_dims(w1, axis=2)
        w2 = np.expand_dims(w2, axis=2)
    
    # Fusionar con pesos
    result = (img1 * w1 + img2 * w2).astype(img1.dtype)
    
    return result


def multiband_blend(img1: np.ndarray, img2: np.ndarray,
                   mask1: np.ndarray = None, mask2: np.ndarray = None,
                   levels: int = 4) -> np.ndarray:
    """
    Fusión multiband (pirámide de Laplace).
    Ofrece mejor calidad para panoramas.
    
    Args:
        img1: Primera imagen
        img2: Segunda imagen
        mask1: Máscara de la primera imagen
        mask2: Máscara de la segunda imagen
        levels: Número de niveles de la pirámide
        
    Returns:
        Imagen fusionada
    """
    if mask1 is None:
        mask1 = (img1 > 0).any(axis=2 if len(img1.shape) == 3 else None).astype(np.uint8) * 255
    if mask2 is None:
        mask2 = (img2 > 0).any(axis=2 if len(img2.shape) == 3 else None).astype(np.uint8) * 255
    
    # Convertir máscaras a float
    mask1_float = mask1.astype(float) / 255.0
    mask2_float = mask2.astype(float) / 255.0
    
    # Crear máscaras de mezcla
    blend_mask = mask1_float / (mask1_float + mask2_float + 1e-10)
    
    if len(img1.shape) == 3:
        blend_mask = np.expand_dims(blend_mask, axis=2)
    
    # Construir pirámides Gaussianas
    def build_gaussian_pyramid(img, levels):
        pyramid = [img.astype(float)]
        for i in range(levels - 1):
            img = cv2.pyrDown(img)
            pyramid.append(img)
        return pyramid
    
    # Construir pirámides Laplacianas
    def build_laplacian_pyramid(gauss_pyramid):
        laplacian_pyramid = []
        for i in range(len(gauss_pyramid) - 1):
            upsampled = cv2.pyrUp(gauss_pyramid[i + 1])
            # Asegurar que tengan el mismo tamaño
            upsampled = cv2.resize(upsampled, (gauss_pyramid[i].shape[1], gauss_pyramid[i].shape[0]))
            laplacian = gauss_pyramid[i] - upsampled
            laplacian_pyramid.append(laplacian)
        laplacian_pyramid.append(gauss_pyramid[-1])
        return laplacian_pyramid
    
    # Construir pirámides
    gauss1 = build_gaussian_pyramid(img1, levels)
    gauss2 = build_gaussian_pyramid(img2, levels)
    gauss_mask = build_gaussian_pyramid(blend_mask, levels)
    
    laplacian1 = build_laplacian_pyramid(gauss1)
    laplacian2 = build_laplacian_pyramid(gauss2)
    
    # Fusionar pirámides
    blended_pyramid = []
    for l1, l2, mask in zip(laplacian1, laplacian2, gauss_mask):
        # Asegurar que la máscara tenga las dimensiones correctas
        mask = cv2.resize(mask, (l1.shape[1], l1.shape[0]))
        if len(l1.shape) == 3 and len(mask.shape) == 2:
            # Expandir máscara para imágenes RGB
            mask = np.expand_dims(mask, axis=2)
        # Asegurar que mask tenga el mismo número de canales que las imágenes
        if len(l1.shape) == 3:
            # Si las imágenes son RGB, asegurar que mask tenga 3 canales
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)
        blended = l1 * mask + l2 * (1 - mask)
        blended_pyramid.append(blended)
    
    # Reconstruir imagen desde la pirámide
    result = blended_pyramid[-1]
    for i in range(len(blended_pyramid) - 2, -1, -1):
        result = cv2.pyrUp(result)
        result = cv2.resize(result, (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0]))
        result = result + blended_pyramid[i]
    
    # Asegurar que los valores estén en el rango correcto
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result


class ImageStitcher:
    """
    Clase para fusionar múltiples imágenes en un panorama.
    """
    
    def __init__(self, blend_method: str = 'feather', image_weights: List[float] = None):
        """
        Inicializa el stitcher.
        
        Args:
            blend_method: Método de blending ('simple', 'feather', 'multiband')
            image_weights: Lista de pesos para cada imagen (None = pesos iguales)
        """
        self.blend_method = blend_method
        self.image_weights = image_weights
        self.homographies = []
        self.reference_idx = None
    
    def estimate_transforms(self, images: List[np.ndarray],
                           matches_list: List[Dict],
                           reference_idx: int = 0) -> bool:
        """
        Estima las transformaciones entre imágenes.
        
        Args:
            images: Lista de imágenes
            matches_list: Lista de diccionarios con matches entre pares de imágenes
            reference_idx: Índice de la imagen de referencia
            
        Returns:
            True si todas las transformaciones se estimaron correctamente
        """
        self.reference_idx = reference_idx
        n_images = len(images)
        
        # Inicializar matriz de homografías
        # Cada H[i] transforma la imagen i a la referencia
        self.homographies = [None] * n_images
        self.homographies[reference_idx] = np.eye(3)
        
        # Para simplificar, asumimos imágenes secuenciales
        # matches_list[i] contiene matches entre imagen i e i+1
        
        # Estimar homografías de izquierda a derecha desde referencia
        for i in range(reference_idx, n_images - 1):
            match_data = matches_list[i]
            points1, points2 = match_data['points1'], match_data['points2']
            
            # Estimar H que lleva img[i+1] a img[i]
            H, mask = estimate_homography(points2, points1)
            
            if H is None or not validate_homography(H, images[i + 1].shape):
                print(f"Advertencia: No se pudo estimar homografía entre imágenes {i} y {i+1}")
                return False
            
            # Refinar
            H = refine_homography(H, points2, points1, mask)
            
            # Acumular transformación respecto a referencia
            if i == reference_idx:
                self.homographies[i + 1] = H
            else:
                self.homographies[i + 1] = self.homographies[i] @ H
        
        # Estimar homografías de derecha a izquierda desde referencia
        for i in range(reference_idx, 0, -1):
            match_data = matches_list[i - 1]
            points1, points2 = match_data['points1'], match_data['points2']
            
            # Estimar H que lleva img[i-1] a img[i]
            H, mask = estimate_homography(points1, points2)
            
            if H is None or not validate_homography(H, images[i - 1].shape):
                print(f"Advertencia: No se pudo estimar homografía entre imágenes {i-1} y {i}")
                return False
            
            # Refinar
            H = refine_homography(H, points1, points2, mask)
            
            # Acumular transformación respecto a referencia
            if i == reference_idx:
                self.homographies[i - 1] = H
            else:
                self.homographies[i - 1] = self.homographies[i] @ H
        
        return True
    
    def stitch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Fusiona las imágenes en un panorama.
        
        Args:
            images: Lista de imágenes a fusionar
            
        Returns:
            Imagen panorámica fusionada
        """
        if not self.homographies or self.reference_idx is None:
            raise ValueError("Primero debe estimar las transformaciones")
        
        # Calcular tamaño del canvas
        h_ref, w_ref = images[self.reference_idx].shape[:2]
        
        # Encontrar límites del panorama
        min_x, min_y = 0, 0
        max_x, max_y = w_ref, h_ref
        
        for i, H in enumerate(self.homographies):
            if i == self.reference_idx:
                continue
            
            h, w = images[i].shape[:2]
            corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            min_x = min(min_x, transformed_corners[:, 0, 0].min())
            min_y = min(min_y, transformed_corners[:, 0, 1].min())
            max_x = max(max_x, transformed_corners[:, 0, 0].max())
            max_y = max(max_y, transformed_corners[:, 0, 1].max())
        
        # Dimensiones del canvas
        canvas_width = int(np.ceil(max_x - min_x))
        canvas_height = int(np.ceil(max_y - min_y))
        offset = (int(-min_x), int(-min_y))
        
        print(f"Tamaño del panorama: {canvas_width}x{canvas_height}")
        
        # Transformar y fusionar imágenes
        result = None
        result_mask = None
        
        for i, (img, H) in enumerate(zip(images, self.homographies)):
            print(f"Procesando imagen {i+1}/{len(images)}...")
            
            # Warp imagen
            warped = warp_image(img, H, (canvas_height, canvas_width), offset)
            warped_mask = create_mask(img.shape, H, offset, (canvas_height, canvas_width))
            
            # Fusionar
            if result is None:
                result = warped
                result_mask = warped_mask
            else:
                # Obtener pesos relativos
                if self.image_weights is not None and len(self.image_weights) == len(images):
                    # Para la primera imagen (result), usar su peso individual
                    # Para la nueva imagen (warped), usar su peso individual
                    # Solo necesitamos los pesos relativos entre las dos imágenes
                    weight_result = self.image_weights[0] if i == 1 else 1.0
                    weight_new = self.image_weights[i] if i < len(self.image_weights) else 1.0
                else:
                    weight_result = 1.0
                    weight_new = 1.0
                
                if self.blend_method == 'simple':
                    result = simple_blend(result, warped, result_mask, warped_mask,
                                         weight1=weight_result, weight2=weight_new)
                elif self.blend_method == 'feather':
                    result = feather_blend(result, warped, result_mask, warped_mask,
                                          weight1=weight_result, weight2=weight_new)
                elif self.blend_method == 'multiband':
                    # Multiband no soporta pesos aún, usar sin pesos
                    result = multiband_blend(result, warped, result_mask, warped_mask)
                
                # Actualizar máscara
                result_mask = np.maximum(result_mask, warped_mask)
        
        return result


def compute_reprojection_error(points1: np.ndarray, points2: np.ndarray,
                               H: np.ndarray) -> float:
    """
    Calcula el error de reproyección promedio.
    
    Args:
        points1: Puntos originales
        points2: Puntos transformados
        H: Homografía
        
    Returns:
        Error promedio en píxeles
    """
    if len(points1) == 0:
        return float('inf')
    
    # Transformar puntos1 usando H
    points1_homogeneous = np.hstack([points1, np.ones((len(points1), 1))])
    projected = (H @ points1_homogeneous.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    
    # Calcular distancias
    errors = np.linalg.norm(projected - points2, axis=1)
    
    return np.mean(errors)


