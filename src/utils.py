"""
Módulo de utilidades para el proyecto de registro de imágenes.
Contiene funciones auxiliares para visualización y procesamiento.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
import os


def load_image(image_path: str, grayscale: bool = False) -> np.ndarray:
    """
    Carga una imagen desde un archivo.
    
    Args:
        image_path: Ruta al archivo de imagen
        grayscale: Si True, carga la imagen en escala de grises
        
    Returns:
        Imagen como array de numpy
    """
    if grayscale:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if img is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    return img


def save_image(image: np.ndarray, output_path: str, is_rgb: bool = True) -> None:
    """
    Guarda una imagen en un archivo.
    
    Args:
        image: Imagen como array de numpy
        output_path: Ruta donde guardar la imagen
        is_rgb: Si True, la imagen está en formato RGB (se convierte a BGR para OpenCV)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if is_rgb and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, image)


def visualize_images(images: List[np.ndarray], titles: List[str] = None, 
                     figsize: Tuple[int, int] = (15, 5), 
                     save_path: Optional[str] = None) -> None:
    """
    Visualiza múltiples imágenes en una figura.
    
    Args:
        images: Lista de imágenes a visualizar
        titles: Lista de títulos para cada imagen
        figsize: Tamaño de la figura
        save_path: Si se proporciona, guarda la figura en esta ruta
    """
    n_images = len(images)
    
    if titles is None:
        titles = [f'Imagen {i+1}' for i in range(n_images)]
    
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def draw_keypoints(image: np.ndarray, keypoints: List, 
                   color: Tuple[int, int, int] = (0, 255, 0),
                   size: int = 4) -> np.ndarray:
    """
    Dibuja keypoints detectados en una imagen.
    
    Args:
        image: Imagen donde dibujar los keypoints
        keypoints: Lista de keypoints detectados
        color: Color de los keypoints en RGB
        size: Tamaño de los keypoints
        
    Returns:
        Imagen con keypoints dibujados
    """
    img_with_keypoints = image.copy()
    
    # Convertir a BGR para OpenCV
    if len(image.shape) == 3:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Dibujar keypoints
    color_bgr = (color[2], color[1], color[0])  # RGB a BGR
    img_with_kp = cv2.drawKeypoints(img_bgr, keypoints, None, 
                                     color=color_bgr,
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Convertir de vuelta a RGB
    img_with_kp = cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB)
    
    return img_with_kp


def draw_matches(img1: np.ndarray, kp1: List, 
                 img2: np.ndarray, kp2: List,
                 matches: List, num_matches: int = None,
                 match_color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Dibuja las correspondencias entre keypoints de dos imágenes.
    
    Args:
        img1: Primera imagen
        kp1: Keypoints de la primera imagen
        img2: Segunda imagen
        kp2: Keypoints de la segunda imagen
        matches: Lista de correspondencias
        num_matches: Número máximo de matches a dibujar (None = todos)
        match_color: Color de las líneas de correspondencia en RGB
        
    Returns:
        Imagen con las correspondencias dibujadas
    """
    if num_matches is not None:
        matches = matches[:num_matches]
    
    # Convertir imágenes a BGR
    if len(img1.shape) == 3:
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    else:
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    
    # Color en BGR
    color_bgr = (match_color[2], match_color[1], match_color[0])
    
    # Dibujar matches
    img_matches = cv2.drawMatches(img1_bgr, kp1, img2_bgr, kp2, matches, None,
                                   matchColor=color_bgr,
                                   singlePointColor=color_bgr,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Convertir a RGB
    img_matches = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    
    return img_matches


def compute_transformation_error(H_estimated: np.ndarray, 
                                 H_ground_truth: np.ndarray,
                                 image_shape: Tuple[int, int]) -> dict:
    """
    Calcula el error entre una homografía estimada y el ground truth.
    
    Args:
        H_estimated: Homografía estimada
        H_ground_truth: Homografía ground truth
        image_shape: Forma de la imagen (alto, ancho)
        
    Returns:
        Diccionario con métricas de error
    """
    height, width = image_shape[:2]
    
    # Esquinas de la imagen
    corners = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ]).reshape(-1, 1, 2)
    
    # Transformar esquinas con ambas homografías
    corners_estimated = cv2.perspectiveTransform(corners, H_estimated)
    corners_gt = cv2.perspectiveTransform(corners, H_ground_truth)
    
    # RMSE en las esquinas
    rmse = np.sqrt(np.mean((corners_estimated - corners_gt) ** 2))
    
    # Error angular (diferencia en rotación)
    # Extraer la parte de rotación de las matrices
    H_est_norm = H_estimated / H_estimated[2, 2]
    H_gt_norm = H_ground_truth / H_ground_truth[2, 2]
    
    # Calcular el error en la matriz
    matrix_error = np.linalg.norm(H_est_norm - H_gt_norm, 'fro')
    
    return {
        'rmse': rmse,
        'matrix_frobenius_error': matrix_error,
        'corner_errors': np.linalg.norm(corners_estimated - corners_gt, axis=2).flatten()
    }


def create_panorama_canvas(images: List[np.ndarray], 
                           homographies: List[np.ndarray],
                           reference_idx: int = 0) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Crea un canvas para el panorama y calcula los offsets necesarios.
    
    Args:
        images: Lista de imágenes a fusionar
        homographies: Lista de homografías (respecto a la imagen de referencia)
        reference_idx: Índice de la imagen de referencia
        
    Returns:
        Canvas inicializado y lista de offsets para cada imagen
    """
    h, w = images[reference_idx].shape[:2]
    
    # Calcular límites del panorama
    min_x, min_y = 0, 0
    max_x, max_y = w, h
    
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    
    for i, H in enumerate(homographies):
        if i == reference_idx:
            continue
        
        h_i, w_i = images[i].shape[:2]
        corners_i = np.float32([[0, 0], [w_i, 0], [w_i, h_i], [0, h_i]]).reshape(-1, 1, 2)
        
        # Transformar esquinas
        transformed_corners = cv2.perspectiveTransform(corners_i, H)
        
        # Actualizar límites
        min_x = min(min_x, transformed_corners[:, 0, 0].min())
        min_y = min(min_y, transformed_corners[:, 0, 1].min())
        max_x = max(max_x, transformed_corners[:, 0, 0].max())
        max_y = max(max_y, transformed_corners[:, 0, 1].max())
    
    # Dimensiones del canvas
    canvas_width = int(np.ceil(max_x - min_x))
    canvas_height = int(np.ceil(max_y - min_y))
    
    # Offset para centrar
    offset_x = int(-min_x)
    offset_y = int(-min_y)
    
    # Crear canvas vacío
    if len(images[0].shape) == 3:
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    else:
        canvas = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
    
    return canvas, (offset_x, offset_y)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza una imagen al rango [0, 255].
    
    Args:
        image: Imagen a normalizar
        
    Returns:
        Imagen normalizada
    """
    img_normalized = image.copy()
    
    if img_normalized.dtype == np.float32 or img_normalized.dtype == np.float64:
        img_normalized = (img_normalized - img_normalized.min()) / (img_normalized.max() - img_normalized.min())
        img_normalized = (img_normalized * 255).astype(np.uint8)
    
    return img_normalized


def resize_image(image: np.ndarray, scale: float = None, 
                 width: int = None, height: int = None) -> np.ndarray:
    """
    Redimensiona una imagen.
    
    Args:
        image: Imagen a redimensionar
        scale: Factor de escala (si se proporciona)
        width: Ancho deseado (si se proporciona)
        height: Alto deseado (si se proporciona)
        
    Returns:
        Imagen redimensionada
    """
    if scale is not None:
        new_width = int(image.shape[1] * scale)
        new_height = int(image.shape[0] * scale)
    elif width is not None and height is not None:
        new_width = width
        new_height = height
    elif width is not None:
        aspect_ratio = image.shape[0] / image.shape[1]
        new_width = width
        new_height = int(width * aspect_ratio)
    elif height is not None:
        aspect_ratio = image.shape[1] / image.shape[0]
        new_height = height
        new_width = int(height * aspect_ratio)
    else:
        raise ValueError("Debe proporcionar scale, width, height, o width y height")
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)


def plot_error_metrics(errors: dict, save_path: Optional[str] = None) -> None:
    """
    Visualiza métricas de error.
    
    Args:
        errors: Diccionario con métricas de error
        save_path: Ruta donde guardar la figura
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # RMSE
    if 'rmse' in errors:
        axes[0].bar(['RMSE'], [errors['rmse']], color='steelblue')
        axes[0].set_ylabel('Error (píxeles)')
        axes[0].set_title('Root Mean Square Error')
        axes[0].grid(axis='y', alpha=0.3)
    
    # Error por esquina
    if 'corner_errors' in errors:
        corners = ['Sup. Izq.', 'Sup. Der.', 'Inf. Der.', 'Inf. Izq.']
        axes[1].bar(corners, errors['corner_errors'], color='coral')
        axes[1].set_ylabel('Error (píxeles)')
        axes[1].set_title('Error por Esquina')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_progress(current: int, total: int, message: str = "") -> None:
    """
    Imprime el progreso de una operación.
    
    Args:
        current: Número de elemento actual
        total: Número total de elementos
        message: Mensaje adicional a mostrar
    """
    percentage = (current / total) * 100
    print(f"\rProgreso: {percentage:.1f}% ({current}/{total}) {message}", end='', flush=True)
    
    if current == total:
        print()  # Nueva línea al finalizar


