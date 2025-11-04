"""
Generador de imágenes sintéticas para validación del pipeline de registro.
Aplica transformaciones conocidas para poder comparar con ground truth.
"""

import cv2
import numpy as np
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt
import os


def create_test_image(width: int = 800, height: int = 600,
                     pattern_type: str = 'checkerboard') -> np.ndarray:
    """
    Crea una imagen de prueba con patrones reconocibles.
    
    Args:
        width: Ancho de la imagen
        height: Alto de la imagen
        pattern_type: Tipo de patrón ('checkerboard', 'circles', 'mixed')
        
    Returns:
        Imagen sintética
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if pattern_type == 'checkerboard':
        # Tablero de ajedrez
        square_size = 50
        for i in range(0, height, square_size):
            for j in range(0, width, square_size):
                if ((i // square_size) + (j // square_size)) % 2 == 0:
                    img[i:i+square_size, j:j+square_size] = [255, 255, 255]
                else:
                    img[i:i+square_size, j:j+square_size] = [50, 50, 50]
    
    elif pattern_type == 'circles':
        # Círculos de colores
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255)
        ]
        
        img.fill(200)  # Fondo gris claro
        
        for i, color in enumerate(colors):
            x = (i % 3) * width // 3 + width // 6
            y = (i // 3) * height // 2 + height // 4
            cv2.circle(img, (x, y), 60, color, -1)
            cv2.circle(img, (x, y), 60, (0, 0, 0), 3)
    
    elif pattern_type == 'mixed':
        # Patrón mixto
        img.fill(220)
        
        # Tablero pequeño en esquinas
        square_size = 30
        for corner_y, corner_x in [(0, 0), (0, width-300), (height-300, 0)]:
            for i in range(10):
                for j in range(10):
                    y = corner_y + i * square_size
                    x = corner_x + j * square_size
                    if (i + j) % 2 == 0:
                        img[y:y+square_size, x:x+square_size] = [255, 255, 255]
        
        # Círculos en el centro
        cv2.circle(img, (width//2, height//2), 100, (255, 0, 0), -1)
        cv2.circle(img, (width//2, height//2), 70, (0, 255, 0), -1)
        cv2.circle(img, (width//2, height//2), 40, (0, 0, 255), -1)
        
        # Líneas
        for i in range(5):
            y = height // 6 * (i + 1)
            cv2.line(img, (0, y), (width, y), (100, 100, 100), 2)
    
    # Agregar texto para orientación
    cv2.putText(img, 'TOP', (width//2 - 30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'BOTTOM', (width//2 - 60, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img


def apply_rotation(image: np.ndarray, angle_degrees: float,
                  center: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica rotación a una imagen.
    
    Args:
        image: Imagen a rotar
        angle_degrees: Ángulo de rotación en grados
        center: Centro de rotación (x, y). Si None, usa el centro de la imagen
        
    Returns:
        Tupla (imagen rotada, matriz de transformación)
    """
    h, w = image.shape[:2]
    
    if center is None:
        center = (w // 2, h // 2)
    
    # Matriz de rotación
    M = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Calcular nuevo tamaño para que quepa toda la imagen
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Ajustar matriz de traslación
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Aplicar rotación
    rotated = cv2.warpAffine(image, M, (new_w, new_h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(128, 128, 128))
    
    # Convertir a homografía 3x3
    H = np.vstack([M, [0, 0, 1]])
    
    return rotated, H


def apply_translation(image: np.ndarray, tx: float, ty: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica traslación a una imagen.
    
    Args:
        image: Imagen a trasladar
        tx: Traslación en x (píxeles)
        ty: Traslación en y (píxeles)
        
    Returns:
        Tupla (imagen trasladada, matriz de transformación)
    """
    h, w = image.shape[:2]
    
    # Matriz de traslación
    M = np.float32([[1, 0, tx],
                    [0, 1, ty]])
    
    # Calcular nuevo tamaño
    new_w = w + abs(int(tx))
    new_h = h + abs(int(ty))
    
    # Aplicar traslación
    translated = cv2.warpAffine(image, M, (new_w, new_h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(128, 128, 128))
    
    # Convertir a homografía 3x3
    H = np.vstack([M, [0, 0, 1]])
    
    return translated, H


def apply_scale(image: np.ndarray, scale_x: float, scale_y: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica escalado a una imagen.
    
    Args:
        image: Imagen a escalar
        scale_x: Factor de escala en x
        scale_y: Factor de escala en y (si None, usa scale_x)
        
    Returns:
        Tupla (imagen escalada, matriz de transformación)
    """
    if scale_y is None:
        scale_y = scale_x
    
    h, w = image.shape[:2]
    
    # Nuevo tamaño
    new_w = int(w * scale_x)
    new_h = int(h * scale_y)
    
    # Aplicar escalado
    scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Matriz de transformación
    H = np.array([[scale_x, 0, 0],
                  [0, scale_y, 0],
                  [0, 0, 1]], dtype=np.float32)
    
    return scaled, H


def apply_perspective(image: np.ndarray, 
                     src_points: np.ndarray = None,
                     dst_points: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica transformación de perspectiva a una imagen.
    
    Args:
        image: Imagen a transformar
        src_points: Puntos fuente (4 puntos)
        dst_points: Puntos destino (4 puntos)
        
    Returns:
        Tupla (imagen transformada, matriz de homografía)
    """
    h, w = image.shape[:2]
    
    if src_points is None:
        # Puntos por defecto: esquinas de la imagen
        src_points = np.float32([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ])
    
    if dst_points is None:
        # Transformación de perspectiva por defecto
        offset = min(w, h) * 0.15
        dst_points = np.float32([
            [offset, offset],
            [w - offset, offset * 0.5],
            [w - offset * 0.5, h - offset],
            [offset * 0.5, h - offset * 0.5]
        ])
    
    # Calcular homografía
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Aplicar transformación
    transformed = cv2.warpPerspective(image, H, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(128, 128, 128))
    
    return transformed, H


def apply_combined_transform(image: np.ndarray,
                            rotation: float = 0,
                            translation: Tuple[float, float] = (0, 0),
                            scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica una combinación de transformaciones.
    
    Args:
        image: Imagen a transformar
        rotation: Ángulo de rotación en grados
        translation: Traslación (tx, ty) en píxeles
        scale: Factor de escala
        
    Returns:
        Tupla (imagen transformada, matriz de homografía combinada)
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Construir matriz de transformación
    # Orden: Escala -> Rotación -> Traslación
    
    # Matriz de escala
    S = np.array([[scale, 0, 0],
                  [0, scale, 0],
                  [0, 0, 1]], dtype=np.float32)
    
    # Matriz de rotación (alrededor del centro)
    angle_rad = np.deg2rad(rotation)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    R = np.array([[cos_a, -sin_a, center[0] * (1 - cos_a) + center[1] * sin_a],
                  [sin_a, cos_a, center[1] * (1 - cos_a) - center[0] * sin_a],
                  [0, 0, 1]], dtype=np.float32)
    
    # Matriz de traslación
    T = np.array([[1, 0, translation[0]],
                  [0, 1, translation[1]],
                  [0, 0, 1]], dtype=np.float32)
    
    # Combinar transformaciones
    H = T @ R @ S
    
    # Calcular tamaño de salida
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    min_x = transformed_corners[:, 0, 0].min()
    min_y = transformed_corners[:, 0, 1].min()
    max_x = transformed_corners[:, 0, 0].max()
    max_y = transformed_corners[:, 0, 1].max()
    
    # Ajustar homografía para offset
    H_offset = np.array([[1, 0, -min_x],
                        [0, 1, -min_y],
                        [0, 0, 1]], dtype=np.float32)
    
    H_final = H_offset @ H
    
    # Aplicar transformación
    new_w = int(np.ceil(max_x - min_x))
    new_h = int(np.ceil(max_y - min_y))
    
    transformed = cv2.warpPerspective(image, H_final, (new_w, new_h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(128, 128, 128))
    
    return transformed, H


def generate_synthetic_dataset(base_image: np.ndarray = None,
                              n_images: int = 3,
                              output_dir: str = 'data/synthetic') -> List[Dict]:
    """
    Genera un conjunto de imágenes sintéticas con transformaciones conocidas.
    
    Args:
        base_image: Imagen base (si None, crea una sintética)
        n_images: Número de imágenes a generar
        output_dir: Directorio de salida
        
    Returns:
        Lista de diccionarios con información de cada imagen
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear imagen base si no se proporciona
    if base_image is None:
        base_image = create_test_image(800, 600, 'mixed')
    
    # Guardar imagen base
    base_path = os.path.join(output_dir, 'base_image.png')
    cv2.imwrite(base_path, cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR))
    
    dataset = [{
        'id': 0,
        'image': base_image,
        'path': base_path,
        'transformation': np.eye(3),
        'parameters': {'rotation': 0, 'translation': (0, 0), 'scale': 1.0}
    }]
    
    # Generar transformaciones variadas
    transformations = [
        {'rotation': 15, 'translation': (50, 30), 'scale': 1.0},
        {'rotation': -10, 'translation': (-30, 40), 'scale': 0.95},
        {'rotation': 20, 'translation': (20, -50), 'scale': 1.05},
        {'rotation': -15, 'translation': (60, 20), 'scale': 0.9},
    ]
    
    for i in range(1, n_images):
        params = transformations[min(i-1, len(transformations)-1)]
        
        # Aplicar transformación
        transformed, H = apply_combined_transform(
            base_image,
            rotation=params['rotation'],
            translation=params['translation'],
            scale=params['scale']
        )
        
        # Guardar imagen
        img_path = os.path.join(output_dir, f'transformed_{i}.png')
        cv2.imwrite(img_path, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))
        
        dataset.append({
            'id': i,
            'image': transformed,
            'path': img_path,
            'transformation': H,
            'parameters': params
        })
    
    print(f"Dataset sintético generado en: {output_dir}")
    print(f"Número de imágenes: {len(dataset)}")
    
    return dataset


def visualize_synthetic_dataset(dataset: List[Dict], save_path: str = None):
    """
    Visualiza el dataset sintético generado.
    
    Args:
        dataset: Lista de diccionarios con las imágenes
        save_path: Ruta donde guardar la visualización
    """
    n = len(dataset)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for i, data in enumerate(dataset):
        ax = axes[i]
        ax.imshow(data['image'])
        
        params = data['parameters']
        title = f"Imagen {data['id']}\n"
        if data['id'] == 0:
            title += "Base (sin transformación)"
        else:
            title += (f"Rot: {params['rotation']}°, "
                     f"Trans: {params['translation']}, "
                     f"Scale: {params['scale']}")
        
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Ocultar ejes sobrantes
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Visualización guardada en: {save_path}")
    
    plt.show()


if __name__ == '__main__':
    # Generar dataset de ejemplo
    print("Generando dataset sintético...")
    dataset = generate_synthetic_dataset(n_images=4)
    
    print("\nVisualizando dataset...")
    visualize_synthetic_dataset(dataset, 'data/synthetic/dataset_overview.png')
    
    print("\nDataset generado exitosamente!")


