"""
Módulo de calibración y medición para estimación de dimensiones reales.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Rectangle, FancyArrowPatch
import warnings


class Calibrator:
    """
    Clase para calibrar el sistema usando objetos de referencia.
    """
    
    def __init__(self):
        """Inicializa el calibrador."""
        self.reference_objects = []
        self.pixels_per_cm = None
        self.calibrated = False
    
    def add_reference_object(self, name: str, pixel_length: float, real_length_cm: float):
        """
        Agrega un objeto de referencia para calibración.
        
        Args:
            name: Nombre del objeto
            pixel_length: Longitud en píxeles medida en la imagen
            real_length_cm: Longitud real en centímetros
        """
        scale = pixel_length / real_length_cm
        
        self.reference_objects.append({
            'name': name,
            'pixel_length': pixel_length,
            'real_length_cm': real_length_cm,
            'scale': scale
        })
        
        # Recalcular escala promedio
        self._update_calibration()
    
    def _update_calibration(self):
        """Actualiza la calibración basándose en todos los objetos de referencia."""
        if not self.reference_objects:
            self.calibrated = False
            return
        
        # Usar promedio de todas las escalas
        scales = [obj['scale'] for obj in self.reference_objects]
        self.pixels_per_cm = np.mean(scales)
        self.calibrated = True
        
        print(f"Calibración actualizada: {self.pixels_per_cm:.4f} píxeles/cm")
        
        # Mostrar información de cada objeto
        for obj in self.reference_objects:
            error = abs(obj['scale'] - self.pixels_per_cm) / self.pixels_per_cm * 100
            print(f"  - {obj['name']}: {obj['scale']:.4f} px/cm (error: {error:.2f}%)")
    
    def pixel_to_cm(self, pixel_length: float) -> Optional[float]:
        """
        Convierte longitud en píxeles a centímetros.
        
        Args:
            pixel_length: Longitud en píxeles
            
        Returns:
            Longitud en centímetros, o None si no está calibrado
        """
        if not self.calibrated:
            warnings.warn("El sistema no está calibrado")
            return None
        
        return pixel_length / self.pixels_per_cm
    
    def cm_to_pixel(self, cm_length: float) -> Optional[float]:
        """
        Convierte longitud en centímetros a píxeles.
        
        Args:
            cm_length: Longitud en centímetros
            
        Returns:
            Longitud en píxeles, o None si no está calibrado
        """
        if not self.calibrated:
            warnings.warn("El sistema no está calibrado")
            return None
        
        return cm_length * self.pixels_per_cm
    
    def get_uncertainty(self) -> Optional[float]:
        """
        Calcula la incertidumbre en la calibración.
        
        Returns:
            Desviación estándar relativa (%), o None si no hay suficientes referencias
        """
        if len(self.reference_objects) < 2:
            return None
        
        scales = [obj['scale'] for obj in self.reference_objects]
        std = np.std(scales)
        mean = np.mean(scales)
        
        return (std / mean) * 100  # Porcentaje
    
    def get_calibration_info(self) -> Dict:
        """
        Retorna información completa de la calibración.
        
        Returns:
            Diccionario con información de calibración
        """
        info = {
            'calibrated': self.calibrated,
            'pixels_per_cm': self.pixels_per_cm,
            'n_references': len(self.reference_objects),
            'references': self.reference_objects.copy(),
            'uncertainty_percent': self.get_uncertainty()
        }
        
        return info


class InteractiveMeasurementTool:
    """
    Herramienta interactiva para medir distancias en una imagen.
    """
    
    def __init__(self, image: np.ndarray, calibrator: Calibrator):
        """
        Inicializa la herramienta de medición.
        
        Args:
            image: Imagen donde realizar mediciones
            calibrator: Calibrador con la escala del sistema
        """
        self.image = image
        self.calibrator = calibrator
        self.points = []
        self.measurements = []
        self.fig = None
        self.ax = None
        self.current_line = None
    
    def measure_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> Dict:
        """
        Mide la distancia entre dos puntos.
        
        Args:
            point1: Primer punto (x, y)
            point2: Segundo punto (x, y)
            
        Returns:
            Diccionario con información de la medición
        """
        # Calcular distancia en píxeles
        pixel_distance = np.linalg.norm(np.array(point1) - np.array(point2))
        
        # Convertir a centímetros
        cm_distance = self.calibrator.pixel_to_cm(pixel_distance)
        
        # Calcular incertidumbre
        uncertainty = self.calibrator.get_uncertainty()
        if uncertainty is not None and cm_distance is not None:
            uncertainty_cm = cm_distance * (uncertainty / 100)
        else:
            uncertainty_cm = None
        
        measurement = {
            'point1': point1,
            'point2': point2,
            'pixel_distance': pixel_distance,
            'cm_distance': cm_distance,
            'uncertainty_cm': uncertainty_cm,
            'uncertainty_percent': uncertainty
        }
        
        self.measurements.append(measurement)
        
        return measurement
    
    def launch_interactive_tool(self):
        """
        Lanza la herramienta interactiva para medición con clicks.
        """
        if not self.calibrator.calibrated:
            print("Advertencia: El sistema no está calibrado. Las mediciones serán solo en píxeles.")
        
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(self.image)
        self.ax.set_title('Click para seleccionar dos puntos y medir distancia\n'
                         'Click derecho para eliminar el último punto')
        
        self.points = []
        self.lines = []
        self.texts = []
        self.markers = []  # Lista para guardar los marcadores de puntos
        
        def onclick(event):
            if event.inaxes != self.ax:
                return
            
            if event.button == 1:  # Click izquierdo
                x, y = int(event.xdata), int(event.ydata)
                self.points.append((x, y))
                
                # Dibujar punto y guardar referencia
                marker, = self.ax.plot(x, y, 'ro', markersize=4)
                self.markers.append(marker)
                
                if len(self.points) == 2:
                    # Medir distancia
                    measurement = self.measure_distance(self.points[0], self.points[1])
                    
                    # Dibujar línea
                    line = self.ax.plot([self.points[0][0], self.points[1][0]],
                                       [self.points[0][1], self.points[1][1]],
                                       'r-', linewidth=2)[0]
                    self.lines.append(line)
                    
                    # Mostrar medición
                    mid_x = (self.points[0][0] + self.points[1][0]) / 2
                    mid_y = (self.points[0][1] + self.points[1][1]) / 2
                    
                    if measurement['cm_distance'] is not None:
                        if measurement['uncertainty_cm'] is not None:
                            text = f"{measurement['cm_distance']:.2f} ± {measurement['uncertainty_cm']:.2f} cm"
                        else:
                            text = f"{measurement['cm_distance']:.2f} cm"
                    else:
                        text = f"{measurement['pixel_distance']:.1f} px"
                    
                    t = self.ax.text(mid_x, mid_y, text,
                                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                                    fontsize=10, ha='center')
                    self.texts.append(t)
                    
                    # Limpiar puntos para próxima medición
                    self.points = []
                    self.markers = []
                
                self.fig.canvas.draw()
            
            elif event.button == 3:  # Click derecho - eliminar último punto
                if self.points:
                    # Eliminar el último punto de la lista
                    self.points.pop()
                    
                    # Eliminar el último marcador visual
                    if self.markers:
                        marker = self.markers.pop()
                        marker.remove()
                    
                    self.fig.canvas.draw()
        
        self.fig.canvas.mpl_connect('button_press_event', onclick)
        
        plt.show()
    
    def get_measurements_table(self) -> str:
        """
        Retorna una tabla formateada con todas las mediciones.
        
        Returns:
            String con la tabla
        """
        if not self.measurements:
            return "No hay mediciones disponibles"
        
        table = "\n" + "="*80 + "\n"
        table += f"{'#':<5} {'Punto 1':<15} {'Punto 2':<15} {'Píxeles':<12} {'CM':<15} {'Incertidumbre'}\n"
        table += "="*80 + "\n"
        
        for i, m in enumerate(self.measurements, 1):
            p1_str = f"({m['point1'][0]}, {m['point1'][1]})"
            p2_str = f"({m['point2'][0]}, {m['point2'][1]})"
            px_str = f"{m['pixel_distance']:.2f}"
            
            if m['cm_distance'] is not None:
                cm_str = f"{m['cm_distance']:.2f}"
                if m['uncertainty_cm'] is not None:
                    unc_str = f"±{m['uncertainty_cm']:.2f} cm ({m['uncertainty_percent']:.2f}%)"
                else:
                    unc_str = "N/A"
            else:
                cm_str = "N/A"
                unc_str = "N/A"
            
            table += f"{i:<5} {p1_str:<15} {p2_str:<15} {px_str:<12} {cm_str:<15} {unc_str}\n"
        
        table += "="*80 + "\n"
        
        return table
    
    def save_measurements(self, filepath: str):
        """
        Guarda las mediciones en un archivo CSV.
        
        Args:
            filepath: Ruta del archivo CSV
        """
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ID', 'Point1_X', 'Point1_Y', 'Point2_X', 'Point2_Y',
                           'Pixel_Distance', 'CM_Distance', 'Uncertainty_CM', 'Uncertainty_Percent'])
            
            for i, m in enumerate(self.measurements, 1):
                writer.writerow([
                    i,
                    m['point1'][0], m['point1'][1],
                    m['point2'][0], m['point2'][1],
                    m['pixel_distance'],
                    m['cm_distance'] if m['cm_distance'] is not None else '',
                    m['uncertainty_cm'] if m['uncertainty_cm'] is not None else '',
                    m['uncertainty_percent'] if m['uncertainty_percent'] is not None else ''
                ])
        
        print(f"Mediciones guardadas en: {filepath}")


def measure_distance_simple(image: np.ndarray, point1: Tuple[int, int], 
                           point2: Tuple[int, int],
                           calibrator: Calibrator = None,
                           show_plot: bool = True) -> Dict:
    """
    Función simple para medir distancia entre dos puntos.
    
    Args:
        image: Imagen de referencia
        point1: Primer punto (x, y)
        point2: Segundo punto (x, y)
        calibrator: Calibrador opcional
        show_plot: Si True, muestra visualización
        
    Returns:
        Diccionario con la medición
    """
    # Calcular distancia en píxeles
    pixel_distance = np.linalg.norm(np.array(point1) - np.array(point2))
    
    measurement = {
        'point1': point1,
        'point2': point2,
        'pixel_distance': pixel_distance
    }
    
    # Convertir a centímetros si hay calibrador
    if calibrator is not None and calibrator.calibrated:
        cm_distance = calibrator.pixel_to_cm(pixel_distance)
        uncertainty = calibrator.get_uncertainty()
        
        measurement['cm_distance'] = cm_distance
        measurement['uncertainty_percent'] = uncertainty
        
        if uncertainty is not None:
            measurement['uncertainty_cm'] = cm_distance * (uncertainty / 100)
    
    # Visualizar si se solicita
    if show_plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(image)
        
        # Dibujar puntos
        ax.plot([point1[0], point2[0]], [point1[1], point2[1]], 'ro-', 
               markersize=10, linewidth=2)
        
        # Texto con medición
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2
        
        if 'cm_distance' in measurement:
            if 'uncertainty_cm' in measurement:
                text = (f"Distancia: {measurement['cm_distance']:.2f} ± "
                       f"{measurement['uncertainty_cm']:.2f} cm")
            else:
                text = f"Distancia: {measurement['cm_distance']:.2f} cm"
        else:
            text = f"Distancia: {pixel_distance:.1f} píxeles"
        
        ax.text(mid_x, mid_y, text,
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               fontsize=12, ha='center')
        
        ax.set_title('Medición de Distancia')
        ax.axis('off')
        plt.tight_layout()
        plt.show()
    
    return measurement





