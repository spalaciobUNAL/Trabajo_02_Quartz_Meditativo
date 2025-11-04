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
                         'Click derecho para limpiar selección')
        
        self.points = []
        self.lines = []
        self.texts = []
        
        def onclick(event):
            if event.inaxes != self.ax:
                return
            
            if event.button == 1:  # Click izquierdo
                x, y = int(event.xdata), int(event.ydata)
                self.points.append((x, y))
                
                # Dibujar punto
                self.ax.plot(x, y, 'ro', markersize=8)
                
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
                
                self.fig.canvas.draw()
            
            elif event.button == 3:  # Click derecho - limpiar
                self.points = []
                # Limpiar visualización temporal
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


def create_measurement_report(measurements: List[Dict], 
                             calibrator: Calibrator,
                             save_path: Optional[str] = None) -> None:
    """
    Crea un reporte visual de las mediciones.
    
    Args:
        measurements: Lista de mediciones
        calibrator: Calibrador usado
        save_path: Ruta donde guardar el reporte (opcional)
    """
    if not measurements:
        print("No hay mediciones para reportar")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Reporte de Mediciones', fontsize=16, fontweight='bold')
    
    # 1. Histograma de distancias
    ax = axes[0, 0]
    distances_cm = [m['cm_distance'] for m in measurements if m.get('cm_distance') is not None]
    
    if distances_cm:
        ax.hist(distances_cm, bins=min(20, len(distances_cm)), 
               color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(distances_cm), color='red', linestyle='--', 
                  linewidth=2, label=f'Media: {np.mean(distances_cm):.2f} cm')
        ax.set_xlabel('Distancia (cm)', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title('Distribución de Distancias Medidas')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No hay mediciones en CM', 
               ha='center', va='center', transform=ax.transAxes)
    
    # 2. Tabla de mediciones
    ax = axes[0, 1]
    ax.axis('off')
    
    table_data = [['ID', 'Distancia (cm)', 'Incert. (%)']]
    for i, m in enumerate(measurements, 1):
        if m.get('cm_distance') is not None:
            dist = f"{m['cm_distance']:.2f}"
            unc = f"{m.get('uncertainty_percent', 0):.2f}" if m.get('uncertainty_percent') else 'N/A'
        else:
            dist = 'N/A'
            unc = 'N/A'
        table_data.append([str(i), dist, unc])
    
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                    colWidths=[0.2, 0.4, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Estilo para encabezado
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Tabla de Mediciones')
    
    # 3. Información de calibración
    ax = axes[1, 0]
    ax.axis('off')
    
    cal_info = calibrator.get_calibration_info()
    info_text = f"""
    INFORMACIÓN DE CALIBRACIÓN
    
    Estado: {'Calibrado' if cal_info['calibrated'] else 'No calibrado'}
    Escala: {cal_info['pixels_per_cm']:.4f} píxeles/cm
    Referencias: {cal_info['n_references']}
    Incertidumbre: {cal_info['uncertainty_percent']:.2f}% 
                   (si está disponible)
    
    Objetos de Referencia:
    """
    
    for ref in cal_info['references']:
        info_text += f"\n  • {ref['name']}: {ref['real_length_cm']:.1f} cm"
    
    ax.text(0.1, 0.5, info_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # 4. Estadísticas
    ax = axes[1, 1]
    
    if distances_cm:
        stats_text = f"""
        ESTADÍSTICAS
        
        N° Mediciones: {len(distances_cm)}
        Media: {np.mean(distances_cm):.2f} cm
        Mediana: {np.median(distances_cm):.2f} cm
        Desv. Std: {np.std(distances_cm):.2f} cm
        Mínimo: {np.min(distances_cm):.2f} cm
        Máximo: {np.max(distances_cm):.2f} cm
        Rango: {np.max(distances_cm) - np.min(distances_cm):.2f} cm
        """
        
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='center',
               fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    else:
        ax.text(0.5, 0.5, 'No hay suficientes datos\npara estadísticas',
               ha='center', va='center', transform=ax.transAxes)
    
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reporte guardado en: {save_path}")
    
    plt.show()


def analyze_measurement_uncertainty(calibrator: Calibrator,
                                   n_simulations: int = 1000) -> Dict:
    """
    Analiza la incertidumbre en las mediciones mediante simulación Monte Carlo.
    
    Args:
        calibrator: Calibrador a analizar
        n_simulations: Número de simulaciones
        
    Returns:
        Diccionario con análisis de incertidumbre
    """
    if not calibrator.calibrated or len(calibrator.reference_objects) < 2:
        return {'error': 'Calibrador no válido o insuficientes referencias'}
    
    # Extraer escalas de referencias
    scales = np.array([obj['scale'] for obj in calibrator.reference_objects])
    mean_scale = np.mean(scales)
    std_scale = np.std(scales)
    
    # Simular mediciones
    simulated_scales = np.random.normal(mean_scale, std_scale, n_simulations)
    
    # Para una distancia de referencia de 100 cm
    reference_cm = 100
    simulated_measurements = reference_cm * simulated_scales
    
    analysis = {
        'mean_scale': mean_scale,
        'std_scale': std_scale,
        'cv_percent': (std_scale / mean_scale) * 100,  # Coeficiente de variación
        'reference_distance_cm': reference_cm,
        'simulated_mean': np.mean(simulated_measurements),
        'simulated_std': np.std(simulated_measurements),
        'confidence_interval_95': np.percentile(simulated_measurements, [2.5, 97.5])
    }
    
    return analysis


