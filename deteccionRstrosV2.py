import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

# --- CONFIGURACIÓN ---
MODEL_PATH = r"C:\Users\Bravo\PycharmProjects\Robotica\deteccionRostro\blaze_face_short_range.tflite"

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
FaceDetectorResult = mp.tasks.vision.FaceDetectorResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Variable global para el resultado
LATEST_RESULT = None


# --- FUNCIÓN AUXILIAR PARA DIBUJAR DETECCIONES ---
def draw_face_detections(image, result):
    annotated_image = image.copy()
    height, width, _ = image.shape
    
    # Contador de rostros
    face_count = len(result.detections) if result.detections else 0
    
    # Color base para las detecciones
    base_color = (255, 0, 0)  # Azul
    
    if result.detections:
        for i, detection in enumerate(result.detections):
            # 1. Dibujar Bounding Box
            bbox = detection.bounding_box
            start_point = (bbox.origin_x, bbox.origin_y)
            end_point = (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height)
            
            # Usar color diferente para cada rostro (rotación de colores)
            color = [
                (255, 0, 0),    # Azul
                (0, 255, 0),    # Verde
                (0, 0, 255),    # Rojo
                (255, 255, 0),  # Cyan
                (255, 0, 255),  # Magenta
                (0, 255, 255)   # Amarillo
            ][i % 6]
            
            cv2.rectangle(annotated_image, start_point, end_point, color, 3)
            
            # 2. Dibujar Puntos Clave
            if detection.keypoints:
                for keypoint in detection.keypoints:
                    kx = int(keypoint.x * width)
                    ky = int(keypoint.y * height)
                    cv2.circle(annotated_image, (kx, ky), 5, (0, 255, 0), -1)
            
            # 3. Mostrar ID y Confianza
            score = detection.categories[0].score
            label = f"Rostro {i+1}: {score:.2f}"
            cv2.putText(annotated_image, label, 
                       (bbox.origin_x, bbox.origin_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return annotated_image, face_count


# --- APLICAR REGLA DE PRIVACIDAD ---
def apply_privacy_rule(image, face_count):
    height, width = image.shape[:2]
    
    # Regla de privacidad:
    # Permitido: exactamente 1 rostro en pantalla
    # No permitido: 0 rostros o 2+ rostros
    
    if face_count == 0:
        # Caso: No hay rostros
        message = "SIN ROSTROS DETECTADOS"
        color = (0, 165, 255)  # Naranja
        action = "puchale a la luz o acercate"
        
        # Crear overlay semitransparente
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        
        # Mostrar mensaje principal
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        cv2.putText(image, message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        
        # Mostrar acción sugerida
        action_size = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        action_x = (width - action_size[0]) // 2
        action_y = text_y + 50
        cv2.putText(image, action, (action_x, action_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Estado del sistema
        status = "SISTEMA DON WOR"
        status_color = (0, 0, 255)
        
    elif face_count == 1:
        # Caso: Exactamente 1 rostro - PERMITIDO
        message = "PRIVACIDAD GARANTIZADA"
        color = (0, 255, 0)  # Verde
        action = "Único rostro - AUTORIZO"
        
        # Estado del sistema
        status = "SISTEMA ACTIVO"
        status_color = (0, 255, 0)
        
        # Mostrar mensaje en la parte superior
        cv2.putText(image, message, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(image, action, (20, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        
    else:
        # Caso: 2 o más rostros - NO PERMITIDO
        message = "DEMASIADOS ROSTROS"
        color = (0, 0, 255)  # Rojo
        action = f"Son {face_count} rostros - NO AUTORIZO"
        
        # Crear overlay rojo semitransparente
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 100), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)
        
        # Mostrar mensaje principal centrado
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2 - 50
        cv2.putText(image, message, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        
        # Mostrar acción
        action_size = cv2.getTextSize(action, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        action_x = (width - action_size[0]) // 2
        action_y = text_y + 50
        cv2.putText(image, action, (action_x, action_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Estado del sistema
        status = "SISTEM IS BLOQUEADO"
        status_color = (0, 0, 255)
        
        # Efecto visual adicional: Parpadeo de alerta
        current_time = time.time()
        if int(current_time * 2) % 2 == 0:  # Parpadea cada 0.5 segundos
            cv2.rectangle(image, (50, height - 100), (width - 50, height - 50), 
                         (0, 0, 255), -1)
            cv2.putText(image, "¡WATERS!",
                       (width // 2 - 150, height - 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Mostrar estado del sistema en la esquina superior derecha
    cv2.putText(image, status, (width - 250, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    # Mostrar contador de rostros en la esquina superior izquierda
    counter_text = f"Rostros detectados: {face_count}"
    cv2.putText(image, counter_text, (20, height - 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return image


# --- CALLBACK ---
def save_result(result: FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
    global LATEST_RESULT
    LATEST_RESULT = result


# --- INICIALIZACIÓN ---
options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    min_detection_confidence=0.5,
    result_callback=save_result
)

# --- BUCLE PRINCIPAL ---
cap = cv2.VideoCapture(0)

# Información inicial
print("=" * 60)
print("SISTEMA DE DETECCIÓN DE MÚLTIPLES ROSTROS")
print("=" * 60)
print("REGLAS DE PRIVACIDAD:")
print("1 rostro: Acceso permitido")
print("0 rostros: Sistema inactivo")
print("2+ rostros: Acceso denegado")
print("=" * 60)
print("Presiona 'q' para salir")
print("=" * 60)

with FaceDetector.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preparar imagen para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Lógica de timestamp
        timestamp_ms = int(time.time() * 1000)
        
        # Detección Asíncrona
        detector.detect_async(mp_image, timestamp_ms)
        
        # Visualización y aplicación de reglas
        if LATEST_RESULT:
            # Dibujar detecciones y obtener conteo
            frame, face_count = draw_face_detections(frame, LATEST_RESULT)
            
            # Aplicar regla de privacidad
            frame = apply_privacy_rule(frame, face_count)
            
            # Mostrar información adicional en consola
            if face_count > 0:
                print(f"\rRostros detectados: {face_count} | Última actualización: {time.strftime('%H:%M:%S')}", 
                      end="", flush=True)
        
        # Mostrar frame
        cv2.imshow('Sistema de Detección Facial - Reglas de Privacidad', frame)
        
        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("\n\nSistema finalizado.")