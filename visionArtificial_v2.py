import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import math

# --- CONFIGURATION ---
MODEL_PATH = r"C:\Users\Bravo\PycharmProjects\Robotica\hand_landmarker.task"

# --- CONSTANTS ---
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variable to hold the latest result from the async callback
LATEST_RESULT = None

# --- GESTOS RECONOCIDOS ---
GESTOS = {
    "OK",
    "PULGAR_ARRIBA",
    "VICTORIA",
    "L",
    "NINGUNO": "..."
}

# --- FUNCIONES DE DETECCIÓN DE GESTOS ---

def is_okay_sign(hand_landmarks):
    # Extraer coordenadas
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    
    # Lógica 1: Pulgar e índice tocándose
    distance_thumb_index = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 +
        (thumb_tip.y - index_tip.y) ** 2
    )
    is_touching = distance_thumb_index < 0.05
    
    # Lógica 2: Otros dedos extendidos
    middle_extended = hand_landmarks[12].y < hand_landmarks[10].y
    ring_extended = hand_landmarks[16].y < hand_landmarks[14].y
    pinky_extended = hand_landmarks[20].y < hand_landmarks[18].y
    
    others_extended = middle_extended and ring_extended and pinky_extended
    
    return is_touching and others_extended

def is_thumbs_up(hand_landmarks):
    # Pulgar arriba: pulgar extendido, otros dedos cerrados
    
    # 1. Pulgar extendido (punta más alta que la base)
    thumb_extended = hand_landmarks[4].y < hand_landmarks[3].y
    
    # 2. Otros dedos cerrados
    # Para índice: punta más baja que la base (o casi igual)
    index_closed = hand_landmarks[8].y > hand_landmarks[6].y - 0.05
    middle_closed = hand_landmarks[12].y > hand_landmarks[10].y - 0.05
    ring_closed = hand_landmarks[16].y > hand_landmarks[14].y - 0.05
    pinky_closed = hand_landmarks[20].y > hand_landmarks[18].y - 0.05
    
    all_fingers_closed = index_closed and middle_closed and ring_closed and pinky_closed
    
    # 3. Pulgar apuntando hacia arriba (posición x cercana a la base del pulgar)
    thumb_upwards = abs(hand_landmarks[4].x - hand_landmarks[2].x) < 0.1
    
    return thumb_extended and all_fingers_closed and thumb_upwards

def is_victory_sign(hand_landmarks):
    # Victoria: índice y medio extendidos, otros dedos cerrados
    
    # 1. Índice y medio extendidos
    index_extended = hand_landmarks[8].y < hand_landmarks[6].y
    middle_extended = hand_landmarks[12].y < hand_landmarks[10].y
    
    # 2. Anular y meñique cerrados
    ring_closed = hand_landmarks[16].y > hand_landmarks[14].y - 0.05
    pinky_closed = hand_landmarks[20].y > hand_landmarks[18].y - 0.05
    
    # 3. Pulgar puede estar abierto o cerrado (flexible)
    # 4. Los dedos extendidos deben estar separados
    distance_index_middle = math.sqrt(
        (hand_landmarks[8].x - hand_landmarks[12].x) ** 2 +
        (hand_landmarks[8].y - hand_landmarks[12].y) ** 2
    )
    fingers_separated = distance_index_middle > 0.05
    
    return index_extended and middle_extended and ring_closed and pinky_closed and fingers_separated

def is_L_sign(hand_landmarks):
    # Letra L: pulgar e índice extendidos formando ángulo de 90°, otros dedos cerrados
    
    # 1. Pulgar extendido
    thumb_extended = hand_landmarks[4].y < hand_landmarks[3].y
    
    # 2. Índice extendido
    index_extended = hand_landmarks[8].y < hand_landmarks[6].y
    
    # 3. Otros dedos cerrados
    middle_closed = hand_landmarks[12].y > hand_landmarks[10].y - 0.05
    ring_closed = hand_landmarks[16].y > hand_landmarks[14].y - 0.05
    pinky_closed = hand_landmarks[20].y > hand_landmarks[18].y - 0.05
    
    # 4. Forman ángulo L (diferencia significativa en X o Y entre pulgar e índice)
    # Pulgar generalmente más horizontal, índice más vertical
    diff_x = abs(hand_landmarks[4].x - hand_landmarks[8].x)
    diff_y = abs(hand_landmarks[4].y - hand_landmarks[8].y)
    
    # Ambos deben estar extendidos y en diferente orientación
    forms_L_shape = diff_x > 0.1 and diff_y > 0.1
    
    return thumb_extended and index_extended and middle_closed and ring_closed and pinky_closed and forms_L_shape

def detect_gesto(hand_landmarks):
    """
    Detecta cuál gesto está realizando la mano
    """
    if is_okay_sign(hand_landmarks):
        return "OK"
    elif is_thumbs_up(hand_landmarks):
        return "PULGAR_ARRIBA"
    elif is_victory_sign(hand_landmarks):
        return "VICTORIA"
    elif is_L_sign(hand_landmarks):
        return "L"
    else:
        return "NINGUNO"

# --- FUNCIÓN DE DIBUJADO (MODIFICADA) ---
def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = rgb_image.copy()
    height, width, _ = annotated_image.shape

    # Loop through the detected hands
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        # 1. Draw Keypoints (Dots)
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)

        # 2. Draw Connections (Lines)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
        ]

        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]

            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]

            x1, y1 = int(start_point.x * width), int(start_point.y * height)
            x2, y2 = int(end_point.x * width), int(end_point.y * height)

            cv2.line(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
            
        # 3. Detectar y mostrar gesto para cada mano
        gesto = detect_gesto(hand_landmarks)
        emoji = GESTOS[gesto]
        
        # Mostrar gesto cerca de la muñeca (landmark 0)
        wrist_x = int(hand_landmarks[0].x * width)
        wrist_y = int(hand_landmarks[0].y * height)
        
        cv2.putText(annotated_image, f"Mano {idx+1}: {emoji}", 
                   (wrist_x - 50, wrist_y - 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

    return annotated_image

# --- CALLBACK FUNCTION ---
def save_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global LATEST_RESULT
    LATEST_RESULT = result

# --- INITIALIZE LANDMARKER ---
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=save_result)

# --- PANEL DE INFORMACIÓN ---
def draw_info_panel(frame, gesto_detectado, num_manos):
    """Dibuja un panel informativo en la parte superior"""
    height, width = frame.shape[:2]
    
    # Fondo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 90), (40, 40, 40), -1)
    frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    
    # Título
    cv2.putText(frame, "DETECTOR DE GESTOS - MEDIAPIPE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
    
    # Información de gestos
    cv2.putText(frame, f"Gestos activos: OK, Pulgar Arriba, Victoria, L", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    # Estado actual
    color = (0, 255, 0) if gesto_detectado != "NINGUNO" else (0, 0, 255)
    texto_estado = f"Estado: {GESTOS[gesto_detectado]} {gesto_detectado}" if gesto_detectado != "NINGUNO" else "Estado: Esperando gesto..."
    
    cv2.putText(frame, texto_estado, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Número de manos detectadas
    cv2.putText(frame, f"Manos: {num_manos}", (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    
    return frame

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convertir frame a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Timestamp para modo LIVE_STREAM
        frame_timestamp_ms = int(time.time() * 1000)

        # Enviar a MediaPipe (Async)
        landmarker.detect_async(mp_image, frame_timestamp_ms)

        # Variables para el panel de información
        gesto_actual = "NINGUNO"
        num_manos = 0
        
        # Procesar resultados si existen
        if LATEST_RESULT:
            num_manos = len(LATEST_RESULT.hand_landmarks) if LATEST_RESULT.hand_landmarks else 0
            
            # Dibujar landmarks
            frame = draw_landmarks_on_image(frame, LATEST_RESULT)
            
            # Detectar gestos para cada mano
            if LATEST_RESULT.hand_landmarks:
                for i, hand_landmarks in enumerate(LATEST_RESULT.hand_landmarks):
                    gesto_mano = detect_gesto(hand_landmarks)
                    
                    # Si hay múltiples manos, mostrar el gesto más "activo" (no NINGUNO)
                    if gesto_mano != "NINGUNO":
                        gesto_actual = gesto_mano
        
        # Dibujar panel informativo
        frame = draw_info_panel(frame, gesto_actual, num_manos)
        
        # Mostrar frame
        cv2.imshow('DETECTOR DE GESTOS - MediaPipe', frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
