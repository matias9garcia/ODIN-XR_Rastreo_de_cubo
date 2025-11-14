import cv2
import cv2.aruco as aruco

# Inicializar cámara (0 para la cámara por defecto)
cap = cv2.VideoCapture(0)

# Obtener diccionario ArUco (4x4 bits, 50 marcadores posibles)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()

ID_OBJETIVO = 3  # ID del marcador que queremos rastrear

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar marcadores en la imagen
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # Recorrer todos los marcadores detectados
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == ID_OBJETIVO:
                # Dibujar contorno del marcador
                aruco.drawDetectedMarkers(frame, [corners[i]])
                
                # Calcular centro del marcador
                c = corners[i][0]
                cx = int(c[:, 0].mean())
                cy = int(c[:, 1].mean())

                # Mostrar ID y centro
                cv2.putText(frame, f"ID {marker_id}", (cx - 20, cy - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow("Rastreo ArUco ID 0", frame)

    # Presiona ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
