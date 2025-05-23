import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

def detectar_objetos(frame):
    results = model(frame)[0]
    objetos = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        if score > 0.4:
            objetos.append({
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "score": float(score),
                "class_id": int(class_id)
            })
    return objetos
