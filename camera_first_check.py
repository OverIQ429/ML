from ultralytics import YOLO
import cv2
import numpy as np
import random

model = YOLO('yolov8n.pt')

# Словарь соответствия имен классов и цветов
colors = {}


def get_random_color():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# История треков (словарь {id объекта: [(x, y), (x, y), ...]})
track_history = {}
track_id_counter = 0

capture = cv2.VideoCapture(2)
fps = capture.get(cv2.CAP_PROP_FPS)
def predict_position(track, future_time, fps):
    if len(track) < 2:
        return track[-1]

    N = min(len(track), 25)
    track = np.array(track[-N:])

    times = np.arange(-N + 1, 1)

    A = np.vstack([times, np.ones(len(times))]).T
    k_x, b_x = np.linalg.lstsq(A, track[:, 0], rcond=None)[0]
    k_y, b_y = np.linalg.lstsq(A, track[:, 1], rcond=None)[0]

    future_frames = future_time * fps
    future_x = k_x * future_frames + b_x
    future_y = k_y * future_frames + b_y

    return future_x, future_y

while True:
    ret, frame = capture.read()
    results = model(frame)[0]

    detections = []
    for class_id, box in zip(results.boxes.cls.cpu().numpy(), results.boxes.xyxy.cpu().numpy().astype(np.int32)):
        class_name = results.names[int(class_id)]
        x1, y1, x2, y2 = box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        detections.append((cx, cy, class_name))

    # Сопоставление детектированных объектов с существующими треками
    new_tracks = []
    matched_detections = []
    for cx, cy, class_name in detections:
        matched = False
        for track_id, track in track_history.items():
            if len(track) > 0:  # Проверка, есть ли уже координаты в треке
                last_cx, last_cy = track[-1]
                distance = np.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
                if distance < 50:  # порог расстояния для сопоставления
                    track.append((cx, cy))
                    matched = True
                    matched_detections.append((cx, cy, class_name, track_id))
                    break
        if not matched:
            track_id_counter += 1
            track_history[track_id_counter] = [(cx, cy)]
            new_tracks.append((cx, cy, class_name, track_id_counter))

    # Отрисовка
    for cx, cy, class_name, track_id in matched_detections + new_tracks:

        if class_name not in colors:
            colors[class_name] = get_random_color()
        current_color = colors[class_name]

        # Получаем bbox, только если detections для данного класса были
        class_indices = np.where(results.boxes.cls.cpu().numpy() == class_id)[0]
        if len(class_indices) > 0:
            x1, y1, x2, y2 = results.boxes.xyxy[class_indices[0]]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)

        track = track_history[track_id]
        points = np.array(track).reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(frame, [points], isClosed=False, color=current_color, thickness=2)


        future_time = 1.5  # секунд
        future_x, future_y = predict_position(track, future_time, fps)

        if len(track) > 1:
            last_x, last_y = track[-1]
            cv2.line(frame, (int(last_x), int(last_y)), (int(future_x), int(future_y)), (0, 255, 255), 2)

        cv2.circle(frame, (int(future_x), int(future_y)), 5, (0, 255, 0), -1)
        cv2.putText(frame, 'Predicted', (int(future_x), int(future_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)

    cv2.imshow('YOLOv8 Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()