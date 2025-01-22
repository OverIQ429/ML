from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import threading
model = YOLO('yolov8n.pt')

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

color = (0, 255, 255)
video_path = "input.mp4"
capture = cv2.VideoCapture(0) #VideoCapture - это камера
fps = capture.get(cv2.CAP_PROP_FPS)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

track_history = defaultdict(lambda: [])

output_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для записи
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while True:
    ret, frame = capture.read()
    results = model.track(frame, tracker="bytetrack.yaml", persist=True)[0]
    if results.boxes is not None and results.boxes.id is not None:
        for class_id, box, track_id in zip(results.boxes.cls.cpu().numpy(),
                                 results.boxes.xyxy.cpu().numpy().astype(np.int32),
                                 results.boxes.id.int().cpu().tolist()):
            class_name = results.names[int(class_id)]
            x1, y1, x2, y2 = box
            track = track_history[track_id]

            track.append((float((x1 + x2) / 2), float((y1 + y2) / 2)))  # добавление координат центра объекта в историю
            if len(track) > 30:  # ограничение длины истории до 30 кадров
                track.pop(0)

            # Рисование линий трека
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            future_time = 1.5  # секунд
            future_x, future_y = predict_position(track, future_time, fps)
            if len(track) > 1:
                last_x, last_y = track[-1]
                cv2.line(frame, (int(last_x), int(last_y)), (int(future_x), int(future_y)), (0, 255, 255), 2)

            cv2.circle(frame, (int(future_x), int(future_y)), 5, (0, 255, 0), -1)
            cv2.putText(frame,
                        class_name,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        color, 2)
            cv2.putText(frame, 'Predicted', (int(future_x), int(future_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 0), 2)
    cv2.imshow('Camera', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()