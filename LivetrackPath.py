import cv2 as cv
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tracker import Tracker

class PeopleCounter:
    def __init__(self, video_path, model_path='yolov8s.pt', max_path_length=50):
        self.model = YOLO(model_path)
        self.cap = cv.VideoCapture(video_path)
        self.tracker = Tracker()
        self.object_paths = {}
        self.object_colors = {}
        self.max_path_length = max_path_length
        self.colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
        self.catergory = 'person' # category to track in the yolo model output
        cv.namedWindow('FRAME')
        cv.setMouseCallback('FRAME', self._coordinates_callback)

        # Coco text
        with open("coco.txt", "r") as my_file:
            data = my_file.read()
        self.class_list = data.split("\n")

    def _coordinates_callback(self, event, x, y, flags, param):
        if event == cv.EVENT_MOUSEMOVE:
            colorsBGR = [x, y]
            print(colorsBGR)

    def _process_frame(self, frame):
        results = self.model(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        
        objects_list = []
        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])

            c = self.class_list[d]
            if self.catergory in c:
                objects_list.append([x1, y1, x2, y2])

        return objects_list

    def _draw_paths(self, frame):
        for obj_id, path in self.object_paths.items():
            color = self.object_colors[obj_id]
            for i in range(1, len(path)):
                cv.line(frame, path[i - 1], path[i], color, 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv.resize(frame, (1020, 500))
            objects_list = self._process_frame(frame)
            
            bboxes_id = self.tracker.update(objects_list)

            new_ids = set()
            for bbox in bboxes_id:
                x1, y1, x2, y2, obj_id = bbox
                center_x = int(x1 + x2) // 2
                center_y = int(y1 + y2) // 2
                cv.circle(frame, (center_x, center_y), 4, (255, 0, 0), -1)

                new_ids.add(obj_id)

                # Store object path
                if obj_id in self.object_paths:
                    self.object_paths[obj_id].append((center_x, center_y))
                else:
                    self.object_paths[obj_id] = [(center_x, center_y)]
                    # Assign color to object
                    self.object_colors[obj_id] = self.colors[len(self.object_paths) % len(self.colors)]

                # Limit path length
                if len(self.object_paths[obj_id]) > self.max_path_length:
                    self.object_paths[obj_id].pop(0)

            # Remove disconnected paths
            for obj_id in list(self.object_paths.keys()):  
                if obj_id not in new_ids:
                    self.object_paths.pop(obj_id, None)
                    self.object_colors.pop(obj_id, None)

            self._draw_paths(frame)

            num_people = len(new_ids)
            cv.putText(frame, f"People: {num_people}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv.imshow('FRAME', frame)
            if cv.waitKey(1) & 0xFF == 27:
                break

        self.cap.release()
        cv.destroyAllWindows()

if __name__ == "__main__":
    people_counter = PeopleCounter('./vidp/vidp.mp4')
    people_counter.run()
