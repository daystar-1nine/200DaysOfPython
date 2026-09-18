
import math

class CentroidTracker:
    def __init__(self, max_disappeared=5):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = []
        for (startX, startY, endX, endY) in rects:
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids.append((cX, cY))

        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Simple matching (Mocking full Hungarian for baseline)
            used_rows = set()
            used_cols = set()
            
            for i, obj_c in enumerate(object_centroids):
                best_dist = 9999
                best_j = -1
                for j, inp_c in enumerate(input_centroids):
                    if j in used_cols: continue
                    d = math.dist(obj_c, inp_c)
                    if d < best_dist:
                        best_dist = d
                        best_j = j
                
                if best_j != -1 and best_dist < 50:
                    object_id = object_ids[i]
                    self.objects[object_id] = input_centroids[best_j]
                    self.disappeared[object_id] = 0
                    used_rows.add(i)
                    used_cols.add(best_j)
            
            # Unused detections
            for j in range(0, len(input_centroids)):
                if j not in used_cols:
                    self.register(input_centroids[j])
                    
            # Missing tracks
            for i in range(0, len(object_centroids)):
                if i not in used_rows:
                    object_id = object_ids[i]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)

        return self.objects
