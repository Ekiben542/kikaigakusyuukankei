import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from scipy.special import expit  # Sigmoid 

class SnakeAI:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self.build_model()
        self.hidden_state = np.zeros((1, 128))  
    
    def build_model(self):
        model = Sequential()
        model.add(LSTM(128, input_shape=(1, self.state_size), return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(self.action_size, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
    
    def update_state_and_evaluate(self, food_matches, nearest_food_distance, nearest_food_vector, small_snakes, nearest_small_snake_length, nearest_small_snake_distance, nearest_small_snake_vector, large_snakes, nearest_large_snake_length, nearest_large_snake_distance, nearest_large_snake_vector, killed_snakes, additional_feature):
        nearest_food_vector = nearest_food_vector if nearest_food_vector is not None else (0, 0)
        nearest_small_snake_vector = nearest_small_snake_vector if nearest_small_snake_vector is not None else (0, 0)
        nearest_large_snake_vector = nearest_large_snake_vector if nearest_large_snake_vector is not None else (0, 0)

        state = np.array([
            food_matches, nearest_food_distance or 0, nearest_food_vector[0], nearest_food_vector[1], 
            small_snakes, nearest_small_snake_length or 0, nearest_small_snake_distance or 0, nearest_small_snake_vector[0], nearest_small_snake_vector[1], 
            large_snakes, nearest_large_snake_length or 0, nearest_large_snake_distance or 0, nearest_large_snake_vector[0], nearest_large_snake_vector[1], 
            killed_snakes or 0, additional_feature or 0
        ])
        state = np.reshape(state, (1, 1, self.state_size))
        importance = self.calculate_importance(state)
        self.hidden_state = self.update_rnn_state(self.hidden_state, importance)
        
        action_probabilities = self.model.predict(state)
        return action_probabilities
    
    def calculate_importance(self, state):
        importance = 1.0 / (state[0, 0, 6] + 1)  
        return importance
    
    def update_rnn_state(self, hidden_state, importance):
        if importance > 0.5:
            hidden_state = np.zeros_like(hidden_state)
        return hidden_state
    

    def angle(self, vector):
        angle = np.arctan2(vector[1], vector[0])
        return angle
    
    def decide_boost(self, distance):
        return expit(distance)


import numpy as np
from PIL import ImageGrab, Image
import cv2
import matplotlib.pyplot as plt

rect = (0, 0, 1920, 1080)

def get_game_screen_data():
    img = np.asarray(ImageGrab.grab(bbox=rect))
    return img

def preprocess_image(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_food1 = np.array([0, 150, 100])
    upper_food1 = np.array([10, 255, 255])
    lower_food2 = np.array([170, 150, 100])
    upper_food2 = np.array([180, 255, 255])
   
    mask_food1 = cv2.inRange(hsv, lower_food1, upper_food1)
    mask_food2 = cv2.inRange(hsv, lower_food2, upper_food2)
    mask_food = cv2.bitwise_or(mask_food1, mask_food2)
    lower_snake = np.array([100, 150, 0])
    upper_snake = np.array([140, 255, 255])
    mask_snake = cv2.inRange(hsv, lower_snake, upper_snake)
   
    kernel = np.ones((3, 3), np.uint8)
    mask_food = cv2.morphologyEx(mask_food, cv2.MORPH_OPEN, kernel)
    mask_food = cv2.morphologyEx(mask_food, cv2.MORPH_CLOSE, kernel)
    mask_snake = cv2.morphologyEx(mask_snake, cv2.MORPH_OPEN, kernel)
    mask_snake = cv2.morphologyEx(mask_snake, cv2.MORPH_CLOSE, kernel)
   
    return mask_food, mask_snake

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

def generate_rotated_templates(template, angles=[0, 45, 90, 135, 180, 225, 270, 315]):
    rotated_templates = []
    for angle in angles:
        rotated_template = rotate_image(template, angle)
        rotated_templates.append(rotated_template)
    return rotated_templates

def template_matching(img, templates, threshold=0.8):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    matches = []
    for template in templates:
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        result = cv2.matchTemplate(gray_img, gray_template, cv2.TM_CCOEFF_NORMED)
        locations = np.where(result >= threshold)
        w, h = gray_template.shape[::-1]
        for pt in zip(*locations[::-1]):
            matches.append((pt[0], pt[1], w, h))
    return matches

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []
   
    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = x1 + boxes[:, 2]
    y2 = y1 + boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(last):
            j = idxs[pos]
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            overlap = float(w * h) / area[j]

            if overlap > overlapThresh:
                suppress.append(pos)

        idxs = np.delete(idxs, suppress)

    return boxes[pick].astype("int")

def detect_long_objects(mask_snake, min_length=100):
    contours, _ = cv2.findContours(mask_snake, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    long_objects = []
    for contour in contours:
        if cv2.arcLength(contour, True) > min_length:
            long_objects.append(contour)
    return long_objects

def draw_matches(img, matches, color, is_template=False):
    if is_template:
        for (x, y, w, h) in matches:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    else:
        for contour in matches:
            cv2.drawContours(img, [contour], -1, color, 2)
    return img

def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

def is_snake_killed(contour, img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
   
    white_mask = ((img[:, :, 0] > 200) & (img[:, :, 1] > 200) & (img[:, :, 2] > 200)).astype(np.uint8) * 255
   
    white_area = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=white_mask))
   
    return white_area > 0

def main():
    img = get_game_screen_data()
    mask_food, mask_snake = preprocess_image(img)
    food_template_path = 'food_template.png'
    snake_template_path = 'snake_template.png'
   
    food_template = cv2.imread(food_template_path)
    snake_template = cv2.imread(snake_template_path)
   
    if food_template is None:
        raise FileNotFoundError(f"Food template not found: {food_template_path}")
    if snake_template is None:
        raise FileNotFoundError(f"Snake template not found: {snake_template_path}")
    rotated_templates = generate_rotated_templates(snake_template)
    food_matches = template_matching(img, [food_template])
    snake_matches = template_matching(img, rotated_templates)
   
    food_matches = non_max_suppression(food_matches, overlapThresh=0.3)
    snake_matches = non_max_suppression(snake_matches, overlapThresh=0.3)
    long_snake_matches = detect_long_objects(mask_snake)
    screen_center = (img.shape[1] // 2, img.shape[0] // 2)
   
    self_snake_radius = 50
   
    food_distances = [calculate_distance(screen_center, (x + w // 2, y + h // 2)) for (x, y, w, h) in food_matches]
    nearest_food_distance = min(food_distances) if food_distances else float('inf')
    nearest_food_vector = None
    if food_distances:
        nearest_food_idx = food_distances.index(nearest_food_distance)
        nearest_food_x, nearest_food_y, nearest_food_w, nearest_food_h = food_matches[nearest_food_idx]
        nearest_food_vector = (nearest_food_x + nearest_food_w // 2 - screen_center[0], nearest_food_y + nearest_food_h // 2 - screen_center[1])
   
    small_snakes = [match for match in snake_matches if calculate_distance(screen_center, (match[0] + match[2] // 2, match[1] + match[3] // 2)) > self_snake_radius and calculate_distance(screen_center, (match[0] + match[2] // 2, match[1] + match[3] // 2)) <= 200]
    large_snakes = [contour for contour in long_snake_matches if calculate_distance(screen_center, tuple(contour[0][0])) > self_snake_radius and calculate_distance(screen_center, tuple(contour[0][0])) <= 200]
   
    small_snake_distances = [calculate_distance(screen_center, (x + w // 2, y + h // 2)) for (x, y, w, h) in small_snakes]
    nearest_small_snake_distance = min(small_snake_distances) if small_snake_distances else float('inf')
    nearest_small_snake_vector = None
    nearest_small_snake_length = None
    if small_snake_distances:
        nearest_small_snake_idx = small_snake_distances.index(nearest_small_snake_distance)
        nearest_small_snake_x, nearest_small_snake_y, nearest_small_snake_w, nearest_small_snake_h = small_snakes[nearest_small_snake_idx]
        nearest_small_snake_vector = (nearest_small_snake_x + nearest_small_snake_w // 2 - screen_center[0], nearest_small_snake_y + nearest_small_snake_h // 2 - screen_center[1])
        nearest_small_snake_length = nearest_small_snake_w + nearest_small_snake_h

    large_snake_distances = [calculate_distance(screen_center, tuple(contour[0][0])) for contour in large_snakes]
    nearest_large_snake_distance = min(large_snake_distances) if large_snake_distances else float('inf')
    nearest_large_snake_vector = None
    nearest_large_snake_length = None
    if large_snake_distances:
        nearest_large_snake_idx = large_snake_distances.index(nearest_large_snake_distance)
        nearest_large_snake_contour = large_snakes[nearest_large_snake_idx]
        nearest_large_snake_vector = (nearest_large_snake_contour[0][0][0] - screen_center[0], nearest_large_snake_contour[0][0][1] - screen_center[1])
        nearest_large_snake_length = cv2.arcLength(nearest_large_snake_contour, True)
    additional_feature = 0 
    killed_snakes = any(is_snake_killed(contour, img) for contour in long_snake_matches)
    print("周囲の餌の個数:", len(food_matches))
    print("最も近い餌との距離:", nearest_food_distance)
    print("最も近い餌の方向ベクトル (dx, dy):", nearest_food_vector)
    print("近くにいる小さい蛇の個数:", len(small_snakes))
    print("近くにいる大きな蛇の個数:", len(large_snakes))
    print("最も近い敵蛇の長さ (小さい蛇):", nearest_small_snake_length)
    print("最も近い敵蛇の距離 (小さい蛇):", nearest_small_snake_distance)
    print("最も近い敵蛇の方向ベクトル (小さい蛇) (dx, dy):", nearest_small_snake_vector)
    print("最も近い敵蛇の長さ (大きな蛇):", nearest_large_snake_length)
    print("最も近い敵蛇の距離 (大きな蛇):", nearest_large_snake_distance)
    print("最も近い敵蛇の方向ベクトル (大きな蛇) (dx, dy):", nearest_large_snake_vector)
    print(killed_snakes)
    ai = SnakeAI(state_size=16, action_size=4)

    current_state = ai.update_state_and_evaluate(
        len(food_matches), nearest_food_distance, nearest_food_vector,
        len(small_snakes), nearest_small_snake_length,
        nearest_small_snake_distance, nearest_small_snake_vector,
        len(large_snakes), nearest_large_snake_length, nearest_large_snake_distance, nearest_large_snake_vector, killed_snakes, additional_feature
    )

    action_probabilities = current_state[0]  # Corrected to get the action probabilities from the 2D array

    chosen_action = np.argmax(action_probabilities)

    if nearest_food_vector is not None:
        angle_to_food = ai.angle(nearest_food_vector)
        boost = ai.decide_boost(nearest_food_distance)
        print("向くべき最適な角度:", angle_to_food)
        print("ブーストするか？", boost) 

if __name__ == "__main__":
    while True:
        main()
