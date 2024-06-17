import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
import math

class SnakeAI:
    def __init__(self):
        print("Initializing SnakeAI")
        self.scaler = StandardScaler()
        self.primary_model = self.create_model(16) 
        self.secondary_model = self.create_model(16)  
        self.train_models()

    def create_model(self, input_dim):
        print(f"Creating model with input_dim: {input_dim}")
        model = Sequential()
        model.add(Dense(128, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        optimizer = Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='mse')
        return model

    def generate_training_data(self):
        print("Generating training data")
        X = np.random.rand(1000, 15) * 100  # 15 

        for i in range(1000):
            X[i][2] = np.random.rand() * 2 - 1  # nearest_food_vector_x
            X[i][3] = np.random.rand() * 2 - 1  # nearest_food_vector_y
            X[i][7] = np.random.rand() * 2 - 1  # nearest_small_snake_vector_x
            X[i][8] = np.random.rand() * 2 - 1  # nearest_small_snake_vector_y
            X[i][12] = np.random.rand() * 2 - 1  # nearest_large_snake_vector_x
            X[i][13] = np.random.rand() * 2 - 1  # nearest_large_snake_vector_y

        killed_snakes = np.random.randint(2, size=(1000, 1)) 
        X = np.hstack((X, killed_snakes))  

        y = np.random.rand(1000)  
        return X, y

    def train_models(self):
        print("Training models")
        X, y = self.generate_training_data()
        print(f"Before scaling: X.shape = {X.shape}")
        X = self.scaler.fit_transform(X)
        print(f"After scaling: X.shape = {X.shape}, y.shape = {y.shape}")
        self.primary_model.fit(X, y, epochs=100, batch_size=32)
        self.secondary_model.fit(X, y, epochs=100, batch_size=32)

    def state_to_model_input(self, state):
        model_input = [
            state[0] if isinstance(state[0], (int, float)) else 0,
            min(state[1], 1e6) if isinstance(state[1], (int, float)) else 1e6,
            state[2] if isinstance(state[2], (int, float)) else 0,  # nearest_food_vector_x
            state[3] if isinstance(state[3], (int, float)) else 0,  # nearest_food_vector_y
            state[4] if isinstance(state[4], (int, float)) else 0,
            state[5] if isinstance(state[5], (int, float)) else 0,
            min(state[6], 1e6) if isinstance(state[6], (int, float)) else 1e6,
            state[7] if isinstance(state[7], (int, float)) else 0,  # nearest_small_snake_vector_x
            state[8] if isinstance(state[8], (int, float)) else 0,  # nearest_small_snake_vector_y
            state[9] if isinstance(state[9], (int, float)) else 0,
            state[10] if isinstance(state[10], (int, float)) else 0,
            min(state[11], 1e6) if isinstance(state[11], (int, float)) else 1e6,
            state[12] if isinstance(state[12], (int, float)) else 0,  # nearest_large_snake_vector_x
            state[13] if isinstance(state[13], (int, float)) else 0,  # nearest_large_snake_vector_y
            1 if state[14] else 0,  # killed_snakes as 1 (True) or 0 (False)
            1 if state[15] else 0  # Additional feature
        ]
        return model_input

    def update_state_and_evaluate(self, food_matches, nearest_food_distance, nearest_food_vector, small_snakes, nearest_small_snake_length, nearest_small_snake_distance, nearest_small_snake_vector, large_snakes, nearest_large_snake_length, nearest_large_snake_distance, nearest_large_snake_vector, killed_snakes, additional_feature):
        print("Updating state and evaluating")
        current_state = [
            food_matches if food_matches is not None else 0,
            nearest_food_distance if nearest_food_distance is not None else float('inf'),
            nearest_food_vector[0] if nearest_food_vector is not None else 0,  # nearest_food_vector_x
            nearest_food_vector[1] if nearest_food_vector is not None else 0,  # nearest_food_vector_y
            small_snakes if small_snakes is not None else 0,
            nearest_small_snake_length if nearest_small_snake_length is not None else 0,
            nearest_small_snake_distance if nearest_small_snake_distance is not None else float('inf'),
            nearest_small_snake_vector[0] if nearest_small_snake_vector is not None else 0,  # nearest_small_snake_vector_x
            nearest_small_snake_vector[1] if nearest_small_snake_vector is not None else 0,  # nearest_small_snake_vector_y
            large_snakes if large_snakes is not None else 0,
            nearest_large_snake_length if nearest_large_snake_length is not None else 0,
            nearest_large_snake_distance if nearest_large_snake_distance is not None else float('inf'),
            nearest_large_snake_vector[0] if nearest_large_snake_vector is not None else 0,  # nearest_large_snake_vector_x
            nearest_large_snake_vector[1] if nearest_large_snake_vector is not None else 0,  # nearest_large_snake_vector_y
            1 if killed_snakes else 0,  
            1 if additional_feature else 0  
        ]

        print(f"Updated state: {current_state}")

        model_input = self.state_to_model_input(current_state)
        model_input = np.array(model_input).reshape(1, -1)
        print(f"Model input shape: {model_input.shape}")
        model_input = self.scaler.transform(model_input)  

        return model_input, current_state 

    def evaluate_new_state(self, model, new_state):
        evaluation = model.predict(new_state)
        return evaluation[0][0]

    def evaluate_possible_moves(self, current_state):
        current_state = current_state.reshape(-1).tolist()  
        possible_moves = []
        for angle in range(0, 360):
            possible_move = current_state.copy()
            possible_move[2] = math.cos(math.radians(angle))  # nearest_food_vector_x
            possible_move[3] = math.sin(math.radians(angle))  # nearest_food_vector_y
            possible_moves.append(self.state_to_model_input(possible_move))
        possible_moves = np.array(possible_moves)
        possible_moves = self.scaler.transform(possible_moves)  
        move_evaluations = self.primary_model.predict(possible_moves)
        move_evaluations += np.random.normal(0, 0.01, size=move_evaluations.shape)
       
        best_move_index = np.argmax(move_evaluations)
        best_move = possible_moves[best_move_index]
        return best_move, move_evaluations

    def boost(self, current_state):
        _, move_evaluations = self.evaluate_possible_moves(current_state)
        closest_enemy_distance = min(
            current_state[5] if current_state[5] != float('inf') else float('inf'),
            current_state[9] if current_state[9] != float('inf') else float('inf')
        )
        boost_needed = closest_enemy_distance < 50
        return boost_needed

    def angle_list(self, current_state):
        _, move_evaluations = self.evaluate_possible_moves(current_state)
        angle_list = [(angle, move_evaluations[angle]) for angle in range(360)]
        return angle_list

    def angle_list_max(self, current_state):
        _, move_evaluations = self.evaluate_possible_moves(current_state)
        max_move_index = np.argmax(move_evaluations)
        return max_move_index

    def angle_list_min(self, current_state):
        _, move_evaluations = self.evaluate_possible_moves(current_state)
        min_move_index = np.argmin(move_evaluations)
        return min_move_index





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
   
    # Create a mask for white areas in the image
    white_mask = ((img[:, :, 0] > 200) & (img[:, :, 1] > 200) & (img[:, :, 2] > 200)).astype(np.uint8) * 255
   
    # Perform bitwise_and with uint8 mask
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
    ai = SnakeAI()
    current_state = ai.update_state_and_evaluate(
        len(food_matches), nearest_food_distance, nearest_food_vector,
        len(small_snakes), nearest_small_snake_length,
        nearest_small_snake_distance, nearest_small_snake_vector,
        len(large_snakes), nearest_large_snake_length, nearest_large_snake_distance, nearest_large_snake_vector, killed_snakes, additional_feature
    )

    snake_ai = SnakeAI()
    angle_evaluations = snake_ai.angle_list(current_state)
    for angle, evaluation in angle_evaluations:
        print(f"角度: {angle}°, 評価値: {evaluation}")
     
if __name__ == "__main__":
    while True:
        main()
