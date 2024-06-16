import os
import numpy as np
import pytest
from time import time
from sklearn.metrics import confusion_matrix, roc_curve, accuracy_score, precision_score, recall_score, f1_score

snakeai_module = os.environ.get('SNAKEAI_MODULE', 'snake_ai_module')
SnakeAI = getattr(__import__(snakeai_module, fromlist=['SnakeAI']), 'SnakeAI')

@pytest.fixture
def snake_ai():
    state_size = 16  
    action_size = 4  
    return SnakeAI(state_size, action_size)

def generate_random_data():
    return (
        np.random.randint(0, 10),  # food_matches
        np.random.uniform(0, 1),  # nearest_food_distance
        (np.random.uniform(-1, 1), np.random.uniform(-1, 1)),  # nearest_food_vector
        np.random.randint(0, 10),  # small_snakes
        np.random.randint(0, 10),  # nearest_small_snake_length
        np.random.uniform(0, 1),  # nearest_small_snake_distance
        (np.random.uniform(-1, 1), np.random.uniform(-1, 1)),  # nearest_small_snake_vector
        np.random.randint(0, 10),  # large_snakes
        np.random.randint(0, 10),  # nearest_large_snake_length
        np.random.uniform(0, 1),  # nearest_large_snake_distance
        (np.random.uniform(-1, 1), np.random.uniform(-1, 1)),  # nearest_large_snake_vector
        np.random.randint(0, 10),  # killed_snakes
        np.random.uniform(0, 1)  # additional_feature
    )

def test_snake_ai_performance(snake_ai):
    epochs_list = [100, 1000]
    for epochs in epochs_list:
        losses = []
        true_labels = []
        predictions = []

        start_time = time()
        for epoch in range(epochs):
            data = generate_random_data()
            action_probabilities, state = snake_ai.update_state_and_evaluate(*data)
            
            if time() - start_time > 3:
                pytest.fail(f"Epoch time exceeded 3 seconds at epoch {epoch + 1}")
            
            loss = -np.log(action_probabilities[0][data[0] % snake_ai.action_size])  
            
            losses.append(loss)
            true_labels.append(data[0])  
            predictions.append(np.argmax(action_probabilities))

        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=1)
        recall = recall_score(true_labels, predictions, average='weighted')
        f1 = f1_score(true_labels, predictions, average='weighted')
        conf_matrix = confusion_matrix(true_labels, predictions)
        fpr, tpr, thresholds = roc_curve(true_labels, [prob[1] for prob in action_probabilities], pos_label=1)

        print(f"Epochs: {epochs}")
        print(f"Losses: {losses}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"Confusion Matrix: \n{conf_matrix}")
        print(f"ROC Curve: \nFPR: {fpr}, \nTPR: {tpr}, \nThresholds: {thresholds}")

        assert accuracy > 0.5, "Accuracy is below acceptable level"
        assert precision > 0.5, "Precision is below acceptable level"
        assert recall > 0.5, "Recall is below acceptable level"
        assert f1 > 0.5, "F1 Score is below acceptable level"

