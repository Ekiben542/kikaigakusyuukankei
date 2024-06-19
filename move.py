def move(action):
    angle = action[0]*8000000
    boost = action[1]*-100000
    radians = math.radians(angle)
    x_direction = math.cos(radians)
    y_direction = math.sin(radians)
    x = 960 + 200 * x_direction
    y = 540 + 200 * y_direction
    pyautogui.moveTo(x, y)
    if boost >= 0.5:
            pyautogui.mouseDown()
    else:
            pyautogui.mouseUp()
