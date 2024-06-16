from telnetlib import GA
from webbrowser import get
import numpy as np
import cv2
from PIL import ImageGrab
import pyautogui
import pygetwindow as gw
import time
import random
from collections import deque
import webbrowser
import ctypes
import win32gui
import win32con
import keyboard
from DQN import Qlearning
import math
import ast

rect=0,0,1920,1080 #”CˆÓ
def get_game_screen_data():
    img = np.asarray(ImageGrab.grab(rect))
    #img = img.flatten()
    return img

class IsKilled:
    def __init__(self):
        self.flag = False

    def point(self, screen):
        hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
        lower_color = np.array([0, 100, 100])
        upper_color = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 5000:  
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if aspect_ratio < 0.5 or aspect_ratio > 2.0:  
                    M = cv2.moments(contour)
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    self.flag = True
                    return self.flag, (cX, cY)
        self.flag = False
        return self.flag, None

    def hantei(self, screen):
        if self.flag:
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            lower_color = np.array([0, 100, 100])
            upper_color = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_color, upper_color)
            if np.sum(mask) < 1000:
                self.flag = False
                return True
        self.flag = False
        return False

def is_gameover(screen):
    gray_screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('play_again_button.png', 0)
    if template is None:
        raise FileNotFoundError("Template image 'play_again_button.png' not found.")
    res = cv2.matchTemplate(gray_screen, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    if len(loc[0]) > 0:
        return True
    return False

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


url="https://slither.io"
webbrowser.open(url)
time.sleep(6)

try:
    handle = ctypes.windll.user32.FindWindowW(0,"slither.io - Google Chrome")
    rect = 0,0,1920,1080
    win32gui.SetWindowPos(handle, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
except IndexError as e:
    print("Index error")

pyautogui.click(1883, 139)
pyautogui.click(1891, 75)
pyautogui.click(1883, y=565)
DQN=Qlearning(0.5,0.8,1728,2,[1.0,1.0,1.0,1.0,1.0])
kill_hantei=IsKilled()
end=False
with open('weight.txt', 'r') as file:
    weight_content = file.read().strip() 
if weight_content:
    data_list = ast.literal_eval(weight_content)
    numpy_arrays = [np.array(matrix) for matrix in data_list]
    DQN.change_t_q_network_weight(numpy_arrays)
with open('Kill_or_Death.txt', 'r') as file:
    weight_content = file.read().strip() 
if weight_content:
    data_list = ast.literal_eval(weight_content)
    kill=data_list[0]
    death=data_list[1]
else:
    kill=0
    death=0
pyautogui.click(897, 484)
pyautogui.write('a')
while True:
    pyautogui.click(958, 555)
    screen=get_game_screen_data()
    screen_changed=cv2.resize(screen,(32,18)).flatten().flatten().flatten()/1000000
    while True:
        kill_hantei.point(screen)
        boosted=move(DQN.act(screen_changed))
        screen=get_game_screen_data()
        screen_changed=cv2.resize(screen,(32,18)).flatten().flatten().flatten()/1000000
        if kill_hantei.hantei(screen):
            kill+=1
            DQN.note(screen_changed,10.0,False,10000)
        elif is_gameover(screen):
            death+=1
            DQN.note(screen_changed,-10.0,True,10000)
            break
        else:
            DQN.note(screen_changed,0.0,False,10000)
        DQN.learn(100,-1)
        if keyboard.is_pressed("e"):
            end=True
            break
    if end:
        break
f=open("weight.txt","w+")
tmp=[]
for i in DQN.show_t_network_weight():
    tmp.append(i.tolist())
f.write(str(tmp))
f.close()

f=open("Kill_or_Death.txt","w+")
f.write(str([kill,death]))
f.close()

win32gui.SetWindowPos(handle, win32con.HWND_NOTOPMOST, 0,0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
