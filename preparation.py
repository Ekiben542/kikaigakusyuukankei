def preparation():
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
