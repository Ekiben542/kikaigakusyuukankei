def get_game_screen_data():
    img = np.asarray(ImageGrab.grab(rect))
    #img = img.flatten()
    return img
