def is_snake_killed(contour, img):
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
   
    # Create a mask for white areas in the image
    white_mask = ((img[:, :, 0] > 200) & (img[:, :, 1] > 200) & (img[:, :, 2] > 200)).astype(np.uint8) * 255
   
    # Perform bitwise_and with uint8 mask
    white_area = cv2.countNonZero(cv2.bitwise_and(mask, mask, mask=white_mask))
   
    return white_area > 0
