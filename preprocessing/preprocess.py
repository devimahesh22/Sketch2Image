import cv2
import numpy as np

def resize_image(img_array):
    resized_image = cv2.resize(img_array, (600,600))
    return resized_image

def greyscale_image(img_array):
    greyscaled_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    return greyscaled_image

def convert_to_hsv(img_array):
    hsv_image = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
    return hsv_image

def extract_green_mask(hsv_image):
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])

    green_mask = cv2.inRange(
        hsv_image,
        lower_green,
        upper_green
    )
    return green_mask


def remove_noise(img_array):
    remove_noise = cv2.medianBlur(img_array, 5)
    return remove_noise

def threshold_image(img_array):
    threshold_image = cv2.adaptiveThreshold(img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return threshold_image

def detect_edges(img_array, lower_threshold, higher_threshold):
    detect_edges = cv2.Canny(img_array, lower_threshold, higher_threshold, apertureSize= 3, L2gradient=False)
    return detect_edges

def morphological_operation(img_array):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    # morphological_operation_open = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
    morphological_operation_close = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
    return morphological_operation_close

def draw_contours(img_array):
    output = img_array.copy()
    contours, hierarchy = cv2.findContours(output, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    contour_image= cv2.drawContours(output, contours, -1, (255,0,0), 2)
    return contour_image, contour_count



