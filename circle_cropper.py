import numpy as np
import math
import random
import cv2
from matplotlib import pyplot as plt
from pip import main
import os
from PIL import Image, ImageDraw


# helper function to get the path string for the image based on image number
def image_path_generator(main_path_string: str, image_number: int) -> str:

    image_number = str(image_number)
    padding_size = 3 - len(image_number)
    padding = ""

    for _ in range(padding_size):
        padding += "0"

    image_number = padding + image_number

    return main_path_string + image_number + ".jpeg"




# helper to find the midpoint of the ball given the image
def find_ball_location(image_path):
# main path string

    # load in the image
    read_image = cv2.imread(image_path, 1)

    # add in gaussian blur so that the inside things on the squash ball are not identified as circles
    blurred_read_image = cv2.GaussianBlur(read_image, (21, 21), cv2.BORDER_DEFAULT)
    blurred_read_image_copy = blurred_read_image.copy()
    blurred_read_image_copy = cv2.cvtColor(blurred_read_image_copy, cv2.COLOR_BGR2GRAY)

    all_circles = cv2.HoughCircles(blurred_read_image_copy, cv2.HOUGH_GRADIENT, 1, 20, param1=60, param2=40, minRadius=35, maxRadius=0)

    if all_circles is not None:

        all_circles = np.uint16(np.around(all_circles))
        num_circles = all_circles.shape[1]

        if num_circles > 1:
            biggest_rad = all_circles[0][0][2]
            biggest_num = 0

            for num in range(num_circles):

                if all_circles[0][num][2] > biggest_rad:
                    biggest_rad = all_circles[0][num][2]
                    biggest_num = num

            main_circle = all_circles[0][biggest_num]
        else:
            main_circle = all_circles

        main_circle = np.squeeze(main_circle)

        # cv2.circle(read_image,(main_circle[0],main_circle[1]),main_circle[2],(0,255,0),2)
        # # draw the center of the circle
        # cv2.circle(read_image,(main_circle[0],main_circle[1]),2,(0,0,255),3)
        # cv2.imshow('detected circles',read_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return main_circle

    else:

        # cv2.imshow('detected circles',read_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return False



def create_circle_cutout(main_path_string, main_circle):

    image = cv2.imread('test_imgs/test_imgs.001.jpeg')
    mask = np.zeros(image.shape, dtype=np.uint8)
    x,y = main_circle[0], main_circle[1]
    cv2.circle(mask, (x,y), main_circle[2], (255,255,255), -1)

    # Bitwise-and for ROI
    ROI = cv2.bitwise_and(image, mask)

    # Crop mask and turn background white
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    x,y,w,h = cv2.boundingRect(mask)
    result = ROI[y:y+h,x:x+w]
    mask = mask[y:y+h,x:x+w]
    result[mask==0] = (255,255,255)

    cv2.imshow('result', result)
    cv2.waitKey()




def compare_cutouts(old_cut, new_cut):
    pass










folder_path = "test_imgs"
all_images_paths = []

num_images = 0
for path in os.listdir(folder_path):
    relative_path = os.path.join(folder_path, path)
    if os.path.isfile(relative_path):
        num_images += 1
        all_images_paths.append(relative_path)


ball_presence = []
x_locations = []
y_locations = []
rotation_amounts = []

old_cutout = 0
new_cutout = 0


for i in range(num_images):

    main_circle = find_ball_location(all_images_paths[i])
    new_cutout = create_circle_cutout(all_images_paths[i], main_circle)

    if main_circle is not False:
        print("ball at {}".format(i))

    else:
        print("ball not at {}".format(i))

    if i != 0:
        compare_cutouts(old_cutout, new_cutout)


    




# Create mask and draw circle onto mask



