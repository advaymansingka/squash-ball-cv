import numpy as np
import math
import random
import cv2
from matplotlib import pyplot as plt
from pip import main
import os
import imutils


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

        cv2.circle(read_image,(main_circle[0],main_circle[1]),main_circle[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(read_image,(main_circle[0],main_circle[1]),2,(0,0,255),3)
        cv2.imshow('detected circles',read_image)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return main_circle

    else:

        # cv2.imshow('detected circles',read_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return False



def create_circle_cutout(main_path_string, main_circle):

    image = cv2.imread(main_path_string)
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

    return result




def convolve_cutouts(old_cutout, new_cutout):

    rows = old_cutout.shape[0]
    cols = old_cutout.shape[1]

    img_center = (cols / 2, rows / 2)
    M = cv2.getRotationMatrix2D(img_center, 5, 1)
    old_cut_plus_five = cv2.warpAffine(old_cutout, M, (cols, rows), borderValue=255)

    min_found = False
    minima_index = 0
    i = 0

    convolution_returns = []

    while not min_found:

        rows = old_cut_plus_five.shape[0]
        cols = old_cut_plus_five.shape[1]

        img_center = (cols / 2, rows / 2)
        M = cv2.getRotationMatrix2D(img_center, -0.05 * i, 1)

        rotated_image = cv2.warpAffine(old_cut_plus_five, M, (cols, rows), borderValue=255)

        new_grey_cutout = cv2.cvtColor(new_cutout, cv2.COLOR_BGR2GRAY)
        rotated_image = cv2.cvtColor(rotated_image, cv2.COLOR_BGR2GRAY)

        width_diff = rotated_image.shape[1] - new_grey_cutout.shape[1]
        left = abs(width_diff) // 2
        right = abs(width_diff) - left

        if width_diff == 0:
            pass

        elif width_diff > 0:
            new_grey_cutout = cv2.copyMakeBorder(new_grey_cutout, 0, 0, left, right, cv2.BORDER_CONSTANT, value = 255)

        elif width_diff < 0:
            rotated_image = cv2.copyMakeBorder(rotated_image, 0, 0, left, right, cv2.BORDER_CONSTANT, value = 255)


        height_diff = rotated_image.shape[0] - new_grey_cutout.shape[0]
        top = abs(height_diff) // 2
        bottom = abs(height_diff) - top

        if height_diff == 0:
            pass

        elif height_diff > 0:
            new_grey_cutout = cv2.copyMakeBorder(new_grey_cutout, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value = 255)

        elif height_diff < 0:
            rotated_image = cv2.copyMakeBorder(rotated_image, top, bottom, 0, 0, cv2.BORDER_CONSTANT, value = 255)

        convolved = cv2.subtract(rotated_image, new_grey_cutout)

        convolution_ret = cv2.sumElems(convolved)[0]
        convolution_returns.append(convolution_ret)

        if i > 8:
            potential_minima = convolution_returns[-4]
            behind = (convolution_returns[-7] + convolution_returns[-6] + convolution_returns[-5]) // 3
            ahead = (convolution_returns[-3] + convolution_returns[-2] + convolution_returns[-1]) // 3

            if potential_minima < behind and potential_minima < ahead:
                min_found = True
                minima_index = i

        i += 1

    if not minima_index:
        return False

    else:
        minima_angle = 5 - 0.05 * minima_index
        minima_angle = round(minima_angle, 3)

        return minima_angle












folder_path = "PerfectNick_soft_rotation_1"
all_images_paths = []

num_images = 0
for path in os.listdir(folder_path):
    relative_path = os.path.join(folder_path, path)
    if os.path.isfile(relative_path):
        num_images += 1
        all_images_paths.append(relative_path)

all_images_paths.sort()



ball_presence = []
x_locations = []
y_locations = []
rotation_amounts = []

old_cutout = 0
new_cutout = 0

main_circle = find_ball_location(all_images_paths[17])
old_cutout = create_circle_cutout(all_images_paths[17], main_circle)
print("old", main_circle)


main_circle = find_ball_location(all_images_paths[18])
new_cutout = create_circle_cutout(all_images_paths[18], main_circle)
print("new", main_circle)


print(convolve_cutouts(old_cutout, new_cutout))



# for i in range(num_images):

#     main_circle = find_ball_location(all_images_paths[i])
#     new_cutout = create_circle_cutout(all_images_paths[i], main_circle)

#     if main_circle is not False:
#         print("ball at {}".format(i))

#     else:
#         print("ball not at {}".format(i))

#     if i != 0:
#         compare_cutouts(old_cutout, new_cutout)
