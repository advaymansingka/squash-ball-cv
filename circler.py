import numpy as np
import math
import random
import cv2
from matplotlib import pyplot as plt
from pip import main
import os


# helper function to get the path string for the image based on image number
def image_path_generator(main_path_string: str, image_number: int) -> str:

    image_number = str(image_number)
    padding_size = 5 - len(image_number)
    padding = ""

    for _ in range(padding_size):
        padding += "0"

    image_number = padding + image_number

    return main_path_string + image_number + ".bmp"


# helper to find the midpoint of the ball given the image
def find_ball_location(main_path_string: str, image_number: int):
# main path string

    image_path = image_path_generator(main_path_string, image_number)

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


        contour_read_image = cv2.cvtColor(read_image, cv2.COLOR_BGR2GRAY)
        contour_read_image = cv2.threshold(contour_read_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        cnts = cv2.findContours(contour_read_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        print(cnts)
        # for cnt in cnts:
        #     approx = cv2.contourArea(cnt)
        #     print(approx)




        cv2.circle(read_image,(main_circle[0],main_circle[1]),main_circle[2],(0,255,0),2)
        # draw the center of the circle
        cv2.circle(read_image,(main_circle[0],main_circle[1]),2,(0,0,255),3)
        cv2.imshow('detected circles',read_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return main_circle

    else:

        cv2.imshow('detected circles',read_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return False





folder_path = "PerfectNick_soft_rotation_1"

num_images = 0
for ele in os.scandir(folder_path):
    num_images += os.stat(ele).st_size


x_locations = []
y_locations = []
main_path_string = "PerfectNick_soft_rotation_1/PerfectNick__soft_rotation_1_C001H001S00010"

for image_number in range(1, num_images-1):

    main_circle = find_ball_location(main_path_string, image_number)

    if main_circle is not False:
        print("ball at {}".format(image_number))

    else:
        print("ball not at {}".format(image_number))












# read_image = cv2.imread(image_path, 1)
# img = cv2.medianBlur(read_image,3)
# img_copy = img.copy()
# # Convert to greyscale
# img_gray = cv2.cvtColor(img_copy,cv2.COLOR_BGR2GRAY)

# # Apply Hough transform to greyscale image
# circles = cv2.HoughCircles(img_gray,cv2.HOUGH_GRADIENT,1,20,
#                      param1=60,param2=40,minRadius=0,maxRadius=0)
# circles = np.uint16(np.around(circles))
# # Draw the circles
# for i in circles[0,:]:
#     # draw the outer circle
#     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
#     # draw the center of the circle
#     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
# cv2.imshow('detected circles',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()