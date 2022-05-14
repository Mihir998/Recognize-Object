
from webcolors import rgb_percent_to_hex, hex_to_name, hex_to_rgb
import os
from darkflow.net.build import TFNet
import time
import cv2
import numpy as np
import numexpr as ne
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
import threading
import webcolors
from sklearn.cluster import KMeans
import pyttsx3
# import pickle

FONT = cv2.FONT_HERSHEY_SIMPLEX

options = {
    'model': 'cfg/yolov2-tiny.cfg',
    'load': 'bin/yolov2-tiny.weights',
    'threshold': 0.42,
    'gpu': 0.82
}

tfnet = TFNet(options)

BLUR = 24
CANNY_THRESH_1 = 10
CANNY_THRESH_2 = 200
MASK_DILATE_ITER = 10
MASK_ERODE_ITER = 10
MASK_COLOR = (0.0, 0.0, 1.0)  # In BGR format

engine = pyttsx3.init()
colors = [tuple(255 * np.random.rand(3)) for i in range(7)]
# data = pickle.loads(open(DATASET_FILE_NAME, "rb").read())


def predict(img):
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return tfnet.return_predict(converted_img)


def say(label):
    try:
        engine.say(label)
        engine.runAndWait()
        engine.stop()
    except:  # Exception as e:
        pass


object_with_colors = [
    'car',
    'motorbike',
    "bird",
    "cat",
    "dog",
    "horse",
    "cow",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "kite",
    "skateboard",
    "bottle",
    "cup",
    "fork",
    "knife",
    "spoon"
    "cell phone",
    "knife",
    "sofa",
    "mouse",
    "cake",
    "clock",
    "toothbrush",
    'mouse',
    'bottle',
    'laptop'
]


def fetchMetaInformation(result, image):
    label = result['label']

    #     tl = (result['topleft']['x'],result['topleft']['y'])
    #     br = (result['bottomright']['x'],result['bottomright']['y'])

    #     clipped = image[tl[1]:br[1], tl[0]:br[0]]

    #     if label in object_with_colors:
    #         try:
    #             clipped = rm_bg(clipped)
    #             _,actual = color(clipped)
    #         except:
    #             actual = ''
    #             pass

    # #         clipped = cv2.putText(clipped, actual, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    #         return "%s %s" % (actual, label)

    #     if label == "person":
    #         label = get_name(image)
    #         print("person", label)
    return label


def recz(img, prev_output):
    label = ""
    async_result = pool.apply_async(predict, (img, ))

    stime = time.time()
    results = async_result.get()

    out_string = "There is "
    c = 0

    for clr, result in zip(colors, results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])

        label = fetchMetaInformation(result, img)

        out_string += "a %s, " % (label)

        img = cv2.rectangle(img, tl, br, clr, 7)
        img = cv2.putText(
            img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

        c += 1

    if(c > 0 and prev_output != out_string):
        threading.Thread(target=say, args=(out_string + ".", )).start()
        prev_output = out_string

    fps = 1 / (time.time() - stime)

    img = cv2.putText(
        img,
        'FPS {:.1f}'.format(fps),
        (10, 40),
        cv2.FONT_HERSHEY_COMPLEX,
        .8,
        (255, 255, 255),
        2
    )

    # print('\rFPS {:.1f}'.format(fps), end="\r")

    return (img, prev_output)


def rm_bg(img):
    hMin = 29  # Hue minimum
    sMin = 30  # Saturation minimum
    vMin = 0   # Value minimum (Also referred to as brightness)
    hMax = 179  # Hue maximum
    sMax = 255  # Saturation maximum
    vMax = 255  # Value maximum
    # Set the minimum and max HSV values to display in the output image using numpys' array function. We need the numpy array since OpenCVs' inRange function will use those.
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])
    # Create HSV Image and threshold it into the proper range.
    # Converting color space from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Create a mask based on the lower and upper range, using the new HSV image
    mask = cv2.inRange(hsv, lower, upper)
    # Create the output image, using the mask created above. This will perform the removal of all unneeded colors, but will keep a black background.
    output = cv2.bitwise_and(img, img, mask=mask)
    # Add an alpha channel, and update the output image variable
    *_, alpha = cv2.split(output)
    dst = cv2.merge((output, alpha))
    return output


def color(img):
    data = np.reshape(img, (-1, 3))
    data = np.float32(data)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, labels, centers = cv2.kmeans(
        data, 1, None, criteria, 10, flags)

    color = centers[0].astype(np.int32)
    color = (color[0], color[1], color[2])
    color_hex = '#%02x%02x%02x' % color

    return (color_hex, get_colour_name(color)[1])


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


cap = cv2.VideoCapture(0)

pool = ThreadPool(processes=10)
prev_output = ""

engine.say("Started")
while(True):
    ret, frame = cap.read()

    img, prev_output = recz(frame, prev_output)

    cv2.imshow('ImageWindow', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break
