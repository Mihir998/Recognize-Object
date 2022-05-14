#!/usr/bin/env python
# coding: utf-8

# In[1]:
import win32gui
from PIL import ImageGrab
import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt
import numpy as np
import threading
from multiprocessing.pool import ThreadPool
import time

# In[2]:
%config InlineBackend.figure_format = 'svg'

options = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolo.weights',
    'threshold': 0.2,
    'gpu': 1.0
}

tfnet = TFNet(options)


# %%
cap = cv2.VideoCapture(0)


def rec(img):
    result = tfnet.return_predict(img)

    print(result)
    for a in result:
        tl = (a['topleft']['x'], a['topleft']['y'])
        br = (a['bottomright']['x'], a['bottomright']['y'])
        label = a['label']
        img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
        img = cv2.putText(
            img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    a = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('frame', a)


while(True):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

# %%
img = cv2.imread("abc.jpeg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
result = tfnet.return_predict(img)

for a in result:
    label = a['label']

    tl = (a['topleft']['x'], a['topleft']['y'])
    br = (a['bottomright']['x'], a['bottomright']['y'])

    img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
    img = cv2.putText(
        img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
plt.imshow(img)
plt.show()

# %%
hwnd = win32gui.FindWindow(None, r'DroidCam Video')
# win32gui.SetForegroundWindow(hwnd)
dimensions = win32gui.GetWindowRect(hwnd)


def predict(img):
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return tfnet.return_predict(converted_img)

#     for a in result:
#         label = a['label']

#         tl = (a['topleft']['x'], a['topleft']['y'])
#         br = (a['bottomright']['x'], a['bottomright']['y'])

#         img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
#         img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)


pool = ThreadPool(processes=1)

while(True):
    image = ImageGrab.grab(dimensions)
    stime = time.time()
    img = np.array(image)
    img = img[:, :, ::-1].copy()

    async_result = pool.apply_async(predict, (img, ))

    cv2.imshow('ImageWindow', img)

    result = async_result.get()

    for a in result:
        label = a['label']

        tl = (a['topleft']['x'], a['topleft']['y'])
        br = (a['bottomright']['x'], a['bottomright']['y'])

        img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
        img = cv2.putText(
            img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('ImageWindow', img)
    print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        # cap.release()
#         task.stop()
        break

# %%
