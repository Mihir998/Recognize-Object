import time
import cv2
import numpy as np
import numexpr as ne
import face_recognition
from imutils import paths
from darkflow.net.build import TFNet

DATASET_FILE_NAME = 'model_encodings.pickle'
FONT = cv2.FONT_HERSHEY_SIMPLEX

for threshold in [.60]:
    frame = cv2.imread("./sample.jpeg", -1)

    options = {
        'model': 'cfg/yolo.cfg',
        'load': 'bin/yolo.weights',
        'threshold': threshold,
        'gpu': 0.5
    }
    tfnet = TFNet(options)

    stime = time.time()
    # converted_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = tfnet.return_predict(frame)
    colors = [tuple(255 * np.random.rand(3)) for i in range(10)]

    for (color, result) in zip(colors, results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])

        label = result['label']

        img = cv2.rectangle(frame, tl, br, color, 7)
        img = cv2.putText(img, label, tl, FONT, 1, color, 2)

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

    cv2.imwrite("outputs/accuracy_"+str(threshold)+".jpeg", img)
