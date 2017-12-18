# encoding=utf-8

import cv2
import sys
import os

reload(sys)
sys.setdefaultencoding("utf8")

input_pattern = '/Volumes/猪脸识别/video/%d.mp4'
output_dir = '/Volumes/Wanzy/data/pig'

for i in range(29):
    index = i + 2
    input_path = input_pattern % index
    output_pattern = os.path.join(output_dir, '%d' % index)
    if not os.path.isdir(output_pattern):
        os.makedirs(output_pattern)
    cap = cv2.VideoCapture(input_path)
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        output_path = os.path.join(output_pattern, '%d.jpg' % count)
        cv2.imwrite(output_path, frame)

    cap.release()
    #cap.destroyAllWindows()
