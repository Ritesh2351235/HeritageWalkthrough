#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque

import cv2 as cv
import numpy as np
import mediapipe as mp
import keyboard
import time
import pyautogui

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

# Argument parsing #################################################################
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    gesture_map = {
        "No Key":None,
        "W":"W",
        "S":"S",
        "A":"A",
        "D":"D"
    }

    current_gesture = None
    last_gesture_time = 0
    DEBOUNCE_DELAY =0.2
    pressed_key =None

    # Hand gesture recognition
    hand_detector = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    drawing_utils = mp.solutions.drawing_utils

    prev_frame_time = 0
    new_frame_time = 0

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Bounding box calculation
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])


                gesture = gesture_map.get(
                    keypoint_classifier_labels[hand_sign_id]
                )

                #Handle gesture changes with debounce
                if gesture != current_gesture:
                    current_time = time.time()
                    if current_time - last_gesture_time >= DEBOUNCE_DELAY:
                        #Release the previously pressed key
                        if pressed_key and (pressed_key != gesture):
                            keyboard.release(pressed_key)
                            print(f"Released {pressed_key}")

                        if gesture is not None:
                            keyboard.press(gesture)
                            print(f"Pressed {gesture}")
                            pressed_key = gesture

                        current_gesture = gesture
                        last_gesture_time = current_time    
                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == history_length:
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Display box drawing ###############################################
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_info_text(debug_image, brect, handedness,
                                             keypoint_classifier_labels[hand_sign_id],
                                             point_history_classifier_labels[finger_gesture_id],
                                             (10, 20), (0, 0, 255))
                debug_image = drawing_landmarks(debug_image, landmark_list)

        # Display ##################################################################
        cv.imshow('MediaPipe Hands Demo', debug_image)
        fps = cvFpsCalc.calc(cv.getTickCount())
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        print("FPS:", fps)

        # ESC key for break #######################################################
        if cv.waitKey(1) == 27:
            break

    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmark_list):
    if len(landmark_list) < 1:
        return None
    landmark_array = np.empty((0, 2), int)
    for landmark in landmark_list:
        landmark_array = np.append(landmark_array, np.array([[landmark[0], landmark[1]]]), axis=0)

    brect = cv.boundingRect(landmark_array)
    cv.rectangle(image, (brect[0], brect[1]), (brect[0] + brect[2], brect[1] + brect[3]),
                 (255, 0, 0), 1)
    return brect


def calc_landmark_list(image, hand_landmarks):
    landmark_list = []
    image_height, image_width, _ = image.shape
    for _idx, landmark in enumerate(hand_landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_list.append([landmark_x, landmark_y])
    return landmark_list


def pre_process_landmark(landmark_list):
    pre_processed_landmark_list = []
    if len(landmark_list) > 0:
        base_x, base_y = landmark_list[0]
        for landmark in landmark_list:
            landmark_x, landmark_y = landmark
            pre_processed_landmark_list.append([
                landmark_x - base_x, landmark_y - base_y
            ])
    return pre_processed_landmark_list


def pre_process_point_history(debug_image, point_history):
    if len(point_history) == 0:
        return [[0, 0]] * 16
    else:
        base_x, base_y = point_history[0]
        pre_processed_point_history_list = []
        for point in point_history:
            pre_processed_point_history_list.append([
                point[0] - base_x, point[1] - base_y
            ])
        return pre_processed_point_history_list


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[0] + brect[2], brect[1] + brect[3]),
                     (255, 0, 0), 1)
    return image


def draw_info_text(image, brect, handedness, hand_sign_text, finger_gesture_text,
                   position, color):
    cv.putText(image, f'{handedness.classification[0].label} : {handedness.score:.2f}',
               (position[0], position[1] + 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
    cv.putText(image, f'Hand sign  : {hand_sign_text}', (position[0], position[1] + 60),
               cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
    cv.putText(image, f'Gesture    : {finger_gesture_text}', (position[0], position[1] + 90),
               cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
    return image


def drawing_landmarks(image, landmark_list):
    if len(landmark_list) > 0:
        for landmark in landmark_list:
            cv.circle(image, tuple(landmark), 5, (0, 255, 0), thickness=-1)
    return image


def logging_csv(number, mode, landmark_list, point_history_list):
    landmark_list = list(itertools.chain.from_iterable(landmark_list))
    point_history_list = list(itertools.chain.from_iterable(point_history_list))
    data = landmark_list + point_history_list
    data.append(mode)

    with open('mediapipe_test.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)


def select_mode(key, mode):
    if key == 49:  # '1' key
        mode = 0
        print('mode:', mode)
    elif key == 50:  # '2' key
        mode = 1
        print('mode:', mode)
    elif key == 51:  # '3' key
        mode = 2
        print('mode:', mode)
    elif key == 52:  # '4' key
        mode = 3
        print('mode:', mode)
    return key, mode


if __name__ == '__main__':
    main()

