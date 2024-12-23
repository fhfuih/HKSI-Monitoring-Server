from typing import Annotated

import cv2
import dlib
import numpy as np
import numpy.typing as npt


class PimpleDetection:
    def __init__(self, ckpt_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(ckpt_path)

    def get_face_landmarks(self, image):
        # detector = dlib.get_frontal_face_detector()
        # predictor = dlib.shape_predictor("shape_predictor_81_face_landmarks.dat")
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray_image)
        face_landmarks = []

        for face in faces:
            landmarks = self.predictor(gray_image, face)
            for i in range(0, 81):
                face_landmarks.append((landmarks.part(i).x, landmarks.part(i).y))

        return face_landmarks

    def create_face_mask(
        self, image, contour
    ) -> tuple[Annotated[bool, "face_exist"], Annotated[npt.NDArray, "mask"]]:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        face_poly = contour[:17] + contour[68:81]

        if not len(face_poly):
            return False, mask

        points = np.array(face_poly, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)

        # vis_image = image.copy()
        # for point in contour:
        #     cv2.circle(vis_image, point, 3, (255, 255, 0), -1)
        # cv2.imshow('vis_image', vis_image)
        return True, mask

    def color_filter(self, image):
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define the color range for red and pink
        lower_red1 = np.array([0, 150, 160], dtype=np.uint8)
        upper_red1 = np.array([10, 200, 255], dtype=np.uint8)
        # lower_red2 = np.array([160, 70, 50], dtype=np.uint8)
        # upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

        # Create masks for red and pink regions
        mask = cv2.inRange(hsv_image, lower_red1, upper_red1)
        # mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        # mask = cv2.bitwise_or(mask1, mask2)

        return mask

    def remove_noise(self, mask):
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    # Function to draw bounding boxes around detected regions
    def get_pimples(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pimple_num = len(contours)
        pimple_bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            pimple_bboxes.append((x, y, w, h))

        return pimple_num, pimple_bboxes

    # Function to draw detected circles
    def draw_circles(self, image, circles):
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for x, y, r in circles:
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)

    # Function to filter out eye and lip regions using landmarks
    def filter_face_regions(self, mask, landmarks):
        for x, y in landmarks:
            cv2.circle(mask, (x, y), 15, 0, -1)
        return mask

    # Main function to detect pimples
    def run(
        self, image
    ) -> tuple[
        Annotated[bool, "face_exist"],
        Annotated[int, "pimple_num"],
        Annotated[list, "pimple_bboxes"],
    ]:
        landmarks = self.get_face_landmarks(image)

        # Step 1: Create a face mask using face landmarks
        face_exist, face_mask = self.create_face_mask(image, landmarks)
        if not face_exist:
            return False, 0, []

        # Step 2: Color filtering to detect red and pink regions
        color_mask = self.color_filter(image)
        # Step 3: Apply face mask to limit detection to the face area
        color_mask = cv2.bitwise_and(color_mask, face_mask)
        # Step 4: Filter out eye and lip regions using landmarks
        filtered_mask = self.filter_face_regions(color_mask, landmarks)
        # Step 5: Remove noise using morphological operations
        clean_mask = self.remove_noise(filtered_mask)
        # Step 6: Draw bounding boxes around detected regions
        pimple_num, pimple_bboxes = self.get_pimples(clean_mask)

        return True, pimple_num, pimple_bboxes
