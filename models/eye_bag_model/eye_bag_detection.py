import cv2
import dlib
import numpy as np
import math
import sys


class EyeBagDetection:
    def __init__(self, ckpt_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(ckpt_path)

    # Function to get face landmarks using dlib
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

    def sort_points(self, points):
        # 计算多边形的质心
        points = np.array(points, np.int32)
        center = points.mean(axis=0)

        # 计算每个点的极角
        def angle_from_center(point):
            return math.atan2(point[1] - center[1], point[0] - center[0])

        # 按极角对点进行排序
        sorted_points = sorted(points, key=angle_from_center)
        return np.array(sorted_points, np.int32).tolist()

    # Function to extract the average skin color from face landmarks
    def get_average_skin_color(self, image, landmarks):
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        face_poly = landmarks[:17] + landmarks[68:81]
        points = np.array(
            face_poly, dtype=np.int32
        )  # Use the first 68 points for the face
        cv2.fillPoly(mask, [points], 255)
        skin_pixels = cv2.bitwise_and(image, image, mask=mask)
        skin_pixels = skin_pixels[np.where(mask == 255)]
        average_color = np.mean(skin_pixels, axis=0)
        return average_color

    # Function to get eye bags regions
    def get_eye_bags_regions(self, landmarks, offset=5):
        # left: 36, 39, 40, 41
        # right: 42, 45, 46, 47
        left_eye = [landmarks[36], landmarks[39], landmarks[40], landmarks[41]]
        left_eye_height = max(
            landmarks[41][1] - landmarks[37][1], landmarks[40][1] - landmarks[38][1]
        )
        right_eye = [landmarks[42], landmarks[45], landmarks[46], landmarks[47]]
        right_eye_height = max(
            landmarks[47][1] - landmarks[43][1], landmarks[46][1] - landmarks[44][1]
        )

        left_eye_bag = [(x, y + offset) for (x, y) in left_eye] + [
            (x, y + left_eye_height) for (x, y) in left_eye
        ]
        # print(left_eye_bag)
        left_eye_bag = self.sort_points(left_eye_bag)
        right_eye_bag = [(x, y + offset) for (x, y) in right_eye] + [
            (x, y + right_eye_height) for (x, y) in right_eye
        ]
        right_eye_bag = self.sort_points(right_eye_bag)
        return left_eye_bag, right_eye_bag

    # draw lines to connect each key point for each eye
    def draw_eye_lines(self, image, landmarks):
        left_eye = [landmarks[36], landmarks[39], landmarks[40], landmarks[41]]
        right_eye = [landmarks[42], landmarks[45], landmarks[46], landmarks[47]]

        for i in range(4):
            cv2.line(image, left_eye[i], left_eye[(i + 1) % 4], (0, 255, 0), 2)
            cv2.line(image, right_eye[i], right_eye[(i + 1) % 4], (0, 255, 0), 2)
        return image

    # Function to calculate color distance
    def color_distance(self, color1, color2):
        return np.linalg.norm(color1 - color2)

    def create_polygon_mask(self, image_size, polygon_points, mask=None):
        """
        创建一个包含多边形区域的mask图像。

        :param image_size: (高度, 宽度) 元组，表示图像的大小。
        :param polygon_points: 多边形顶点的列表，每个顶点是 (x, y) 元组。
        :return: mask图像，区域内值为255，外部为0。
        """
        # 创建一个全零的黑色图像
        if mask is None:
            mask = np.zeros(image_size, dtype=np.uint8)

        # 将多边形顶点转换为整数
        polygon_points = np.array(polygon_points, dtype=np.int32)

        # 使用cv2.fillPoly填充多边形区域
        cv2.fillPoly(mask, [polygon_points], color=255)

        return mask

    # Function to check for dark circles
    def detect_dark_circles(self, image, landmarks, average_skin_color, threshold=40):
        left_eye_bag, right_eye_bag = self.get_eye_bags_regions(landmarks)

        mask_left = np.zeros(image.shape[:2], dtype=np.uint8)
        mask_right = np.zeros(image.shape[:2], dtype=np.uint8)

        cv2.fillPoly(mask_left, [np.array(left_eye_bag, dtype=np.int32)], 255)
        cv2.fillPoly(mask_right, [np.array(right_eye_bag, dtype=np.int32)], 255)

        left_eye_pixels = cv2.bitwise_and(image, image, mask=mask_left)
        right_eye_pixels = cv2.bitwise_and(image, image, mask=mask_right)

        left_eye_pixels = left_eye_pixels[np.where(mask_left == 255)]
        right_eye_pixels = right_eye_pixels[np.where(mask_right == 255)]

        left_eye_color_distance = np.mean(
            [
                self.color_distance(pixel, average_skin_color)
                for pixel in left_eye_pixels
            ]
        )
        right_eye_color_distance = np.mean(
            [
                self.color_distance(pixel, average_skin_color)
                for pixel in right_eye_pixels
            ]
        )
        return (
            left_eye_color_distance > threshold,
            right_eye_color_distance > threshold,
            left_eye_bag,
            right_eye_bag,
        )

    # Main function to detect dark circles
    def run(self, image):
        landmarks = self.get_face_landmarks(image)
        has_dark_circles_left = False
        has_dark_circles_right = False

        average_skin_color = self.get_average_skin_color(image, landmarks)
        has_dark_circles_left, has_dark_circles_right, left_eye_bag, right_eye_bag = (
            self.detect_dark_circles(image, landmarks, average_skin_color)
        )

        return (
            has_dark_circles_left,
            has_dark_circles_right,
            left_eye_bag,
            right_eye_bag,
        )
