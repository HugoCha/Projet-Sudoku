#!/usr/bin/env python3

import cv2
import numpy as np

from cv2.typing import MatLike, Scalar

class ImageDimension:
    def __init__( self, img:MatLike ):
        self.h = img.shape[0]
        self.w = img.shape[1]
        self.channels = img.shape[2] if len( img.shape ) >= 3 else 1

    def __str__( self ) -> str:
        return f"width: {self.w}, height: {self.h}, channels: {self.channels}"

def copy_make_inside_borders( 
        img:MatLike,
        top:int,
        bottom:int,
        left:int,
        right:int,
        value:Scalar ) -> MatLike:
    border = img.copy()
    dim = ImageDimension( border )
    print( dim )
    if ( left > 0 ):
        left_center = left // 2
        pt1 = [left_center, 0]
        pt2 = [left_center, dim.h]
        cv2.line( border, pt1, pt2, value, left )
    
    if ( right > 0 ):
        right_center = right // 2
        pt1 = [dim.w - right_center, 0]
        pt2 = [dim.w - right_center, dim.h]
        cv2.line( border, pt1, pt2, value, right )
    
    if ( top > 0 ):
        top_center = top // 2
        pt1 = [0, top_center]
        pt2 = [dim.w, top_center]
        cv2.line( border, pt1, pt2, value, top )

    if ( bottom > 0 ):
        bottom_center = bottom // 2
        pt1 = [0, dim.h - bottom_center]
        pt2 = [dim.w, dim.h - bottom_center]
        cv2.line( border, pt1, pt2, value, bottom )

    return border


def crop( img:MatLike, x_min, x_max, y_min, y_max ):
    return img[y_min:y_max, x_min:x_max]

def is_grayscale( img:MatLike ) -> bool:
    return len( img.shape ) == 2

def grayscale( img:MatLike ) -> MatLike:
    return cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )

def gamma_correction( img:MatLike, gamma:float=1.5 ) -> MatLike:
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def otsu( img:MatLike ) -> MatLike:
    ret = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return ret[1]

def is_valid_contour( contour:MatLike ) -> bool:
    area = cv2.contourArea(contour)

    if area < 500 or area > 15000:
        return False

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    if hull_area == 0:
        return False

    solidity = float(area) / hull_area

    if solidity < 0.6:
        return False

    return True

def compute_circularity( contour:MatLike ) -> float:
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter <= 1e-6:
        return 0.0

    circularity = (4 * np.pi * area) / (perimeter * perimeter)
    return circularity

def mask_background(
        img:MatLike,
        min_area:int = 1000 ) -> MatLike:
    if is_grayscale( img ):
        gray = img
    else:
        gray = grayscale( img )

    blur = cv2.GaussianBlur(gray,(7,7),0)
    th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,4 )
    kernel = np.ones((5,5), np.uint8)
    morphology = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = np.ones((7,7), np.uint8)
    morphology = cv2.morphologyEx(morphology, cv2.MORPH_ERODE, kernel, iterations=2)

    contours, _ = cv2.findContours(morphology, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, filtered_contours, -1, 255, thickness=cv2.FILLED)

    if ( is_grayscale(img) ):
        background = np.full_like(img, 255, dtype=np.uint8)
    else:
        background = np.full_like(img, (255,255,255), dtype=np.uint8)
    uniform_bg = cv2.bitwise_and(background, background, mask=mask)
    foreground = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(mask))
    img_with_mask = cv2.add(uniform_bg, foreground)

    return img_with_mask


def extract_bounding_box_image( img, box ):

    def order_points_clockwise(points):
        centroid = np.mean(points, axis=0)
        def angle(p):
            vec = p - centroid
            return np.arctan2(vec[1], vec[0])
        return np.array(sorted(points, key=angle), dtype=np.float32)

    box = order_points_clockwise( box )
    width = int(max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[2] - box[3])))
    height = int(max(np.linalg.norm(box[0] - box[3]), np.linalg.norm(box[1] - box[2])))

    dst_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(box, dst_points)
    sub_image = cv2.warpPerspective(img, M, (width, height))

    return sub_image