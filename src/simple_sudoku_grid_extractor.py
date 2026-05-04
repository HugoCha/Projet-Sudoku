#! /usr/bin/env python3

import cv2
import numpy as np
import sys

from cv2.typing import MatLike
from typing import List, Optional, Set

from src.feature_extractor import Feature, FeatureExtractor
from src.vision_utils import grayscale

from .config import *

class SimpleSudokuGridExtractor(FeatureExtractor):
    def __init__( self, can_show_img:bool ):
        super().__init__( can_show_img )

        self.__image_to_display:Set[str] = {
            "original",
            #"gray",
            #"blur",
            #"threshold",
            #"morphology",
            #"morphology_inverse",
            #"max_contour"
        }

    def isolate_grid( self, img: MatLike ) -> MatLike:
        gray = grayscale( img )
        self.try_show_image( "gray", gray )

        blur = cv2.GaussianBlur(gray,(7,7),0)
        self.try_show_image( "blur", blur )

        th = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,4 )
        self.try_show_image( "threshold", th )
        
        kernel = np.ones((5,5), np.uint8)
        morphology = cv2.morphologyEx(th, cv2.MORPH_ERODE, kernel, iterations=1)
        self.try_show_image( "morphology", morphology )

        morphology_inv = 255 - morphology
        self.try_show_image( "morphology_inverse", morphology_inv )
        
        return morphology_inv

    def select_best_grid_candidate( self, img:MatLike, original_img:MatLike ) -> Optional[Feature]:
        contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if ( len( contours ) == 0 ): 
            print( f"SimpleSudokuGridExtractor: No contour detected")
            return None

        best_contour = max( contours, key=cv2.contourArea )
        best_contour_img = cv2.drawContours( original_img.copy(), [best_contour], -1, (125,0,255), 3 )
        self.try_show_image( "max_contour", best_contour_img )
    
        img_area = img.shape[0] * img.shape[1]

        hull = cv2.convexHull(best_contour)
        hull_area = cv2.contourArea(hull)
        if hull_area < 0.75 * img_area: 
            print( f"SimpleSudokuGridExtractor: Hull Area insufficient: {hull_area} expected: {0.9 * img_area}")
            return None

        area = cv2.contourArea(best_contour)
        solidity = float(area) / hull_area

        if solidity < 0.85: 
            print( f"SimpleSudokuGridExtractor: Solidity insufficient: {solidity}")
            return None

        epsilon = 0.02 * cv2.arcLength(best_contour, True)
        approx = cv2.approxPolyDP(best_contour, epsilon, True)

        if ( approx.shape[0] != 4 ):
            print( f"SimpleSudokuGridExtractor: Approx poly size mismatch: {approx.shape}")
            return None

        approx = approx.astype(np.int32).reshape(4, 2)
        
        copy = original_img.copy()
        x_min, x_max = np.min(approx[:, 0]), np.max(approx[:, 0])
        y_min, y_max = np.min(approx[:, 1]), np.max(approx[:, 1])

        grid = copy[y_min:y_max, x_min:x_max]
        return Feature( "Sudoku grid", grid, approx )

    def extract( self, img:MatLike ) -> List[Feature]:
        self.try_show_image( "original", img )
        morphology_inv = self.isolate_grid( img )
        grid = self.select_best_grid_candidate( morphology_inv, img )
        return [grid] if grid is not None else []
    
    def image_to_display(self) -> Set[str]:
        return self.__image_to_display

if __name__ == "__main__":
    extractor = SimpleSudokuGridExtractor( True )

    if ( len( sys.argv ) <= 1 ):
        for fpath in filepathes:
            extractor.extract_from_path( fpath )
    else:
        arg1 = sys.argv[1]
        fpath = str( arg1 )
        extractor.extract_from_path( fpath )