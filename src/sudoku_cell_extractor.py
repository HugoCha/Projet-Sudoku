#! /usr/bin/env python3

import cv2
import numpy as np
import sys

from cv2.typing import MatLike
from typing import List, Optional, Set

from src.feature_extractor import Feature, FeatureExtractor
from src.simple_sudoku_grid_extractor import SimpleSudokuGridExtractor
from src.vision_utils import *

from .config import *

class SudokuCellExtractor(FeatureExtractor):
    def __init__( self, grid_extractor:FeatureExtractor, can_show_img:bool ):
        super().__init__( can_show_img )
        self.grid_extractor = grid_extractor

        self.__image_to_display = {
            #"original",
            "grid",
            #"gray",
            #"blur",
            #"threshold",
            #"border",
            #"horizontal_lines",
            #"vertical_lines",
            "accumulate",
            "hough_lines"
        }

    def extract( self, img:MatLike ) -> List[Feature]:
        self.try_show_image( "original", img )

        grid = self.grid_extractor.extract( img )
        if ( len( grid ) != 1 ):
            return []
        
        return self.extract_from_grid( grid[0] )

    def extract_from_grid(self, grid:Feature ) -> List[Feature]:
        self.try_show_image( "grid", grid.img )

        gray = grayscale( grid.img )
        self.try_show_image( "gray", gray )

        blur = cv2.GaussianBlur(gray,(5,5),0)
        self.try_show_image( "blur", blur )

        th = cv2.adaptiveThreshold( gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,4 )
        self.try_show_image( "threshold", th )

        border_size = 4
        border = copy_make_inside_borders( 
            th, 
            0, 
            0, 
            border_size, 
            border_size, 
            value=0 )
        self.try_show_image( "border", border )
        horizontal_lines_img = border
        horizontal_lines = cv2.getStructuringElement( cv2.MORPH_RECT, ( 21, 1 ) )
        horizontal_lines_img = cv2.morphologyEx(horizontal_lines_img, cv2.MORPH_CLOSE, horizontal_lines, iterations=1)
        self.try_show_image( "horizontal_lines", horizontal_lines_img )

        border_size = 4
        border = copy_make_inside_borders( 
            th, 
            border_size, 
            border_size, 
            0, 
            0, 
            value=0 )
        self.try_show_image( "border", border )
        vertical_lines_img = border
        vertical_lines = cv2.getStructuringElement( cv2.MORPH_RECT, ( 1, 21 ) )
        vertical_lines_img = cv2.morphologyEx(vertical_lines_img, cv2.MORPH_CLOSE, vertical_lines, iterations=1)
        self.try_show_image( "vertical_lines", vertical_lines_img )

        accumulate = cv2.bitwise_and( vertical_lines_img, horizontal_lines_img )
        self.try_show_image( "accumulate", accumulate )
        
        lines = cv2.HoughLinesP(
            255 - accumulate,                # Input edge image
            rho=1,               # Distance resolution (in pixels)
            theta=np.pi/180,     # Angle resolution (in radians)
            threshold=100,       # Minimum number of votes to detect a line
            minLineLength=100,    # Minimum line length (in pixels)
            maxLineGap=10        # Maximum gap between line segments to treat as a single line
        )

        # Draw the detected lines on the original image
        hough_lines = accumulate
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(hough_lines, (x1, y1), (x2, y2), 0, thickness=2) 
        self.try_show_image( "hough_lines", hough_lines )

        return []
    
    def image_to_display(self) -> Set[str]:
        return self.__image_to_display

if __name__ == "__main__":
    grid_extractor = SimpleSudokuGridExtractor( False )
    cell_extractor = SudokuCellExtractor( grid_extractor, True )

    if ( len( sys.argv ) <= 1 ):
        for fpath in filepathes:
            cell_extractor.extract_from_path( fpath )
    else:
        arg1 = sys.argv[1]
        fpath = str( arg1 )
        cell_extractor.extract_from_path( fpath )