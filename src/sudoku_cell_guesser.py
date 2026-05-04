#! /usr/bin/env python3

import numpy as np
import sys

from typing import List

from src.feature_extractor import Feature, FeatureExtractor
from src.simple_sudoku_grid_extractor import SimpleSudokuGridExtractor
from src.vision_utils import ImageDimension
from src.visualization import show_image

from .config import *

class SudokuCellGuesser:
    def __init__( self, grid:Feature ):
        self.grid = grid

    def guess( self ) -> List[Feature]:
        dim = ImageDimension( self.grid.img )
        print( f"grid {dim}" )
        cell_h = dim.h // 9 
        cell_w = dim.w // 9

        cells:List[Feature] = []

        for i in range( 9 ):
            for j in range( 9 ):
                name = "cell guess " + str(i) + str(j)
                img = self.grid.img[i * cell_h:( i + 1 ) * cell_h , j * cell_w:( j + 1 ) * cell_w]
                pts = np.array(
                    [
                        [ j * cell_w, i * cell_h ],
                        [ ( j + 1 ) * cell_w, i * cell_h ],
                        [ ( j + 1 ) * cell_w, ( i + 1 ) * cell_h ],
                        [ j * cell_w, ( i + 1 ) * cell_h ],
                    ])
                cells.append( Feature( name, img, pts ) )

        return cells

def guess( extractor:FeatureExtractor, f_path:str ):
    grid = extractor.extract_from_path( f_path )
    if ( len(grid) == 1 ):
        guesser = SudokuCellGuesser( grid[0] )
        guesses = guesser.guess()
        for guess in guesses:
            print( guess )
            #show_image( guess.name, guess.img )
    else:
        print( "Sudoku grid not found" )

import cv2
if __name__ == "__main__":
    extractor = SimpleSudokuGridExtractor( False )

    if ( len( sys.argv ) <= 1 ):
        for fpath in filepathes:
            img = cv2.imread( fpath )
            guess( extractor, fpath )
    else:
        arg1 = sys.argv[1]
        fpath = str( arg1 )
        guess( extractor, fpath )