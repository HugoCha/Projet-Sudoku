#! /usr/bin/env python3

from typing import Set, Dict

all_filepathes = {
    0  : "Image/sudoku_photo0.jpg",
    1  : "Image/sudoku_photo1.jpg",
    2  : "Image/sudoku_photo2.jpg",
    3  : "Image/sudoku_photo3.jpg",
    4  : "Image/sudoku_photo4.jpg",
    5  : "Image/sudoku_photo5.jpg",
    6  : "Image/sudoku_photo6.jpg",
    7  : "Image/sudoku_photo7.jpg",
    8  : "Image/sudoku_photo8.jpg",
    9  : "Image/sudoku_photo9.jpg",
    10 : "Image/sudoku_photo10.jpg",
    11 : "Image/grille.jpg",
}

filepathes= {
    #all_filepathes[0],
    #all_filepathes[1],
    #all_filepathes[2],
    #all_filepathes[3],
    #all_filepathes[4],
    all_filepathes[5],
    all_filepathes[6],
    #all_filepathes[7],
    #all_filepathes[8],
    all_filepathes[9],
    all_filepathes[10],
    #all_filepathes[11],
}

all_imgs:Dict[int,str] = {
    0: "original",
    1: "resize",
    2: "normalize_image",
    3: "equalize_histogram",
    4: "sauvola_binarise",
    5: "erosion0",
    6: "clean_background",
    7: "bounding_box",
    8: "intersection",
    9: "erosion1"
}

img_to_display:Set[str] = { 
    #"all",
    #all_imgs[0],
    #all_imgs[1],
    #all_imgs[2],
    #all_imgs[3],
    #all_imgs[4],
    #all_imgs[5],
    all_imgs[6],
    all_imgs[7],
    #all_imgs[8]
    #all_imgs[9]
 }