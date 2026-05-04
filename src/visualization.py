#! /usr/bin/env python3
# coding: utf-8

import cv2 
import sys

from cv2.typing import MatLike
from typing import Tuple, Set

from .config import *

def show_image_match( name_img_pair:Tuple[str,MatLike], img_to_display:Set[str] ):
    if ( "all" in img_to_display or name_img_pair[0] in img_to_display ):
        show_image( name_img_pair[0], name_img_pair[1] )

def show_image( img_name:str, img:MatLike ):
    print( f"show image: {img_name}" )
    cv2.imshow( img_name, img )
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if ( key == 27 or key == ord( 'q' ) ):
        sys.exit(0)