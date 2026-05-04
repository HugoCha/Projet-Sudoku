#! /usr/bin/env python3

import cv2
import numpy as np

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Set, List

from cv2.typing import MatLike

from src.vision_utils import ImageDimension
from src.visualization import show_image

class Feature:
    def __init__( self,
                  name:str,
                  img:MatLike,
                  points:MatLike ):
        self.name = name
        self.img = img
        self.points = points
        self.__dim = ImageDimension( img )
        self.__center:Optional[MatLike] = None

    @property
    def image_dimension( self ):
        return self.__dim
    
    @property
    def image_height( self ):
        return self.__dim.h
    
    @property
    def image_width( self ):
        return self.__dim.w

    @property
    def center( self ) -> MatLike:
        if ( self.__center is None ):
            self.__center = np.mean( self.points, axis=0 )
        return self.__center

    def __str__( self ) -> str:
        return f"{self.name} w: {self.image_width}, h: {self.image_height} center: {self.center}"

class FeatureExtractor(ABC):
    def __init__( self, can_show_image:bool=False ):
        self.can_show_image = can_show_image

    @abstractmethod
    def extract( self, img:MatLike ) -> List[Feature]:
        pass

    def extract_from_path( self, fpath:str, show_result:bool = True ) -> List[Feature]:
        img = cv2.imread( fpath )
        if ( img is None ):
            print( "Unable to read image" )
            return []

        features = self.extract( img )
        
        if ( show_result ):
            for feature in features:
                print( feature )
                self.try_show_image( feature.name, feature.img )
            
        return features

    @abstractmethod
    def image_to_display( self ) -> Set[str]:
        pass

    def try_show_image( self, name:str, img:MatLike ):
        if ( self.can_show_image and name in self.image_to_display() ):
            show_image( name, img )


