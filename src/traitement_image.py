#! /usr/bin/env python3
# coding: utf-8

import cv2 
import numpy as np
import time
import matplotlib.pyplot as plt
from operator import itemgetter
import sys
import logging as lg
"""
filename = "/home/hugo/Sudoku/sudoku_original.jpg"
filename2 = "/home/hugo/Sudoku/sudoku_photo.jpg"
"""

def normalize_image(img, h, w, alpha, beta):
    #Set the range of pixel values between alpha and beta
    new_img = np.copy(img)
    max_img = img.max()
    min_img = img.min()
    diff = new_img - min_img
    coeff = (beta-alpha)/float(max_img-min_img)
    new_img = alpha + coeff*diff

    return new_img

def inverse_image(img):
    inverse_bin = 255 - img
    return inverse_bin

def integer_mean(img, h, w):
    int_mean = np.zeros([h,w])
    for j in range(0, w):
        for i in range(0, h):
            if (i != 0 and j != 0):
                int_mean[i,j] = img[i, j] + int_mean[i-1, j] + int_mean[i, j-1] - int_mean[i-1, j-1]
            elif (i == 0 and j != 0):
                int_mean[i,j] = img[i, j] + int_mean[i, j-1]
            elif (j == 0 and i != 0):
                int_mean[i,j] = img[i, j] + int_mean[i-1, j]
            else:
                int_mean[0,0] = img[0,0]
    return int_mean


def local_mean(int_mean, img, h, w, i, j, v):
    if(i<=v/2 or j<=v/2 or i>=h-v/2 or j>=w-v/2):
        return img[i, j]
    else:
        return ((int_mean[i+v//2][j+v//2] + int_mean[i-v//2][j-v//2] - int_mean[i-v//2][j+v//2] - int_mean[i+v//2][j-v//2])/float(v*v))

def local_mean_array(img, h, w, v):
    int_mean = integer_mean(img, h, w)
    mean_array = np.zeros([h, w])
    for i in range(0, h):
        for j in range(0, w):
            mean_array[i,j] = local_mean(int_mean, img, h, w, i, j, v)
    return mean_array

def ecart_type_local_array(mean_array, img, h, w, v):
    var_array = np.zeros([h, w])
    #square_mean_array = np.multiply(mean_array, mean_array)
    #square_img = np.multiply(img, img)
    ones_v = np.ones([v,v])
    for i in range(v//2, h-v//2):
        for j in range(v//2, w-v//2):
            array = img[i-v//2:i+v//2, j-v//2:j+v//2] - mean_array[i, j]*ones_v
            var_array[i, j] = np.sqrt(np.sum(np.multiply(array, array)))/(v*v)
    return var_array

def sauvola_binarise(img, h, w, v, k):
    binarize = np.zeros([h,w])
    mean = local_mean_array(img, h, w, v)
    ecart_type = ecart_type_local_array(mean, img, h, w, v)
    seuil = mean * (1 + k * ((ecart_type/128.) -1))
    binarize = ((img>=seuil)*255.)
    return binarize

def Houghline(img, h, w, line_length):
    # Hough Line from open cv
    lines = cv2.HoughLines(img.astype(np.uint8),2,(np.pi)/(180),line_length)
    
    lignes_hor = []
    lignes_vert = []
    
    void = np.ones([h,w])*255
    
    # Limit the number of lines from Hough Transform
    number_of_lines = len(lines) if (len(lines)<60) else 60

    number_of_lines_hor = 0
    number_of_lines_vert = 0

    for i in range(0,number_of_lines):
        # Hough function returns two polar coordinates, they are converted in cartesian coordinates
        for rho,theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))

            # In order to simplify two coordinates describes a line (slop, intercept of the line)
            if (x1 != x2):
                pente = ((y1-y2)/float(x1-x2))
                b = (x2*y1 - x1*y2)/(x2-x1)
            elif (x1 == x2 and y1 != y2):
                pente = h
                b = -pente * x1
            else:
                pente = 1/float(w)
                b = y1
            
            # Separation between "horizontal" and "vertical" lines
            if ( abs(pente) > 1 ):
                if (abs(pente)>h):
                    pente = h
                    b = -pente * (x1+x2)/2.
                lignes_vert.append((i, pente, b,  -b/pente))
                number_of_lines_vert +=1
            else:
                lignes_hor.append((i, pente, b, b))
                number_of_lines_hor += 1
        cv2.line(void,(x1,y1),(x2,y2),0,2)
    #cv2.imshow("Grid", void)
    return lignes_vert, lignes_hor, number_of_lines_vert, number_of_lines_hor, void

"""
def clean_hough_lines_hor(img, lignes_hor, height, width):
    # Selection of lines
    lignes_hor = sorted(lignes_hor, key=itemgetter(2))
    ord_ligne_hor = [e[2] for e in lignes_hor]
    pente = np.array([e[1] for e in lignes_hor])
    
    seuil_hor  = (ord_ligne_hor[-1] - ord_ligne_hor[0])/20.

    slop_mean_global = np.mean(pente)
    slop_standard_dev = np.sqrt(np.var(pente))
    slop_standard_dev = slop_standard_dev if (slop_standard_dev> 2 * abs(slop_mean_global)) else 2 * abs(slop_mean_global)

    pente_moy = 0.
    b_moy = 0.
    compt = 0.
    first_of_group = ord_ligne_hor[0]

    pente_hor = []
    b_hor = []
    for i in range(1, len(ord_ligne_hor)):
        if (i >= len(ord_ligne_hor)-1):
            if (abs(ord_ligne_hor[i]-ord_ligne_hor[i-1]) < seuil_hor and abs(ord_ligne_hor[i]-first_of_group) < 2*seuil_hor):
                    if ((pente[i] > slop_mean_global-slop_standard_dev) and (pente[i] < slop_mean_global+slop_standard_dev)):
                        pente_moy += pente[i]
                        b_moy += ord_ligne_hor[i]
                        compt +=1
                    if (compt>0):    
                        pente_moy = pente_moy/compt 
                        b_moy = b_moy/compt

                        pente_hor.append(pente_moy)
                        b_hor.append(b_moy)
            else:
                if (compt>0):    
                    pente_moy = pente_moy/compt 
                    b_moy = b_moy/compt

                    pente_hor.append(pente_moy)
                    b_hor.append(b_moy)
                if ((pente[i] > slop_mean_global-slop_standard_dev) and (pente[i] < slop_mean_global+slop_standard_dev)):
                    pente_hor.append(pente[i])
                    b_hor.append(ord_ligne_hor[i])
            
            

        else:
            if (abs(ord_ligne_hor[i]-ord_ligne_hor[i-1]) < seuil_hor and abs(ord_ligne_hor[i]-first_of_group) < 2*seuil_hor):
                if ((pente[i-1] > slop_mean_global-slop_standard_dev) and (pente[i-1] < slop_mean_global+slop_standard_dev)):
                    pente_moy += pente[i-1]
                    b_moy += ord_ligne_hor[i-1]
                    compt +=1
            else:
                if (compt>0):
                    pente_moy += pente[i-1]
                    b_moy += ord_ligne_hor[i-1]
                    compt +=1

                    pente_moy = pente_moy/compt #if (pente_moy > 1/float(width)) else 1/float(width)
                    b_moy = b_moy/compt
                    
                    pente_hor.append(pente_moy)
                    b_hor.append(b_moy)
                    
                else:
                    if ((pente[i-1] > slop_mean_global-slop_standard_dev) and (pente[i-1] < slop_mean_global+slop_standard_dev)):
                        pente_hor.append(pente[i-1])
                        b_hor.append(ord_ligne_hor[i-1])
                
                first_of_group = ord_ligne_hor[i]
                pente_moy = 0
                b_moy = 0
                compt = 0.
    index2del = []
    for i in range(0, len(pente_hor)-1):
        if (abs((pente_hor[i+1]*(width/2)+b_hor[i+1])-(pente_hor[i]*(width/2)+b_hor[i])) < seuil_hor):
            pente_hor[i] = (pente_hor[i] + pente_hor[i+1])/2.
            b_hor[i] = (b_hor[i] + b_hor[i+1])/2.
            index2del.append(i+1)
    for j in range(len(index2del)):
        del(pente_hor[index2del[j]-j])
        del(b_hor[index2del[j]-j])
    
    if (len(pente_hor) == 8):
        b_hor_sup = np.array(b_hor[1:])
        b_hor_inf = np.array(b_hor[:len(b_hor)-1])
        b_hor_deviation_mean = np.mean(b_hor_sup-b_hor_inf) 
        pente_hor = [pente_hor[0]] + pente_hor + [pente_hor[-1]]
        b_hor = [(b_hor[0]-b_hor_deviation_mean) if (b_hor[0]-b_hor_deviation_mean)>0 else 0] + b_hor \
            + [(b_hor[-1]+b_hor_deviation_mean) if (b_hor[-1]+b_hor_deviation_mean)<height else height]
    
    elif (len(pente_hor) == 9):
        b_sup = np.array(b_hor[1:])
        b_inf = np.array(b_hor[:len(b_hor)-1])
        b_deviation_mean = np.mean(b_sup-b_inf)

        y_line_before1 = int(b_hor[0]-b_deviation_mean) if (b_hor[0]-b_deviation_mean>0) else 0
        y_line_before2 = int(b_hor[0])
        x_line_before1 = 0
        x_line_before2 = width
        line_before = img[y_line_before1:y_line_before2, x_line_before1:x_line_before2]
        
        y_line_after1 = int(b_hor[-1])
        y_line_after2 = int(b_hor[-1]+b_deviation_mean) if (b_hor[-1]+b_deviation_mean<height) else height
        x_line_after1 = 0
        x_line_after2 = width
        line_after = img[y_line_after1:y_line_after2, x_line_after1:x_line_after2]

        mean_line_before = np.mean(line_before)
        mean_line_after = np.mean(line_after)
        
        if (mean_line_before > mean_line_after):
            pente_hor = [pente_hor[0]] + pente_hor
            b_hor = [y_line_before1] + b_hor
        else:
            pente_hor = pente_hor + [pente_hor[-1]]
            b_hor = b_hor + [y_line_after2]

    # To trace line on img
    
    for i in range(len(pente_hor)):
        cv2.line(img, (0, int(b_hor[i])), (height, int(pente_hor[i]*(height)+b_hor[i])), 0, 3)
    
    return (pente_hor, b_hor)
    
def clean_hough_lines_vert(img, lignes_vert, height, width):
    # Selection of lines
    # Sorted in function of the absciss of the vertical lines
    lignes_vert = sorted(lignes_vert, key=itemgetter(3))
    
    pente = np.array([e[1] for e in lignes_vert])
    ord_ligne_vert = [e[2] for e in lignes_vert]
    abs_ligne_vert = [e[3] for e in lignes_vert]
    
    # threshold to define if two lines belong to the same group
    seuil_vert = (abs_ligne_vert[-1] - abs_ligne_vert[0])/20.
    
    # Measure to see if one slop is not too influent or inconsistent compared to other
    slop_mean_global = np.mean(pente)
    slop_standard_dev = np.sqrt(np.var(pente))
    
    pente_moy = 0.
    b_moy = 0.
    b_pente_moy = 0.
    compt = 0.
    first_of_group = abs_ligne_vert[0]

    pente_vert = []
    b_vert = []
    b_pente_vert = []
    
    for i in range(1, len(abs_ligne_vert)):
        # For the last element
        if (i >= len(abs_ligne_vert)-1):
            # if the last element belongs to a group
            if (abs(abs_ligne_vert[i]-abs_ligne_vert[i-1]) <= seuil_vert and abs(abs_ligne_vert[i]-first_of_group) <= 2*seuil_vert):
                if ((pente[i] >= slop_mean_global-slop_standard_dev and pente[i] <= slop_mean_global+slop_standard_dev)):
                    pente_moy += pente[i]
                    b_moy += ord_ligne_vert[i]
                    b_pente_moy += abs_ligne_vert[i]
                    compt += 1
                
                if (compt>0):
                    pente_moy = pente_moy/compt if (pente_moy!=0 and abs(pente_moy)<height) else height
                    b_pente_moy = b_pente_moy/compt
                    b_moy = b_moy/compt if (pente_moy != height) else -b_pente_moy*pente_moy
                    
                    pente_vert.append(pente_moy)
                    b_vert.append(b_moy)
                    b_pente_vert.append(b_pente_moy)
            else:
                if (compt>0):
                    pente_moy = pente_moy/compt if (pente_moy!=0 and abs(pente_moy)<height) else height
                    b_pente_moy = b_pente_moy/compt
                    b_moy = b_moy/compt if (pente_moy != height) else -b_pente_moy*pente_moy
                    
                    pente_vert.append(pente_moy)
                    b_vert.append(b_moy)
                    b_pente_vert.append(b_pente_moy)
                # if the last element is alone
                if (pente[i] >= slop_mean_global-slop_standard_dev and pente[i] <= slop_mean_global+slop_standard_dev or pente[i]==height):
                    pente_vert.append(pente_vert[-1])
                    b_vert.append(-pente_vert[-1]*abs_ligne_vert[i])
                    b_pente_vert.append(abs_ligne_vert[i])
        else:
            if (abs(abs_ligne_vert[i]-abs_ligne_vert[i-1]) <= seuil_vert and abs(abs_ligne_vert[i]-first_of_group) <= 2*seuil_vert):
            #if the ith-element belongs to the group
                if ((pente[i-1] >= slop_mean_global-slop_standard_dev and pente[i-1] <= slop_mean_global+slop_standard_dev) or pente[i-1]==height):
                #if the slop is not inconsistent compared to other
                    if (pente[i-1] == height):
                    # if pente == height, it means that hough detects a line perfectly vertical, so it has to be included in that case
                        pente[i-1] = slop_mean_global if (pente_vert==[]) else pente_vert[-1]
                        ord_ligne_vert[i-1] = - pente[i-1] * abs_ligne_vert[i-1]
                   
                    pente_moy += pente[i-1]
                    b_moy += ord_ligne_vert[i-1]
                    b_pente_moy += abs_ligne_vert[i-1] 
                    compt +=1
                    
            else:
                if (compt>0):
                # if the group is completed
                    pente_moy += pente[i-1]
                    b_moy += ord_ligne_vert[i-1]
                    b_pente_moy += abs_ligne_vert[i-1] 
                    compt +=1

                    pente_moy = pente_moy/compt if (pente_moy!=0) else height
                    b_pente_moy = b_pente_moy/compt
                    b_moy = b_moy/compt if (pente_moy != height) else -b_pente_moy*pente_moy
                    pente_vert.append(pente_moy)
                    b_vert.append(b_moy)
                    b_pente_vert.append(b_pente_moy)
                else:
                # if a line is alone
                    if ((pente[i-1] >= slop_mean_global-slop_standard_dev and pente[i-1] <= slop_mean_global+slop_standard_dev) or pente[i-1] == height):
                        if (pente[i-1] == height):
                            pente[i-1] = slop_mean_global if (pente_vert==[]) else pente_vert[-1]
                        pente_vert.append(pente[i-1])
                        b_vert.append(ord_ligne_vert[i-1])
                        b_pente_vert.append(abs_ligne_vert[i-1])
                
                first_of_group = abs_ligne_vert[i] # to verify if the expand of a group is not too large
                pente_moy = 0
                b_moy = 0
                b_pente_moy = 0
                compt = 0.
    
    # Second filter of the values by comparing there absciss at the limit of the picture (height)
    index2del = []
    for i in range(0, len(pente_vert)-1):
        if (abs((height/2-b_vert[i+1])/pente_vert[i+1]-(height/2-b_vert[i])/pente_vert[i]) < seuil_vert):
            # if the absciss are close we make the mean and then delete one of the values
            pente_vert[i] = (pente_vert[i] + pente_vert[i+1])/2.
            b_vert[i] = (b_vert[i] + b_vert[i+1])/2.
            index2del.append(i+1)
    for j in range(len(index2del)):
        del(pente_vert[index2del[j]-j])
        del(b_vert[index2del[j]-j])
        del(b_pente_vert[index2del[j]-j])
    
    
    if (len(pente_vert) == 8):
        b_pente_sup = np.array(b_pente_vert[1:])
        b_pente_inf = np.array(b_pente_vert[:len(b_pente_vert)-1])
        b_pente_deviation_mean = np.mean(b_pente_sup-b_pente_inf)
        pente_vert = [pente_vert[0]] + pente_vert + [pente_vert[-1]]
        b_vert = [-(b_pente_vert[0]-b_pente_deviation_mean)*pente_vert[0] if (b_pente_vert[0]-b_pente_deviation_mean)>0 else 0] + b_vert \
            + [-(b_pente_vert[-1]+b_pente_deviation_mean)*pente_vert[-1] if (b_pente_vert[-1]+b_pente_deviation_mean)<width else -width*pente_vert[-1]]
    elif (len(pente_vert) == 9):
        b_pente_sup = np.array(b_pente_vert[1:])
        b_pente_inf = np.array(b_pente_vert[:len(b_pente_vert)-1])
        b_pente_deviation_mean = np.mean(b_pente_sup-b_pente_inf)

        x_line_before1 = int(b_pente_vert[0]-b_pente_deviation_mean) if (b_pente_vert[0]-b_pente_deviation_mean>0) else 0
        x_line_before2 = int(b_pente_vert[0])
        y_line_before1 = 0
        y_line_before2 = height
        line_before = img[y_line_before1:y_line_before2, x_line_before1:x_line_before2]
        
        x_line_after1 = int(b_pente_vert[-1])
        x_line_after2 = int(b_pente_vert[-1]+b_pente_deviation_mean) if (b_pente_vert[-1]+b_pente_deviation_mean<width) else width
        y_line_after1 = 0
        y_line_after2 = height
        line_after = img[y_line_after1:y_line_after2, x_line_after1:x_line_after2]

        mean_line_before = np.mean(line_before)
        mean_line_after = np.mean(line_after)
        
        if (mean_line_before > mean_line_after):
            pente_vert = [pente_vert[0]] + pente_vert
            b_vert = [-(x_line_before1)*pente_vert[0]] + b_vert
        else:
            pente_vert = pente_vert + [pente_vert[-1]]
            b_vert = b_vert + [-(x_line_after2)*pente_vert[-1] if (x_line_after2)<width else -width*pente_vert[-1]]

    # to trace the line on the img
    
    for i in range(0,len(pente_vert)):
        cv2.line(img, (int(-b_vert[i]/pente_vert[i]), 0), (int((width - b_vert[i])/pente_vert[i]), width), 0, 3)
    
    return (pente_vert, b_vert)
"""

def compute_intersection(pente_hor, b_hor, pente_vert, b_vert, h, w):
    intersect = []
    new_intersect = []
    len_list_vert = len(pente_vert)
    len_list_hor = len(pente_hor)
    void = 255*np.ones([h,w], dtype=np.uint8)

    """
    if (len_list_hor==8 and len_list_vert == 8):
        len_list = len_list_hor
        for j in range(0, len_list):
            for i in range(0, len_list):
                x = -(b_vert[i]-b_hor[j])/float(pente_vert[i]-pente_hor[j])
                y = pente_vert[i] * x + b_vert[i]
                
                if (i == 0):
                    intersect.append((0, int(y)))
                    cv2.circle(img, (0, int(y)), 3, 0, 3)

                intersect.append((int(x), int(y)))
                cv2.circle(img, (int(x), int(y)), 3, 0, 3)
                
                if (i == len_list-1):
                    intersect.append((h, int(y)))
                    cv2.circle(img, (h, int(y)), 3, 0, 3)
        new_intersect.append([(intersect[i][0], 0) for i in range(10)])
        for i in range(9):
            new_intersect.append(intersect[i*10:i*10+9])
        new_intersect.append([(intersect[i][0],h) for i in range(10)])

                
    else:
    """
    for i in range(0, len_list_vert):
        for j in range(0, len_list_hor):
            x = -(b_vert[i]-b_hor[j])/float(pente_vert[i]-pente_hor[j])
            y = pente_vert[i] * x + b_vert[i]
            intersect.append((int(x), int(y)))
            cv2.circle(void, (int(x), int(y)), 3, 0, 2)
    
    number_inters, labels_inters, stats_inters, centroid_inters = cv2.connectedComponentsWithStats(inverse_image(void))
    area_labels = [e[cv2.CC_STAT_AREA] for e in stats_inters]
    index_max_area = area_labels.index(max(area_labels))

    centroid_inters_list = list(centroid_inters//1)
    del(centroid_inters_list[index_max_area])

    intersection = []
    for i in range(0,10):
        intersection.append(sorted(centroid_inters_list[i*10:(i+1)*10], key=itemgetter(0)))
    return intersection

def detect_white_case(intersection, img, height, width):
    case_array = np.zeros([9,9], dtype=int)
    mean_array = np.zeros([9,9])
    dev_array = np.zeros([9,9])


    crop_coord = []
    for i in range(0,9):
        for j in range(0,9):
            
            #print(np.array([intersection[i][j], intersection[i][j+1], intersection[i+1][j], intersection[i+1][j+1]]))
            cropped1 = crop_non_rect_case(img, np.array([intersection[i][j], intersection[i][j+1], intersection[i+1][j], intersection[i+1][j+1]]).reshape((-1,1,2)).astype(np.int32), height, width)
            #cropped1 = crop_non_rect_case(img, np.array([intersection[i][j], intersection[i][j+1], intersection[i+1][j], intersection[i+1][j+1]]), height, width)
            
            # Connex component
            num_label, label_im, stats_label, centroid_label = cv2.connectedComponentsWithStats(cropped1)

            area = [stats_label[k][4] for k in range(num_label)]
            max_index = area.index(max(area))
            
            [j_white_max,i_white_max] = find_color_max_label(label_im, max_index)
            
            while(cropped1[j_white_max,i_white_max] != 255):
                area[max_index]=0
                max_index = area.index(max(area))
                [j_white_max, i_white_max] = find_color_max_label(label_im, max_index)
            
            x1 = (stats_label[max_index][cv2.CC_STAT_LEFT])//1
            x2 = (x1 + stats_label[max_index][cv2.CC_STAT_WIDTH])//1
            y1 = (stats_label[max_index][cv2.CC_STAT_TOP])//1
            y2 = (y1 + stats_label[max_index][cv2.CC_STAT_HEIGHT])//1
            
            cropped2 = cropped1[y1:y2, x1:x2]
            #cv2.imshow("crop"+str(i)+","+str(j),cropped2)
            intersect = intersection[i][j]
            crop_coord.append((int(intersect[0]+x1), int(intersect[0]+x2), int(intersect[1]+y1), int(intersect[1]+y2)))
            mean_array[i,j] = np.mean(cropped2)
            dev_array[i,j] = np.sqrt(np.var(cropped2))
    
    mean_case = np.mean(mean_array)
    mean_dev = np.sqrt(np.var(dev_array))
    
    mean_array_inf = mean_array<=mean_case
    mean_dev_sup = dev_array>=mean_dev

    case_array[mean_array_inf* mean_dev_sup == True] = 1
    print(case_array)
    return case_array, crop_coord

def crop_non_rect_case(img, points, h, w):
    mask = np.zeros([h,w], dtype=np.uint8)
    #method 1 smooth region
    cv2.drawContours(mask, [points], -1, 255, -1, cv2.LINE_AA)
 
    #method 2 not so smooth region
    #cv2.fillPoly(mask, [points], (255))
 
    res = cv2.bitwise_and(img, img, mask)
    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

    return(cropped)

def clean_background(img, h, w):
    #First label to clear background
    number, labels, stats, centroid = cv2.connectedComponentsWithStats(img)
    
    label_hue = np.uint8(10+179*labels/np.max(labels))

    index2del = []
    area = [stats[i][4] for i in range(number)]
    max_index = area.index(max(area))
    index2del = []

    i=0
    while (area[max_index]>=(h/2)*(w/2)):
        area[max_index]=0
        [i_max,j_max] = find_color_max_label(labels, max_index)
        if (img[i_max,j_max]==0):
            index2del.append(max_index)
        max_index = area.index(max(area))
    
    labeled_img = label_hue
    #cv2.imshow("label_hue", label_hue)
    for e in index2del:
        labeled_img[labels==e] = 0
    labeled_img[labeled_img!=0]=255
    #cv2.imshow("labeled_img",labeled_img)
    # Closing for isolating grid
    #kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    #closing1 = cv2.morphologyEx(labeled_img, cv2.MORPH_CLOSE, kernel2).astype(np.uint8)
    
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    closing = cv2.morphologyEx(labeled_img, cv2.MORPH_ERODE, kernel2).astype(np.uint8)
    # Second label for the mask of the image
    number_mask, labels_mask, stats_mask, centroid_mask = cv2.connectedComponentsWithStats(closing)
    area_mask = [stats_mask[i][4] for i in range(number_mask)]
    max_index_mask = area_mask.index(max(area_mask))
    area_mask[max_index_mask] = 0
    max_index_mask2 = area_mask.index(max(area_mask))
    
    mask = np.zeros([h,w])
    i=0
    j=0
    while(labels_mask[i,j] != max_index_mask):
        if (i == w-1):
            i = 0
            j=j+1
        i=i+1

    if (closing[i,j] == 255):
        mask[labels_mask == max_index_mask] = 255
        mask[labels_mask != max_index_mask] = 0
    else:
        mask[labels_mask == max_index_mask2] = 255
        mask[labels_mask != max_index_mask2] = 0
    #cv2.imshow("mask",mask)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    closing2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2).astype(np.uint8)
    #cv2.imshow("clean bg", closing2)
    return closing2

def extract_and_resize_number(img, case_array, crop_coord, resize_dim):
    list_of_number = []
    final_number = np.zeros(resize_dim, dtype=np.uint8)
    for i in range(0,9):
        for j in range(0,9):
            if (case_array[i,j] == 1):
                x1 = crop_coord[9*i+j][0]
                x2 = crop_coord[9*i+j][1]
                y1 = crop_coord[9*i+j][2]
                y2 = crop_coord[9*i+j][3]

                case = img[y1:y2, x1:x2]
                dim_case = np.shape(case)
                h_case = dim_case[0]
                w_case = dim_case[1]

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
                closing = cv2.morphologyEx(case, cv2.MORPH_CLOSE, kernel)

                number_case, labels_case, stats_case, centroid_case = cv2.connectedComponentsWithStats(inverse_image(closing))

                area = [stats_case[k][4] for k in range(number_case)]
                max_index = area.index(max(area))
                [i_black_max,j_black_max] = find_color_max_label(labels_case, max_index)
                
                if (number_case>1):
                    while(closing[i_black_max,j_black_max] != 0):
                        area[max_index]=0
                        max_index = area.index(max(area))
                        [i_black_max, j_black_max] = find_color_max_label(labels_case, max_index)
                
                x_case1 = int(stats_case[max_index][0])
                x_case2 = x_case1 + int(stats_case[max_index][2])
                y_case1 = int(stats_case[max_index][1])
                y_case2 = y_case1 + int(stats_case[max_index][3])
                case_number = closing[y_case1:y_case2, x_case1:x_case2]
                cv2.imshow("number"+str(i)+str(j), case_number)
                if (h_case/4 < centroid_case[max_index][1] < (3*h_case)/4 and \
                    w_case/4 < centroid_case[max_index][0] < (3*w_case)/4):
                    case_number = closing[y_case1:y_case2, x_case1:x_case2]
                    final_number = cv2.resize(case_number, resize_dim, interpolation=cv2.INTER_NEAREST)
                    if (np.mean(final_number)<230 and final_number.shape[0] == 28 and final_number.shape[1] == 28):
                        cv2.imshow("number"+str(i)+str(j), final_number)
                        list_of_number.append([final_number, (i,j)])
                
                    
    return list_of_number


def find_color_max_label(label_im, max_index):
    i = 0
    j = 0
    dim = np.shape(label_im)
    h = dim[0]
    w = dim[1]
    while (label_im[i,j] != max_index or (j == w-1 and i == h-1)):
        if (j == w-1):
            j=0
            i=i+1
        j=j+1
    return [i,j]
    

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(255*labels/np.max(labels))
    #kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    #closing = cv2.morphologyEx(label_hue, cv2.MORPH_CLOSE, kernel2).astype(np.uint8)
    #cv2.imshow("lab_hue", closing)
    blank_ch = 255*np.ones_like(label_hue)
    
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    #cv2.imshow("blank", labeled_img)
    # cvt to BGR for display
    #labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue<= 1] = 0
    labeled_img[label_hue> 1] = 255
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
    closing = cv2.morphologyEx(labeled_img, cv2.MORPH_CLOSE, kernel2).astype(np.uint8)





def process(filename):
    sudoku = cv2.imread(filename, 0)

    ######## Test if img is void ##########
    if np.shape(sudoku) == ():
        print ("Pas d'image detectee!!")
    else:
        dim = np.shape(sudoku)
        height = dim[0] if (dim[0]<=480) else 480 
        width  = dim[1] if (dim[1]<=640) else 640 
        dim = (width, height)
        

        ####### Resize img #########
        sudoku1 = cv2.resize(sudoku, dim, interpolation = cv2.INTER_LINEAR)
        #else:
        #    sudoku1 = np.copy(sudoku)

        
        ####### Normalize Histogramm #######
        sudoku2 = normalize_image(sudoku1, height, width, 40, 200).astype(np.uint8)
        #res = np.hstack((sudoku1,sudoku2)) 
        #cv2.imshow('norm_histo',sudoku2)
        
        ####### Balance Histogramm #######
        sudoku3 = cv2.equalizeHist(sudoku2)
        #cv2.imshow('eq_histo',sudoku3)

        ####### Binarisation using Sauvola's Method #######
        binarize = sauvola_binarise(sudoku1, height, width, 4, 0.05)
        #cv2.imshow('binarize',binarize)
        
        
        
        ####### Erosion #######
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        inverse_erosion = cv2.morphologyEx(inverse_image(binarize), cv2.MORPH_DILATE, kernel).astype(np.uint8)
        erosion = inverse_image(inverse_erosion)
        #cv2.imshow("erosion", erosion)

        mask = clean_background(inverse_erosion, height, width).astype(np.uint8)
        #
        # res = cv2.bitwise_and(erosion, mask)
        #cv2.imshow("res", res)
        ####### Closing #######
        """
        kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        closing = cv2.morphologyEx(erosion, cv2.MORPH_CLOSE, kernel2).astype(np.uint8)
        cv2.imshow("closing", closing)

        ####### Laplacian #######
        Laplacian = cv2.Laplacian(inverse_image(res),cv2.CV_8U,ksize=3)
        #cv2.imshow("Laplacian", Laplacian)
        """
        ####### Hough Line #######
        grid = mask
        lignes_hor = []
        lignes_vert = []
        lignes_vert, lignes_hor, number_of_lines_vert, number_of_lines_hor, grid = Houghline(mask, height, width, height//2)
        
        pente_vert = [e[1] for e in lignes_vert]
        b_vert = [e[2] for e in lignes_vert]
        pente_hor = [e[1] for e in lignes_hor]
        b_hor = [e[2] for e in lignes_hor]
        
        void = 255*np.ones([height,width], dtype=np.uint8)
        intersection =[]
        intersection = compute_intersection(pente_hor, b_hor, pente_vert, b_vert, height, width)

        case_array = np.zeros([9,9], dtype=int)
        crop_coord = []
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
        inverse_erosion1 = cv2.morphologyEx(inverse_image(binarize), cv2.MORPH_DILATE, kernel).astype(np.uint8)
        erosion1 = inverse_image(inverse_erosion1)
        
        case_array, crop_coord = detect_white_case(intersection, erosion, height, width)
        cv2.imshow("bin", erosion1)
        final_list_of_numbers = extract_and_resize_number(erosion1, case_array, crop_coord, (28,28))
        #final_list_of_numbers = extract_and_resize_number(binarize.astype(np.uint8), case_array, crop_coord, (20,20))
        print(len(final_list_of_numbers))
        #cv2.imshow("number", final_list_of_numbers[1][0])
        
        
        #
        ####### Harris detection #######
        """
        dst = cv2.cornerHarris(inverse_image(mask),4,5,0.06)
        #result is dilated for marking the corners, not important
        #dst = cv2.dilate(dst,None)
        
        # Threshold for an optimal value, it may vary depending on the image.
        void[dst>0.05*dst.max()]=0

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        close = cv2.morphologyEx(void, cv2.MORPH_CLOSE, kernel).astype(np.uint8)
        cv2.imshow("close", close)
        #cornerHarris_demo(grid, sudoku1, 200)
        """

        """
        # Sort Hough lines
        pente_vert = []
        b_vert = []
        pente_hor = []
        b_hor = []
        
        if (number_of_lines_hor>=8 and number_of_lines_vert>=8):
            pente_hor, b_hor = clean_hough_lines_hor(sudoku1, lignes_hor, height, width)
            pente_vert, b_vert = clean_hough_lines_vert(sudoku1, lignes_vert, height, width)
        
            # Compute intersection for cropping
            intersection = []
            intersection = compute_intersection(sudoku1, pente_hor, b_hor, pente_vert, b_vert, height, width)
            
            if (len(intersection) >= 8):
                # Detect white case and isolate cases with number
                case_array = np.zeros([9,9], dtype=int)
                crop_coord = []
                case_array, crop_coord = detect_white_case(intersection, erosion, height, width)
                #img = erosions
                
                final_list_of_numbers = extract_and_resize_number(erosion, case_array, crop_coord, (20,20))

                print(len(final_list_of_numbers))
                cv2.imshow("number", final_list_of_numbers[10][0])
            else:
                print("Not enough intersections detected")
            
        else:
            print("Not enough line detected")
        """
        cv2.imshow("original", sudoku1)
        

        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    lg.basicConfig(level=lg.DEBUG)
    process(str(sys.argv[1]))

if __name__ == "__main__":
    main()

