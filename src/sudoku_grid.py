#! /usr/bin/env python3
# coding: utf-8

import numpy as np
#import CNN_model

class Sudoku_grid():
    """
    Class of complete grid
    """
    def __init__(self, array=None):
        """
        Initialize with void grid full of zeros
        """
        dim = np.shape(array)
        if (dim!=()):
            if (dim[0]==9 and dim[1]==9):
                self.grid = array/1
        else:
            self.grid = np.zeros([9,9], dtype=int)

        self.zero = 0
        self.cases = []
        self.blocks = Sudoku_blocks(self.grid)
        self.lines = Sudoku_lines(self.grid)
        self.cols = Sudoku_cols(self.grid)
        self.init_list()
        self.number_cases = len(self.cases)

    
    def init_list(self):
        for i in range(9):
            for j in range(9):
                self.cases.append(Sudoku_case((i,j), self))
                if (self.cases[-1].value == 0):
                    self.zero +=1
            
    def get_ij(self, index):
        return (self.cases[9*index[0]+index[1]])
    
    def get_left_case(self, case):
        i = case.X
        j = case.Y
        if (i!=0):
            self.get_ij((i-1, j))
        elif (i==0):
            self.get_ij((i,j))
    
    def get_up_case(self, case):
        i = case.X
        j = case.Y
        if (j!=0):
            self.get_ij((i, j-1))
        elif (j==0):
            self.get_ij((i,j))
    
    def get_temp_line(self, case):
        return (self.get_left_case(case).temp_line)
    
    def get_temp_col(self, case):
        return (self.get_up_case(case).temp_col)

    def get_temp_block(self, case):
        i = case.X
        j = case.Y
        if ((i-1)>0 and ((i-1)//3 == i//3)):
            return (self.get_left_case(case).temp_block)
        elif ((j-1)>0 and ((j-1)//3 == j//3)):
            return (self.get_up_case(case).temp_block)
        else:
            return (self.case.temp_block)

    def show(self):
        """
        Displays the sudoku grid
        """
        print(self.grid)



class Sudoku_block():
    def __init__(self, array):
        dim = np.shape(array)

        self.list_of_number = []
        self.list_of_possible = [i for i in range(1,10)]
        self.block_array = array
        
        if (dim[0]==3 and dim[1]==3):
            for i in range(dim[0]):
                for j in range(dim[1]):
                    if (self.check_value_in_block(array[i,j]) == False):
                        self.add_value_block(array[i,j])
                        self.remove_possibility(array[i,j])
                    else:
                        print("Error in initial sudoku grid")
        else:
            print("Error in array block")
            
    def check_value_in_block(self, value):
        if (value in self.list_of_number):
            return True
        return False
    
    def add_possibility(self, value):
        if (1<= value <=9):
            self.list_of_possible.append(value)

    def remove_possibility(self, value):
        if (1<= value <=9):
            self.list_of_possible.remove(value)
    
    def add_value_block(self, value):
        if (1<= value <=9):
            self.list_of_number.append(value)
    
    def remove_value_block(self, value):
        self.list_of_number.remove(value)
    
    def show_list(self):
        print(self.list_of_number)
    
    def show(self):
        print(self.block_array)


class Sudoku_blocks():
    def __init__(self, array=None):
        if (array.shape!=()):
            self.block0 = Sudoku_block(array[0:3,0:3])
            self.block1 = Sudoku_block(array[0:3,3:6])
            self.block2 = Sudoku_block(array[0:3,6:9])
            self.block3 = Sudoku_block(array[3:6,0:3])
            self.block4 = Sudoku_block(array[3:6,3:6])
            self.block5 = Sudoku_block(array[3:6,6:9])
            self.block6 = Sudoku_block(array[6:9,0:3])
            self.block7 = Sudoku_block(array[6:9,3:6])
            self.block8 = Sudoku_block(array[6:9,6:9])
        
        else:
            zero = np.zeros([3,3])
            self.block0 = Sudoku_block(zero)
            self.block1 = Sudoku_block(zero)
            self.block2 = Sudoku_block(zero)
            self.block3 = Sudoku_block(zero)
            self.block4 = Sudoku_block(zero)
            self.block5 = Sudoku_block(zero)
            self.block6 = Sudoku_block(zero)
            self.block7 = Sudoku_block(zero)
            self.block8 = Sudoku_block(zero)
        
        self.switcher = {
            0 : self.block0,
            1 : self.block1, 
            2 : self.block2, 
            3 : self.block3, 
            4 : self.block4, 
            5 : self.block5, 
            6 : self.block6, 
            7 : self.block7, 
            8 : self.block8 
        }
    
    def get_blocki(self,index):
        a = index[0] // 3
        b = index[1] // 3
        return self.switcher.get(3*a+b)  #(3*a+b, "wrong number")

    def check_value_in_blocki(self, i, value):
        self.switcher.get(i, "wrong number").check_value_in_block(value)
    
    def remove_value_blocki(self, i, value):
        self.switcher.get(i, "wrong number").remove_value_block(value)

    def add_value_blocki(self, i, value):
        self.switcher.get(i, "wrong number").add_value_block(value)
    
    def check_and_add_value_blocki(self, value, i):
        if self.check_value_in_block_i(i, value)==False:
            self.add_value_blocki(i, value)
            return True
        return False
    
    def show_all(self):
        for i in range(1,10):
            print("Block"+str(i)+ " :\n")
            blocki = self.switcher[i].show()
            print("\n")
        

class Sudoku_line():
    def __init__(self, array=None):
        dim = np.shape(array)
        self.list_of_number = []
        self.list_of_possible = [i for i in range(1,10)]
        self.line_array = array

        if (dim!=()):
            if (dim[0]==1 and dim[1]==9):
                for j in range(9):
                    if (self.check_value_in_line(array[0,j]) == False):
                        self.add_value_line(array[0,j])
                        self.remove_possibility(array[0,j])
                    else:
                        print("Error in initial sudoku grid")
            else:
                print("Error in array line")
            
    def check_value_in_line(self, value):
        if (value in self.list_of_number):
            return True
        return False

    def add_possibility(self, value):
        if (1<= value <=9):
            self.list_of_possible.append(value)

    def remove_possibility(self, value):
        if (1<= value <=9):
            self.list_of_possible.remove(value)
    
    def add_value_line(self, value):
        if (1<= value <=9):
            self.list_of_number.append(value)
    
    def remove_value_line(self, value):
        self.list_of_number.remove(value)
    
    def show_list(self):
        print(self.list_of_number)

    def show(self):
        print(self.line_array)

class Sudoku_lines():
    def __init__(self, array=None):
        if (array.shape!=()):
            self.line0 = Sudoku_line(array[0:1,0:9])
            self.line1 = Sudoku_line(array[1:2,0:9])
            self.line2 = Sudoku_line(array[2:3,0:9])
            self.line3 = Sudoku_line(array[3:4,0:9])
            self.line4 = Sudoku_line(array[4:5,0:9])
            self.line5 = Sudoku_line(array[5:6,0:9])
            self.line6 = Sudoku_line(array[6:7,0:9])
            self.line7 = Sudoku_line(array[7:8,0:9])
            self.line8 = Sudoku_line(array[8:9,0:9])
        
        else:
            self.line0 = Sudoku_line()
            self.line1 = Sudoku_line()
            self.line2 = Sudoku_line()
            self.line3 = Sudoku_line()
            self.line4 = Sudoku_line()
            self.line5 = Sudoku_line()
            self.line6 = Sudoku_line()
            self.line7 = Sudoku_line()
            self.line8 = Sudoku_line()
            
        self.switcher = {
            0:self.line0,
            1:self.line1, 
            2:self.line2, 
            3:self.line3, 
            4:self.line4, 
            5:self.line5, 
            6:self.line6, 
            7:self.line7, 
            8:self.line8 
        }

    def get_linei(self,i):
        return self.switcher.get(i) #(i, "wrong number")

    def check_value_in_linei(self, i, value):
        self.switcher.get(i, "wrong number").check_value_in_line(value)
    
    def remove_value_linei(self, i, value):
        self.switcher.get(i, "wrong number").remove_value_line(value)

    def add_value_linei(self, i, value):
        self.switcher.get(i, "wrong number").add_value_line(value)
    
    def check_and_add_value_linei(self, value, i):
        if self.check_value_in_linei(i, value)==False:
            self.add_value_linei(i, value)
            return True
        return False
    
    def show(self):
        for i in range(1,10):
            print("Line"+str(i)+ " :\n")
            self.switcher.get(i, "wrong number").show()
            print("\n")


class Sudoku_col():
    def __init__(self, array=None):
        dim = np.shape(array)

        self.list_of_number = []
        self.list_of_possible = [i for i in range(1,10)]
        self.col_array = array

        if (dim!=()):
            if (dim[0]==9 and dim[1]==1):
                for j in range(9):
                    if (self.check_value_in_col(array[j,0]) == False):
                        self.add_value_col(array[j,0])
                        self.remove_possibility(array[j,0])
                    else:
                        print("Error in initial sudoku grid")
            else:
                print("Error in array col")
            
    def check_value_in_col(self, value):
        if (value in self.list_of_number):
            return True
        return False
    
    def add_possibility(self, value):
        if (1<= value <=9):
            self.list_of_possible.append(value)

    def remove_possibility(self, value):
        if (1<= value <=9):
            self.list_of_possible.remove(value)

    def add_value_col(self, value):
        if (1<= value <=9):
            self.list_of_number.append(value)
    
    def remove_value_col(self, value):
        self.list_of_number.remove(value)
    
    def show_list(self):
        print(self.list_of_number)

    def show(self):
        print(self.col_array)

class Sudoku_cols():
    def __init__(self, array=None):
        if (array.shape!=()):
            self.col0 = Sudoku_col(array[0:9,0:1])
            self.col1 = Sudoku_col(array[0:9,1:2])
            self.col2 = Sudoku_col(array[0:9,2:3])
            self.col3 = Sudoku_col(array[0:9,3:4])
            self.col4 = Sudoku_col(array[0:9,4:5])
            self.col5 = Sudoku_col(array[0:9,5:6])
            self.col6 = Sudoku_col(array[0:9,6:7])
            self.col7 = Sudoku_col(array[0:9,7:8])
            self.col8 = Sudoku_col(array[0:9,8:9])
        
        else:
            self.col0 = Sudoku_col()
            self.col1 = Sudoku_col()
            self.col2 = Sudoku_col()
            self.col3 = Sudoku_col()
            self.col4 = Sudoku_col()
            self.col5 = Sudoku_col()
            self.col6 = Sudoku_col()
            self.col7 = Sudoku_col()
            self.col8 = Sudoku_col()
            
        self.switcher = {
            0:self.col0,
            1:self.col1, 
            2:self.col2, 
            3:self.col3, 
            4:self.col4, 
            5:self.col5, 
            6:self.col6, 
            7:self.col7, 
            8:self.col8
        }
    def get_coli(self,i):
        return self.switcher.get(i) #(i, "wrong number")

    def check_value_in_coli(self, i, value):
        self.switcher.get(i, "wrong number").check_value_in_col(value)
    
    def remove_value_coli(self, i, value):
        self.switcher.get(i, "wrong number").remove_value_col(value)

    def add_value_coli(self, i, value):
        self.switcher.get(i, "wrong number").add_value_col(value)
    
    def check_and_add_value_coli(self, value, i):
        if self.check_value_in_coli(i, value)==False:
            self.add_value_coli(i, value)
            return True
        return False
    
    def show(self):
        for i in range(1,10):
            print("Column"+str(i)+ " :\n")
            self.switcher.get(i, "wrong number").show()
            print("\n")

class Sudoku_case():
    def __init__(self, index, sudoku_grid):
        self.position = index
        self.line = sudoku_grid.lines.get_linei(index[0])
        self.temp_line = self.line
        
        self.col = sudoku_grid.cols.get_coli(index[1])
        self.temp_col = self.col

        self.block = sudoku_grid.blocks.get_blocki(index)
        self.temp_block = self.block
        
        self.value = sudoku_grid.grid[index]
        self.possible = self.compute_possible() if (self.value==0) else [self.value]
        
        self.temp_value = self.possible[0]
        self.temp_possible = self.possible

         

    def compute_possible(self):
        possible = []
        for i in range(1,10):
            if ((i not in self.line.list_of_number) and (i not in self.col.list_of_number) and (i not in self.block.list_of_number)):
                possible.append(i)
        return possible
    
    def remove_possibility(self, possible, value):
        possible.remove(value)

    def reset_possible(self):
        self.temp_possible = self.possible
        self.temp_line = self.line
        self.temp_col = self.col
        self.temp_block = self.block

    def update_temp_value(self):
        self.temp_value = self.temp_possible[0]

    def update_temp_line(self, value):
        self.temp_line.add_value_line(value)

    def update_temp_col(self, value):
        self.temp_col.add_value_col(value)
    
    def update_temp_block(self ,value):
        self.temp_block.add_value_block(value)

    def update_possible(self, temp_line, temp_col, temp_block):
        new_temp_possible = []
        for i in self.temp_possible:
            if ((i not in temp_line.list_of_number) and (i not in temp_col.list_of_number) and (i not in temp_block.list_of_number)):
                new_temp_possible.append(i)
        self.temp_possible = new_temp_possible
        return self.temp_possible
    
    def change_possible_value(self):
        if (self.temp_value == self.temp_possible[0]):
            del(self.temp_possible[0])
        if (self.temp_possible!=[]):
            self.update_temp_value()
            self.update_temp_line(self.temp_value)
            self.update_temp_col(self.temp_value)
            self.update_temp_block(self.temp_value)
            return True
        else:
            self.reset_possible()
            return False
    
    @property
    def X(self):
        return self.position[0]
    
    @property
    def Y(self):
        return self.position[1]



def solve_sudoku(sudoku):
    sudoku_grid = Sudoku_grid(sudoku)
    i = 0
    j = 0
    while(sudoku_grid.zero != 0 or i>9 or j>9):
        current_case = sudoku_grid.get_ij((i,j))
        temp_line = sudoku_grid.get_temp_line(current_case)
        temp_col = sudoku_grid.get_temp_col(current_case)
        temp_block = sudoku_grid.get_temp_block(current_case)
        current_case.update_possible(temp_line, temp_col, temp_block)
        if (current_case.change_possible_value() == True):
            sudoku_grid.zero -=1
            if (j == 9):
                j=0
                i+=1
            j+=1
        else:
            sudoku_grid.zero +=1
            if (j == 0):
                j=9
                i-=1
            j-=1
            

puzzle = np.array(
    [[5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]])

solve_sudoku(puzzle)
#sudoku_grid.show()