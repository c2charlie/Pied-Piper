# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 10:46:32 2024

@author: Lili Zheng
"""

"""
Manual input in the console

Create a program that allows the user to input the length and width of the rectangular
and calculate the perimeter & area

"""

length = float(input("请输入矩形的长度："))
width  = float(input("请输入举行的宽度:"))

area = length * width
perimeter = 2 * (length + width)

print("矩形的面积为： ", area)
print("矩形的周长为： ", perimeter)

"""
Read from local file

How to obtain the text content and load it up to python as one of the standard data structure

"""

# txt file address on PC
# input the absolute path / absolute path
score_file_path = "C:\\Users\\Lili Zheng\\Desktop\\score_data.txt"

# open the file as read only
# close the file automatically
with open(score_file_path, "r") as f:
    # use readlines to capture the data
    data = f.readlines()

# define the final output variable type from score_data as dictionary
scores_data = {}
# process data
for line in data:
    
    #1 remove the space, enter
    line = line.strip(" ").strip()
    # remove the empty line
    if not line:
        continue
    
    #2 split each line by ","
    score = line.split(",")
    
    #3 
    scores_data[(score[1], score[0])] = score[2:]

print(scores_data)