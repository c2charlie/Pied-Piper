# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:26:21 2024

@author: Lili Zheng
"""

# pathlib module offers classes representing filesystem paths with semantics appropriate for different operating systems
# Path classes are divided between pure paths, and concrete paths, which inherit from pure paths but also provide I/O operations
from pathlib import Path

# xlwings is a Python library that makes it easy to call Python from Excel and vice versa
import xlwings as xw

dst_folder = Path('C:\\Users\\Lili Zheng\\Desktop')
# 
dst_folder.mkdir(parents = True, exist_ok = True)

app = xw.App(visible = True, add_book = False)

for i in range(1,21):
    workbook = app.books.add()
    file_path = dst_folder / f'分公司{i}.xlsx'
    workbook.save(file_path)
    workbook.close()
app.quit()