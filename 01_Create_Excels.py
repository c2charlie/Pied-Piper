# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:26:21 2024

@author: Lili Zheng
"""

# pathlib模块提供表示文件系统路径的类，其语义适用于不同的操作系统
# Path为从纯路径继承而来但提供I/O操作的具体路径
from pathlib import Path
import xlwings as xw

dst_folder = Path('C:\\Users\\Lili Zheng\\Desktop')
dst_folder.mkdir(parents = True, exist_ok = True)

app = xw.App(visible = True, add_book = False)

for i in range(1,21):
    workbook = app.books.add()
    file_path = dst_folder / f'分公司{i}.xlsx'
    workbook.save(file_path)
    workbook.close()
app.quit()