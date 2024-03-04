# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 17:07:15 2024

@author: Lili Zheng
"""

"""

ChatGPT Instruction：

I have an Excel file named "办公用品采购表.xlsx", in this file there are several sheets.
Please create a Python script, which will save each sheet as indivisual excel file.

"""

import xlwings as xw
from pathlib import Path

# open the original workbook
wb = xw.Book('办公用品采购表.xlsx')

# capture the path of the original file
folder_path = Path(wb.fullname).parent

# loop through all the sheets and save as individual files
for sheet in wb.sheets:
    # new workbook named: original sheet name + ".xlsx"
    new_file_name = sheet.name + '.xlsx'
    # save the new workbook to the original path
    new_file_name = foler_path / new_file_name
    
    sheet.api.Copy(Before = None)
    new_wb = xw.Book.active()
    new_wb.save(str(new_file_apth)))
    new_wb.close()
wb.close()