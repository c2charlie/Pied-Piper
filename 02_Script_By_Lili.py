# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:55:38 2024

@author: Lili Zheng
"""

import xlwings as xw
app = xw.App(visible=False, add_book=False)
workbook = app.books.open('中证100指数.xlsx')
worksheet = workbook.sheets
for i in worksheet:
       i.autofit()
workbook.save()
workbook.close()
app.quit()