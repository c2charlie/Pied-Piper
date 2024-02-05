# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 10:30:50 2024

@author: Lili Zheng
"""

import xlwings as xw

# Path to the original Excel file
file_path = r'E:\Instance\04\新能源汽车备案信息.xlsx'
# Path to save the modified Excel file
save_path = r'E:\Instance\04\新能源汽车备案信息（自适应调整）.xlsx'

# Create an Excel application instance
app = xw.App(visible=False, add_book=False)

try:
    # Open the original Excel workbook
    workbook = app.books.open(file_path)

    # Loop through each worksheet in the workbook
    for worksheet in workbook.sheets:
        # Autofit columns and rows in each worksheet
        worksheet.autofit()

    # Save the modified workbook with a new name
    workbook.save(save_path)
    print(f"File saved to: {save_path}")

finally:
    # Close the workbook and quit the Excel application
    workbook.close()
    app.quit()
