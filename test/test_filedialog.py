
from tkinter import messagebox, filedialog

path = filedialog.askopenfile(mode='r',
                              filetypes=[('Image Files', '*.png'), ('Image Files', '*.tif'),
                                         ('DataContainer', '*.zdc')])

print("test")#
