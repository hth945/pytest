# import tkinter as tk
# from tkinter import *
# top = Tk() #导入tk模块
# from PIL import Image, ImageTk
# image = Image.open("test2.png")
# photo = ImageTk.PhotoImage(image)
# label = tk.Label(top)
# label.pack()
# label.configure(image = photo )
# top.mainloop()


# import tkinter as tk
# from tkinter import *
# top = Tk() #导入tk模块
# from PIL import Image, ImageTk
# image = Image.open("test2.png")
# photo = ImageTk.PhotoImage(image)
# label = tk.Label(top)
# label.pack()
# label.configure(image = photo )
# top.mainloop()





import tkinter as tk
root = tk.Tk()
root.attributes('-alpha', 0.0) #For icon
#root.lower()
root.iconify()
window = tk.Toplevel(root)
window.geometry("1000x1000") #Whatever size
window.overrideredirect(1) #Remove border
#window.attributes('-topmost', 1)
#Whatever buttons, etc
# close = tk.Button(window, text = "Close Window", command = lambda: root.destroy())
# close.pack(fill = tk.BOTH, expand = 1)
from PIL import Image, ImageTk
image = Image.open("test.png")
photo = ImageTk.PhotoImage(image)
label = tk.Label(window)
label.pack()
label.configure(image=photo)
window.mainloop()




# import tkinter as tk
# root = tk.Tk()
# root.attributes('-alpha', 0.0) #For icon
# #root.lower()
# root.iconify()
# window = tk.Toplevel(root)
# window.geometry("3840x2160") #Whatever size
# window.overrideredirect(1) #Remove border
# #window.attributes('-topmost', 1)
# #Whatever buttons, etc
# close = tk.Button(window, text = "Close Window", command = lambda: root.destroy())
# close.pack(fill = tk.BOTH, expand = 1)
# window.mainloop()




