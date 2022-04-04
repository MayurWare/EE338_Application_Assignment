
from tkinter import filedialog
from IPython.display import display, Math, Latex

import numpy as np
import matplotlib.pyplot as plt

import cv2
from scipy import signal
from scipy import misc

import os
import tkinter as ttk
from tkinter import *
import tkinter.messagebox
from PIL import Image, ImageTk,ImageGrab
from tkinter.filedialog import askopenfilename
image=None 
image_stack = []
image_stack_index = 0
file = None


def undoable():
        #Remembering the order in which operations were performed
        global image,image_stack,image_stack_index,file
        #If length==0 no image left in the stack
        if len(image_stack) > 1 and image_stack_index > 0:
            return True
        else:
            return False

def undo():
    #Remembering the order in which operations were performed
    global image,image_stack,image_stack_index,file
    if undoable():
        #Pointer is decremented for every Undo
        image_stack_index -= 1
        image = image_stack[image_stack_index]
        image_stack.pop();
        #Display the new Image
        show_img(image)
        
def undoall():
    global image,image_stack,image_stack_index,file
    while undoable():
        #Pointer is decremented for every Undo
        image_stack_index -= 1
        image = image_stack[image_stack_index]
        image_stack.pop();
        #Display the new Image
        show_img(image)
        
def rotate():
    global image,image_stack,image_stack_index,file
    image=image.rotate(45)
    image_stack.append(image)
    image_stack_index+=1
    show_img(image)

def cumsum(a):#cumulative sum for histogram equalization
    a = iter(a)
    b = [next(a)]
    for i in a:
        b.append(b[-1] + i)#similar to prefix sum calculation
    return np.array(b)

def get_histogram(image, bins):
    #Array with size of bins, set to zeros
    histogram = np.zeros(bins)
    # loop through pixels and sum up counts of pixels
    for pixel in image:
        histogram[pixel] += 1
    # return our final result
    return histogram

def histogram_equalization():
    global image,image_stack,image_stack_index,file
    img=np.asarray(image)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsvImage)
    flat = v.flatten()#flatten it to an array for easier manipulation
    hist = get_histogram(flat, 256)
    cs = cumsum(hist)
    nj = (cs - cs.min()) * 255
    N = cs.max() - cs.min()
    cs = nj / N #normalize
    cs = cs.astype('uint8')
    img_new = cs[flat]
    v = np.reshape(img_new, v.shape)
    image=cv2.merge([h,s,v])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)#converting back to colour image
    image=Image.fromarray(np.reshape(image,np.shape(hsvImage)))
    image_stack.append(image)#adding it to the stack
    image_stack_index+=1
    show_img(image)

def blurr():
    global image,image_stack,image_stack_index,file
    c=int(horizontal_blur.get())
    blur = [[1/(c*c)]*c]*c#creating filter based on input c
    img=np.asarray(image)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsvImage)
    v=signal.convolve2d(v, blur, boundary='symm', mode='same').astype('uint8')#performing convolution with the filter
    image=cv2.merge([h,s,v])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)#converting back to colour image
    image=Image.fromarray(np.reshape(image,np.shape(hsvImage)))
    image_stack.append(image)#adding it to the stack
    image_stack_index+=1
    show_img(image)
    
def sharp():
    global image,image_stack,image_stack_index,file
    c=int(horizontal_sharp.get())
    blur = [[1/(c*c)]*c]*c #creating filter based on input c
    img=np.asarray(image)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsvImage)
    vblur=signal.convolve2d(v, blur, boundary='symm', mode='same').astype('uint8')#performing convolution with the filter
    v=v+v-vblur;
    image=cv2.merge([h,s,v])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)#converting back to colour image
    image=Image.fromarray(np.reshape(image,np.shape(hsvImage)))
    image_stack.append(image)#adding it to the stack
    image_stack_index+=1
    show_img(image)
    
def edge_detector():
    global image,image_stack,image_stack_index,file
    c=int(horizontal_sharp.get())
    edgev= [[-1,0,1],[-2,0,2],[-1,0,1]]
    edgeh= [[1,2,1],[0,0,0],[-1,-2,-1]]#laplacian filter for edge detection h=horizontal,v=vertical
    img=np.asarray(image)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsvImage)
    v1=signal.convolve2d(v, edgev, boundary='symm', mode='same').astype('uint8')#convolution with the filter
    v2=signal.convolve2d(v, edgeh, boundary='symm', mode='same').astype('uint8')
    v=np.sqrt(np.square(v1)+np.square(v1));
    v*= 255.0 / v.max()#Normalize
    v=v.astype('uint8')
    image=cv2.merge([h,s,v])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image=Image.fromarray(np.reshape(image,np.shape(hsvImage)))
    image_stack.append(image)#adding it to the stack
    image_stack_index+=1
    show_img(image)
    
def negative():
    global image,image_stack,image_stack_index,file
    img = np.asarray(image)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#converting it to the hsv for manipulating v channel
    h,s,v=cv2.split(hsvImage)
    flat = v.flatten()
    #gamma=int(inputtxt.get(1.0, "end-1c"))
    flat=(255-flat).astype('uint8')#raising every element to gamma and the normalizing
    v = np.reshape(flat, np.shape(v))
    image=cv2.merge([h,s,v])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image=Image.fromarray(np.reshape(image,np.shape(hsvImage)))
    image_stack.append(image)#adding it to the stack
    image_stack_index+=1
    show_img(image)    

def gamma():
    global image,image_stack,image_stack_index,file
    img = np.asarray(image)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)#converting it to the hsv for manipulating v channel
    h,s,v=cv2.split(hsvImage)
    flat = v.flatten()
    gamma=int(inputtxt.get(1.0, "end-1c"))
    flat=(np.divide(np.power(flat,[gamma]),(pow(flat.max(),gamma-1)))).astype('uint8')#raising every element to gamma and the normalizing
    v = np.reshape(flat, np.shape(v))
    image=cv2.merge([h,s,v])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image=Image.fromarray(np.reshape(image,np.shape(hsvImage)))
    image_stack.append(image)#adding it to the stack
    image_stack_index+=1
    show_img(image)

def logarithm():
    global image,image_stack,image_stack_index,file
    img = np.asarray(image)
    hsvImage = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsvImage)
    flat = v.flatten()
    flat=(32*np.log2(flat+1)).astype('uint8')#normalizing it with 32 s 32 log 256=256
    v = np.reshape(flat, np.shape(v))
    image=cv2.merge([h,s,v])
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    image=Image.fromarray(np.reshape(image,np.shape(hsvImage)))
    image_stack.append(image)#adding it to the stack
    image_stack_index+=1
    show_img(image)
        

def resize(w, h, w_box, h_box, pil_image):
    '''
    resize a pil_image object so it will fit into
    a box of size w_box times h_box, but retain aspect ratio
    '''
    f1 = 1.0*w_box/w  # 1.0 forces float division in Python2
    f2 = 1.0*h_box/h
    factor = min([f1, f2])
    # use best down-sizing filter
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

def show_img(img):
    w_box = 500
    h_box = 500    
    w, h = img.size
    img = resize(w, h, w_box, h_box, img)
    #wr, hr = pil_image_resized.size

    tk_image = ImageTk.PhotoImage(img)
    
    label2.tk_image = ImageTk.PhotoImage(img)
    label2.config(image=label2.tk_image, width=w_box, height=h_box)

def g_quit():
    mExit=tkinter.messagebox.askyesno(title="Quit", message="Are You Sure?")
    if mExit>0:
        mGui.destroy()
        return

#open menu
def open_img():
    global image,image_stack,image_stack_index,file
    file = tkinter.filedialog.askopenfilename(initialdir='D:/Users/')
    w_box = 500
    h_box = 500
    
    image = Image.open(file)
    image_stack.append(image)
    image_stack_index=0;

    w, h = image.size

    image = resize(w, h, w_box, h_box, image)
    #wr, hr = pil_image_resized.size

    tk_image = ImageTk.PhotoImage(image)
    
    label2.tk_image = ImageTk.PhotoImage(image)
    label2.config(image=label2.tk_image, width=w_box, height=h_box)
    


def save_pic():
    global image,image_stack,image_stack_index,file
    result = filedialog.asksaveasfilename(initialdir="/", title="Select file", filetypes=(
        ('JPEG', ('*.jpg', '*.jpeg', '*.jpe')), ('PNG', '*.png'), ('BMP', ('*.bmp', '*.jdib')), ('GIF', '*.gif')))
    if result:
        image.save(result);




mGui = Tk()
mGui.title('Photo Filters')
mGui.geometry('650x500')
mGui.resizable(0, 0) #Disable Resizeability
photoFrame = Frame(mGui, bg="orange", width=500, height=500)
photoFrame.pack(side=LEFT)
filtersFrame = Frame(mGui, bg="yellow", width=150, height=500)
filtersFrame.pack(side=LEFT, fill=Y)
label2 = Label(photoFrame)
label2.pack()


#Create Buttons for All the Possible Filters
undo_btn = Button(filtersFrame, text="Undo",command=undo)
undo_btn.pack()

undoall_btn = Button(filtersFrame, text="Undo All",command=undoall)
undoall_btn.pack()

roatate_btn = Button(filtersFrame, text="Rotate",command=rotate)
roatate_btn.pack()

histogram_equalization_btn = Button(filtersFrame, text="Histogram_Equalization",command=histogram_equalization)
histogram_equalization_btn.pack()

blur_btn = Button(filtersFrame, text="Blur",command=blurr)
blur_btn.pack()

horizontal_blur=Scale(filtersFrame,from_=2,to=20,orient=HORIZONTAL)
horizontal_blur.pack()

gamma_btn = Button(filtersFrame, text="Gamma",command=gamma)
gamma_btn.pack()

inputtxt = Text(filtersFrame,height = 1,width = 5)  
inputtxt.pack()

logarithm_btn = Button(filtersFrame, text="Logarithm",command=logarithm)
logarithm_btn.pack()

edge_btn = Button(filtersFrame, text="Edge",command=edge_detector)
edge_btn.pack()

sharp_btn = Button(filtersFrame, text="Sharp",command=sharp)
sharp_btn.pack()

horizontal_sharp=Scale(filtersFrame,from_=2,to=20,orient=HORIZONTAL)
horizontal_sharp.pack()

negative_btn = Button(filtersFrame, text="Negative_Transform", command=negative)
negative_btn.pack()


#Menu Bar
menubar = Menu(mGui)
filemenu = Menu(menubar)
#Create the Menu Options that go under drop down
filemenu.add_command(label="New")
filemenu.add_command(label="Open", command=open_img)
filemenu.add_command(label="Save As",command=save_pic)
filemenu.add_command(label="Close", command=g_quit)
#Create the Main Button (e.g file) which contains the drop down options
menubar.add_cascade(label="File", menu=filemenu)
mGui.config(menu=menubar)
mGui.mainloop()