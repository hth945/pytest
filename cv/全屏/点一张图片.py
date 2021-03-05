import numpy as np
import cv2
import tkinter as tk


width = 3840
height = 2160


# width = 3838
# height = 2158


def flickNNNN():
    frames = np.ones((height, width, 3), dtype=np.float32)
    # for i in range(int(height/2)):
    #     frames[i*2,:] = [0.5,0.0,0.5]
    #     frames[i*2+1,:] = [0.0,0.5,0.0]

    for i in range(int(height/2)):
        for j in range(width):
            if j %2 == 0:
                frames[i*2,j] = [0.5,0.0,0.5]
                frames[i*2+1,j] = [0.0,0.5,0.0]
            else:
                frames[i * 2+1, j] = [0.5, 0.0, 0.5]
                frames[i * 2, j] = [0.0, 0.5, 0.0]
    frames[:,-1:] = [0.0, 1.0,0.0]
    frames[:, :1] = [0.0, 1.0, 0.0]
    frames[:1, :] = [0.0, 1.0, 0.0]
    frames[-1:, :] = [0.0, 1.0, 0.0]
    return frames
def flick():
    frames = np.ones((height, width, 3), dtype=np.float32)
    for i in range(int(height/2)):
        frames[i*2,:] = [0.5,0.0,0.5]
        frames[i*2+1,:] = [0.0,0.5,0.0]

    frames[:,-1:] = [0.0, 1.0,0.0]
    frames[:, :1] = [0.0, 1.0, 0.0]
    frames[:1, :] = [0.0, 1.0, 0.0]
    frames[-1:, :] = [0.0, 1.0, 0.0]
    return frames

def black():
    frames = np.ones((height, width, 3), dtype=np.float32)
    frames[:,:] = [0.0, 0.0, 0.0]
    frames[:,-5:] = [0.0, 1.0,0.0]
    frames[:, :5] = [0.0, 1.0, 0.0]
    frames[:5, :] = [0.0, 1.0, 0.0]
    frames[-5:, :] = [0.0, 1.0, 0.0]

    return frames

def Gray31():
    frames = np.ones((height, width, 3), dtype=np.float32)
    frames[:,:] = [31/255, 31/255,31/255]
    return frames

def xcolor():
    frames = np.ones((height, width, 3), dtype=np.float32)
    for i in range(width):
        for j in range(height):
            frames[j,i] = [-(j-height)/height,-(j-height)/height,-(j-height)/height]
    return frames

def ycolor():
    frames = np.ones((height, width, 3), dtype=np.float32)
    for i in range(height):
        for j in range(width):
            frames[i,j] = [-(j-width)/width,-(j-width)/width,-(j-width)/width]
    return frames



if __name__ == '__main__':

    # root = tk.Tk()
    # height = root.winfo_screenwidth()
    # width  = root.winfo_screenheight()
    # root.destroy()
    # print(width,height)

    # image = cv2.imread('test.png')
    image = flick()
    # image2=flickNNNN()
    # print(image2[3,3])
    cv2.imwrite("test2.png",image*255.0)








    print(image.shape)
    window_name = 'projector'
    # cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
    #                       cv2.WINDOW_FULLSCREEN)


    cv2.imshow(window_name, image)
    # cv2.moveWindow(window_name, -3840, 1080-2160)
    cv2.moveWindow(window_name, -3840-16, 1080 - 2160 -100)

    cv2.waitKey()
    cv2.destroyAllWindows()

