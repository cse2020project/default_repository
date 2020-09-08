import cv2
import os

path='C:/Users/eunji/Desktop/old_image'

num=1
for i in os.listdir(path):
    print(num)
    img=cv2.imread(path+"/"+i)
    img_trim=img[384:1024,:]
    cv2.imwrite(os.path.join("C:/Users/eunji/Desktop/new_image",i),img_trim)
    print(img_trim.shape)
    num+=1
