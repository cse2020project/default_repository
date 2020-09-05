import math

def coor2angle(x, width, height):
    angle = x/width *2 * math.pi #get radian
    angle *= 57.295779513 #convert to degree
    return angle

# coor2angle(x, w, h) : target x coordinate of the image
angle = coor2angle(1931, 6080, 3040)

print(angle)
