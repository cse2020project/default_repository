import math

def coor(x):
    width = 1280
    angle = x/width *2 * math.pi #radian
    #angle *= 57.295779513 #degree
    return angle
