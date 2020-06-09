import math

def coor(x, width, height):
    angle = x/width *2 * math.pi
    angle *= 57.295779513
    return angle

# coor(x, w, h) : target x coordinate of the image
angle = coor(1931, 6080, 3040)

print(angle)
