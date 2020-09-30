import math

def coor(x):
    width = 1080 #이미지 크기 가져오는 식으로 바꿔야함
    angle = x/width *2 * math.pi
    angle *= 57.295779513
    return angle
