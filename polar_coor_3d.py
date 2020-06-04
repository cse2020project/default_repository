import math

def spherical_coor(x, y, width, height):
    theta = x/width *2 * math.pi
    phi = y/height * math.pi
    return theta, phi


def polar_coor(x, y, w, h):
    theta, phi = spherical_coor(x, y, w, h)
    x = math.cos(theta) * math.sin(phi)
    y = math.sin(theta) * math.sin(phi)
    z = math.cos(phi)

    maximum = max(abs(x), abs(y), abs(z))
    xx = x / maximum
    yy = y / maximum
    zz = z / maximum

    if (xx==1 or xx==-1):
        # project X
        x3D= xx*0.5
        rho = x3D / x
        y3D = rho * y
        z3D = rho * z
    elif (yy==1 or yy==-1):
        # project Y
        y3D = yy*0.5
        rho = y3D /y
        x3D = rho * x
        z3D = rho * z
    else:
        # project Z
        z3D = zz*0.5
        rho = z3D / z
        x3D = rho * x
        y3D = rho * y

    return (x3D, y3D, z3D)


# polar_coor(x, y, w, h)
# x, y : target coordinates in equirectangular image to convert to 3D polar coordinates
# w, h : size of equirectangular image
x, y, z = polar_coor(1931, 1727, 6080, 3040)
print((x,y,z))
