# written by KAIST 20210604 Jung Hyeonu
# code for EE204B HW3; corresponds to problem 3.f to 3.h

import numpy as np
import matplotlib.pyplot as plt
from EE204_HW3_abcde import build_E, calc_area 

global img_num

# plot_contourf: plot contour of pattern using matplotlib.pyplot
# save figure, no return
def plot_contourf(pattern, title, color_label):
    global img_num
    plt.figure(figsize=(14.0, 6.0))
    plt.contourf(pattern)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label = color_label)
    plt.savefig("%s.png" %(img_num))
    #plt.show()
    img_num += 1

def plot_contourf_two(pattern1, title1, pattern2, title2, color_label):
    global img_num
    plt.figure(figsize=(14.0, 12.0))
    
    plt.subplot(2, 1, 1)
    plt.contourf(pattern1)
    plt.title(title1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label = color_label)

    plt.subplot(2, 1, 2)
    plt.title(title2)
    plt.contourf(pattern2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label = color_label)

    plt.tight_layout()
    plt.savefig("%d.png" %(img_num))
    img_num += 1
    
# build_E_field: build E field at certain z value
# return value: Electric field meshgrid
def build_E_field(Type, resolution, pattern, z, tile_charge):

    # Coulomb constant; 1/(4*pi*e0)
    K = 8.987551787e9         

    # px, py: pattern x, y index   fx, fy: field x, y index
    n = np.size(pattern, 0)
    py, px, fy, fx = np.meshgrid(range(n), range(2*n), range(resolution), range(2*resolution), indexing = "ij")
    
    # component of vector r (position vector)
    rx = ((fx+0.5)/resolution - (px+0.5)/n)
    ry = ((fy+0.5)/resolution - (py+0.5)/n)
    rz = z

    # calculate E field using Coulomb's law and superposition thm.
    Eref = (pattern[py, px]) * K * tile_charge * (-1)**(px>n) * ((rx**2 + ry**2 + rz**2)**(-3/2))

    if Type == "Ex":
        Ex = np.sum(Eref*rx, axis=(0,1))
        return Ex
    elif Type == "Ey":
        Ey = np.sum(Eref*ry, axis=(0,1))
        return Ey
    elif Type == "Ez":
        Ez = np.sum(Eref*rz, axis=(0,1))
        return Ez
    elif Type == "Eabs":
        Ex = np.sum(Eref*rx, axis=(0,1))
        Ey = np.sum(Eref*ry, axis=(0,1))
        Ez = np.sum(Eref*rz, axis=(0,1))
        Eabs = (Ex**2 + Ey**2 + Ez**2)**0.5
        return Eabs
    else:
        return None

def center(pattern):
    n, m = np.shape(pattern)
    y, x = np.meshgrid(range(n), range(m))
    meanx = round(np.mean(x * pattern[y, x]) / np.mean(pattern[y, x]))
    meany = round(np.mean(y * pattern[y, x]) / np.mean(pattern[y, x]))
    return meany, meanx

# main function
if __name__ == "__main__":

    resolution = 100
    img_num = 16
    n = 100

    E_pattern = build_E(n)
    EE_pattern = np.concatenate((E_pattern, E_pattern), axis=1)
    plot_contourf(EE_pattern, "n = 100, EE tile" , None)
    area, tile_num = calc_area(n, E_pattern)
    tile_charge = 1e-6 / tile_num

    # problem 3.f
    for z in [0.01, 0.1, 1, 5, 10, 100]:
        Ex = build_E_field('Ex', resolution, EE_pattern, z, tile_charge)
        plot_contourf(Ex, "z = %f m, $E_x$" %(z), "Electric field [V/m]")

    # problem 3.g
    center_y, center_x = center(E_pattern) # rounded value; int type.
    # since center_x,y are charge center coordinates of E_pattern and EE_pattern is concatenation of two E_patterns,
    # the position of each dipole can be set as below.
    dipole_pattern = np.zeros_like(EE_pattern)
    dipole_pattern[center_y, center_x] = True
    dipole_pattern[center_y, n + center_x] = True
    print((center_x/n-1, center_y/n), (center_x/n, center_y/n))
    #print(center_y, center_x)

    #problem 3.h
    dipole_charge = 1e-6
    for z in [0.01, 0.1, 1, 5, 10, 100]:
        Ex = build_E_field('Ex', resolution, EE_pattern, z, tile_charge)
        Ex_dipole = build_E_field('Ex', resolution, dipole_pattern, z, dipole_charge)
        plot_contourf_two(Ex, 'z = %f m, $E_x$' %(z), Ex_dipole, 'z = %f m, $E_x$, dipole' %(z), "Electric field [V/m]")
