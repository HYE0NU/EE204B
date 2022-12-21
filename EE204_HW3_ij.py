# written by KAIST 20210604 Jung Hyeonu
# code for EE204B HW3; corresponds to problem 3.i to 3.j

import numpy as np
import matplotlib.pyplot as plt
from EE204_HW3_abcde import build_E, calc_area 

global img_num

def center(pattern):
    n, m = np.shape(pattern)
    y, x = np.meshgrid(range(n), range(m))
    meanx = round(np.mean(x * pattern[y, x]) / np.mean(pattern[y, x]))
    meany = round(np.mean(y * pattern[y, x]) / np.mean(pattern[y, x]))
    return meany, meanx

# method 1: rigorous evaluation of EE dipole
def get_Er_EE(EE_pattern, tile_charge, origin):

    K = 8.987551787e9 # Coulomb constant
    
    n, m = np.shape(EE_pattern)
    py, px, R = np.meshgrid(range(n), range(m), np.linspace(0.1, 100, 1000), indexing='ij')
    origin_y, origin_x = origin

    # Rx, Ry, Rz: xyz coordinate of measurement (position of R, in new Cartesian)
    theta = np.pi/4     # phi = pi
    Rx,Ry,Rz = -1*R*np.sin(theta), 0, R*np.cos(theta)

    # r: vector (from each point charge on pattern, to R)
    # electric field due to small tile charge is parallel to r vector
    rx = Rx - (px - origin_x)/n
    ry = Ry - (py - origin_y)/n
    rz = Rz

    # R component of electric field : projection of E vector into R vector
    # E_proj_R = (|E|r / |r|) dot (R / |R|) * R
    # E_R = |E| * (r dot R) / (|r||R|)
    Eabs = EE_pattern[py, px] * K * tile_charge * (-1)**(px>n) / (rx**2 + ry**2 + rz**2)
    E_R = Eabs * (Rx*rx + Ry*ry + Rz * rz) / R / (rx**2 + ry**2 + rz**2)**(0.5)

    return np.sum(E_R, axis = (0,1))

# method 2: rigorous evaluation of the regular dipole
def get_Er_dipole(n, dipole_charge, d):
    #d: difference of index between two dipole (same as n, but for clarification)

    K = 8.987551787e9 # Coulomb constant

    # Rx, Ry, Rz: xyz coordinate of measurement (position of R, in new Cartesian)
    R = np.linspace(0.1, 100, 1000)

    theta = np.pi/4     # phi = pi
    Rx, Ry, Rz = -1*R*np.sin(theta), 0, R*np.cos(theta)

    r1x, r1y, r1z = Rx-(-d/2)/n, Ry, Rz
    E1_R = K * dipole_charge * (Rx*r1x + Ry*r1y + Rz*r1z) / R / (r1x**2 + r1y**2 + r1z**2)**1.5

    r2x, r2y, r2z = Rx-(d/2)/n, Ry, Rz
    E2_R = K * (-dipole_charge) * (Rx*r2x + Ry*r2y + Rz*r2z) / R / (r2x**2 + r2y**2 + r2z**2)**1.5

    return E1_R + E2_R

    

# method 3: approximate evaluation of regilar dipole
def get_Er_dipole_aprx():
    
    R = np.linspace(0.1, 100, 1000)

    # Coulomb constant; 1/(4*pi*e0)
    K = 8.987551787e9
    # dipole moment; 1e-6 C * 1 m
    p = 1e-6
    # calculate Er directly through approximation formula
    Er = K * p * R**(-3) * 2 * np.cos(np.pi/4)

    return Er


# main function
if __name__ == "__main__":

    img_num = 29
    n = 100

    E_pattern = build_E(n)
    EE_pattern = np.concatenate((E_pattern, E_pattern), axis=1)
    area, tile_num = calc_area(n, E_pattern)
    tile_charge = 1e-6 / tile_num

    center_y, center_x = center(E_pattern) # rounded value; int type.
    dipole_pattern = np.zeros_like(EE_pattern)
    dipole_pattern[center_y, center_x] = True
    dipole_pattern[center_y, n + center_x] = True

    # problem 3.i
    # by introducing new Cartesian coordinate system, we can set origin as midpoint of two charge centers
    # Origin: [n//2 + center_x, center_y]
    # Path gamma: 0.1m <=r <= 100m, theta = pi/4, phi = pi (-x direction, y=0, z=-x)

    new_origin = (center_y, n//2 + center_x)
    Er_EE = get_Er_EE(EE_pattern, tile_charge, new_origin)
    Er_dipole = get_Er_dipole(n, 1e-6, n)
    Er_dipole_aprx = get_Er_dipole_aprx()
    plt.figure(figsize = (16.0, 12.0)) #figsize = (64.0, 48.0)
    plt.title("$E_r$ at : 0.1 m < r < 100 m, $\\theta = \pi/4, \phi = \pi$")
    plt.plot(Er_EE, label = "EE", alpha = 0.6)
    plt.plot(Er_dipole, label = "Dipole", alpha = 0.6)
    plt.plot(Er_dipole_aprx, label = "Dipole - approximation", alpha = 0.6)
    plt.legend()
    plt.xlabel("r    [m]")
    plt.yscale('log')
    plt.ylabel("$E_r$    [log(V/m)]")
    plt.savefig("%d.png" %(img_num))
    img_num += 1
    
    # drawing x as log scale
    if True:
        plt.title("$E_r$ at\t:\t0.1 m < r < 100 m, $\\theta = \pi/4, \phi = \pi$")
        plt.xscale('log')
        plt.xlabel('r    [log(m)]')
        plt.savefig("%d.png" %(img_num))
        img_num += 1

    # problem 3.j

    rdiff_EE = np.abs(Er_EE - Er_dipole) / Er_dipole
    rdiff_aprx = np.abs(Er_dipole_aprx - Er_dipole) / Er_dipole
    plt.figure(figsize = (16.0, 12.0))
    plt.title("Relative Difference of |$E_r$|")
    plt.plot(rdiff_EE, label = "EE - Dipole", alpha = 0.6)
    plt.plot(rdiff_aprx, label = "Dipole(Approximation) - Dipole", alpha = 0.6)
    plt.legend()
    plt.xlabel("r    [m]")
    plt.yscale('log')
    plt.ylabel("$\\frac{\Delta E_r}{E_r}$    [log(V/m)]")
    plt.savefig("%d.png" %(img_num))
    img_num += 1

    # drawing x as log scale
    if True:
        plt.title("Relative Difference of |$E_r$|")
        plt.xscale('log')
        plt.xlabel('r    [log(m)]')
        plt.savefig("%d.png" %(img_num))
        img_num += 1

    