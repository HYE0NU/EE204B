# written by KAIST 20210604 Jung Hyeonu
# code for EE204B HW3; corresponds to problem 3.a to 3.e

import numpy as np
import matplotlib.pyplot as plt

global img_num

# build_E: produce the 'E' pattern in n*n tiles
# resulting array is consisted of boolean values(True/False) 
def build_E(n):

    xx, yy = np.meshgrid(range(n), range(n))

    # vstick
    vstick_start = (int(n *2 /8), int(n *1 /8))
    vstick_stop = (int(n *3 /8), int(n *7 /8))
    vstick = (xx>=vstick_start[0]) & (xx<vstick_stop[0]) & (yy>=vstick_start[1]) & (yy<vstick_stop[1])

    # upstick
    upstick_start = (int(n *2 /8), int(n *6 /8))
    upstick_stop = (int(n *6 /8), int(n *7 /8))
    upstick = (xx>=upstick_start[0]) & (xx<upstick_stop[0]) & (yy>=upstick_start[1]) & (yy<upstick_stop[1])

    # midstick
    midstick_start = (int(n *2 /8), int(n *4 /8))
    midstick_stop = (int(n *5 /8), int(n *5 /8))
    midstick = (xx>=midstick_start[0]) & (xx<midstick_stop[0]) & (yy>=midstick_start[1]) & (yy<midstick_stop[1])

    # botstick
    botstick_start = (int(n *2 /8), int(n *1 /8))
    botstick_stop = (int(n *6 /8), int(n *2 /8))
    botstick = (xx>=botstick_start[0]) & (xx<botstick_stop[0]) & (yy>=botstick_start[1]) & (yy<botstick_stop[1])

    capital_E = vstick | upstick | midstick | botstick

    return capital_E

# plot_contourf: plot contour of pattern using matplotlib.pyplot
# save figure, no return
def plot_contourf(pattern, title, color_label):
    global img_num
    plt.figure(figsize=(8.0, 6.0))
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
    plt.figure(figsize=(14.0, 6.0))
    
    plt.subplot(1, 2, 1)
    plt.contourf(pattern1)
    plt.title(title1)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label = color_label)

    plt.subplot(1, 2, 2)
    plt.title(title2)
    plt.contourf(pattern2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label = color_label)

    plt.tight_layout()
    plt.savefig("%d.png" %(img_num))
    img_num += 1

# calc_area: count number of True in pattern and calculate the area ratio
# print the result.  return value: (area of E, tile count)
def calc_area(n, E_pattern):
    T_cnt = 0
    for y in range(n):
        for x in range(n):
            if E_pattern[x][y] == True:
                T_cnt += 1
    area = T_cnt/n**2
    #print("Area of E at n=" + str(n) + "\t: " + str(area))
    return area, T_cnt
    
# build_E_field: build E field at certain z value
# return value: Electric field meshgrid
def build_E_field(Type, resolution, pattern, z, tile_charge):

    # Coulomb constant; 1/(4*pi*e0)
    K = 8.987551787e9         

    # px, py: pattern x, y index   fx, fy: field x, y index
    n = np.size(pattern, 0)
    py, px, fy, fx = np.meshgrid(range(n), range(n), range(resolution), range(resolution), indexing='ij')
    # set to 'ij' indexing mode, because 'xy' indexing mode(default) is not applied to 3rd, 4th axes.
    
    # component of vector r (position vector)
    rx = ((fx+0.5)/resolution - (px+0.5)/n)
    ry = ((fy+0.5)/resolution - (py+0.5)/n)
    rz = z

    # calculate E field using Coulomb's law and superposition thm.
    Eref = (pattern[py, px]) * K * tile_charge * ((rx**2 + ry**2 + rz**2)**(-3/2)) #kq/r^3

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

# build_V_field: build V field at certain z
# return value: electric potential field meshgrid
def build_V_field(resolution, pattern, z, tile_charge):
    
    # Coulomb constant; 1/(4*pi*e0)
    K = 8.987551787e9 

    # px, py: pattern x, y index   fx, fy: field x, y index
    n = np.size(pattern, 0)
    py, px, fy, fx = np.meshgrid(range(n), range(n), range(resolution), range(resolution), indexing='ij')
    # set to 'ij' indexing mode, because 'xy' indexing mode(default) is not applied to 3rd, 4th axes.

    # component of vector r
    rx = ((fx+0.5)/resolution - (px+0.5)/n)
    ry = ((fy+0.5)/resolution - (py+0.5)/n)
    rz = z

    V = np.sum((pattern[py, px]) * K * tile_charge * ((rx**2 + ry**2 + rz**2)**(-1/2)), axis=(0,1)) # V = sigma{kq/r}
    
    return V


# main function
if __name__ == "__main__":

    resolution = 100
    img_num = 1
    
    for n in [20, 50, 100]:
        # problem 3.a
        E_pattern_n = build_E(n)
        plot_contourf(E_pattern_n, "n = %d, tile" %(n), None)
        area, tile_num = calc_area(n, E_pattern_n)

        # problem 3.b
        tile_charge = 1e-6 / tile_num
        Eabs = build_E_field("Eabs", resolution, E_pattern_n, 0.1, tile_charge)
        plot_contourf(Eabs, "n = %d, z = 0.1 m, |E|" %(n), "Electric field [V/m]")
    
    # E pattern is already built, and saved in E_pattern_n
    # all variables are set to the condition n=100.

    # problem 3.c
    for z in [0.01, 0.1, 1, 5, 10, 100]:
        Ez = build_E_field("Ez", resolution, E_pattern_n, z, tile_charge)
        plot_contourf(Ez, "n = 100, z = %f m, $E_z$" %(z), "Electric field [V/m]")

    # problem 3.d
    V = build_V_field(resolution, E_pattern_n, 0.1, tile_charge)
    plot_contourf(V, "n = 100, z = 0.1 m, V" , "Electric potential [V]")

    # problem 3.e
    Ex = build_E_field("Ex", resolution, E_pattern_n, 0.1, tile_charge)
    Ey = build_E_field("Ey", resolution, E_pattern_n, 0.1, tile_charge)
    Egradx = np.gradient(V, axis=1) * (-1) * resolution
    Egrady = np.gradient(V, axis=0) * (-1) * resolution
    plot_contourf_two(Ex, "z = 0.1 m, $E_x$", Egradx, "z = 0.1 m, $E_x$ ($\\frac{-dV}{dx}$)", "Electric field [V/m]")
    plot_contourf_two(Ey, "z = 0.1 m, $E_y$", Egrady, "z = 0.1 m, $E_y$ ($\\frac{-dV}{dx}$)", "Electric field [V/m]")

