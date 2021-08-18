import numpy as np
import sys
import argparse
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from multiprocessing import Pool, Lock
import matplotlib as mpl

#The folowing part is to be used only if one wants to run the code over many different mass and coupling points.
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('--mass','-m', default=None, type=float,help='ALP mass')
parser.add_argument('--index','-c', default=None, type=float,help='ALP coupling')

args = parser.parse_args()

#Values of the couplings
Couplings_1 = [8.08*10**-4, 6.84*10**-4, 5.45*10**-4, 4.26*10**-4, 3.16*10**-4, 2.33*10**-4]
Couplings_2 = [1.81*10**-4, 1.45*10**-4, 1.12*10**-4, 8.97*10**-5, 6.32*10**-5, 5.09*10**-5]
Couplings_3 = [3.67*10**-5, 2.94*10**-5, 2.83*10**-5, 2.31*10**-5, 1.97*10**-5, 1.78*10**-5]
Couplings_4 = [1.52*10**-5, 1.44*10**-5, 1.08*10**-5, 1.06*10**-5, 8.22*10**-6, 7.52*10**-6]
Couplings_5 = [5.52*10**-6, 4.08*10**-6, 2.83*10**-6, 2.11*10**-6, 1.56*10**-6, 1.16*10**-6]
Couplings_6 = [8.23*10**-7, 6.07*10**-7, 4.71*10**-7]

#Name to give to the output file for the corresponding coupling.
Couplings_names_1 = ["8,08x10^-4", "6.84x10^-4", "5.45x10^-4", "4.26x10^-4", "3.16x10^-4", "2.33x10^-4"]
Couplings_names_2 = ["1.81x10^-4", "1.45x10^-4", "1.12x10^-4", "8.97x10^-5", "6.32x10^-5", "5.09x10^-5"]
Couplings_names_3 = ["3.67x10^-5", "2.94x10^-5", "2.83x10^-5", "2.31x10^-5", "1.97x10^-5", "1.78x10^-5"]
Couplings_names_4 = ["1.52x10^-5", "1.44x10^-5", "1.08x10^-5", "1.06x10^-5", "8.22x10^-6", "7.52x10^-6"]
Couplings_names_5 = ["5.52x10^-6", "4.08x10^-6", "2.83x10^-6", "2.11x10^-6", "1.56x10^-6", "1.16x10^-6"]
Couplings_names_6 = ["8.23x10^-7", "6.07x10^-7", "4.71x10^-7"]


coup = []
#The code is run over different amples of couplings for limitation of the copmuting power (16Go of RAM), the number od elements in each list can be increased with better computing power or smaller value for the "nsample" variable.
if args.index == 1:
    coup = Couplings_names_1

if args.index == 2:
    coup = Couplings_names_2

if args.index == 3:
    coup = Couplings_names_3

if args.index == 4:
    coup = Couplings_names_4

if args.index == 5:
    coup = Couplings_names_5

if args.index == 6:
    coup = Couplings_names_6

mass = str(args.mass)

path = []
path_save = []
Data_Sep, Data_w_sep, Data_e1_sep, Data_e2_sep = [], [], [], []

#The aim of this code is to retrieve in the end the number of events for a specific energy for both photons (which can differ from each other) and a specific separation.
#The bins for which we want the number of events are then defined for energy of both photons and separation resulting in a 3D matrix.
E_gamma1_bin_edg = [0,100,200,300,400,500,1000,3000,7000]
E_gamma2_bin_edg = [0,100,200,300,400,500,1000,3000,7000]
Sep_bin_edg = [0,0.2,0.3,0.5,1,2,200]

#The following function computes to which element of the 3D matrix a specific Monte Carlo event belongs to.
#It returns the indices along the 3 axis of the 3D matrix and the number of event to be added in this specific position (since not only one Monte Carlo event will fit in each bin of the 3D matrix).
def Get_3D_matrix(idx, e_1, e_2, sep, w, bin_e1 = E_gamma1_bin_edg, bin_e2 = E_gamma2_bin_edg, bin_sep = Sep_bin_edg):
    idx1, idx2, idx3, weight = 0, 0, 0, 0
    for i in range(len(E_gamma1_bin_edg)-1):
        if e_1[idx] <= bin_e1[i+1] and e_1[idx] > bin_e1[i]:
            for j in range(len(E_gamma2_bin_edg)-1):
                if e_2[idx] <= bin_e2[j+1] and e_2[idx] > bin_e2[j]:
                    for k in range(len(Sep_bin_edg)-1):
                        if sep[idx] <= bin_sep[k+1] and sep[idx] > bin_sep[k]:
                            idx1, idx2, idx3, weight = i, j, k, w[idx]
    return idx1, idx2, idx3, weight

#Runing over all the couplings and reading the data for each of the files.
for i in range(len(coup)):
    path = "data/Discovery_potential/Raw_ALP_Decay/Data_Gamma_analysed/Mass_" + mass + "Coup_" + coup[i] + ".npy"
    path_save.append("data/Discovery_potential/3D_XS_Matrix/" + "Mass_" + mass + "Coup_" + coup[i] + ".npy")
    __, __, Sep, w_sep, e1_sep, e2_sep  = np.load(path)
    Data_Sep.append(Sep.tolist())
    Data_w_sep.append(w_sep.tolist())
    Data_e1_sep.append(e1_sep.tolist())
    Data_e2_sep.append(e2_sep.tolist())

#The following function uses the previously defined "Get_3D_matrix" function to compute for each Monte Carlo event, the bins to which it corresponds in the 3D matrix.
#The function returns in the end, the 3D matrix with the number of events for each energies and separation.
def Get_output(idx_bm, e1= Data_e1_sep, e2 = Data_e2_sep, Sep = Data_Sep, W = Data_w_sep):
    sum_weight = 0
    XS_matrix = np.ndarray(shape = (len(E_gamma1_bin_edg)-1, len(E_gamma2_bin_edg)-1, len(Sep_bin_edg)-1))
    XS_matrix.fill(0)
    for idx in range(len(e1[0])):
        idx1,idx2,idx3,w = 0,0,0,0
        idx1, idx2, idx3, w = Get_3D_matrix(idx, e1[idx_bm], e2[idx_bm], Sep[idx_bm], W[idx_bm])
        XS_matrix[idx1][idx2][idx3]+=w

    idx_1_list, idx_2_list, idx_3_list, Matrix_comp = [], [], [], []

    #Transforming the 3D matrix into a list with the elements and 3 corresponding lists giving the bin of each element of the 3D matrix along each direction.
    for i in range(len(E_gamma1_bin_edg)-1):
        for j in range(len(E_gamma2_bin_edg)-1):
            for k in range(len(Sep_bin_edg)-1):
                idx_1_list.append(i)
                idx_2_list.append(j)
                idx_3_list.append(k)
                Matrix_comp.append(XS_matrix[i][j][k])

    # print("The initial number of events is " + str(np.sum(W[idx_bm])))
    # print("The total number of events in the matrix is " + str(np.sum(Matrix_comp)))
    print("The percentage of events in the matrix is " + str(np.sum(Matrix_comp)*100/np.sum([W[idx_bm]])))
    return idx_1_list, idx_2_list, idx_3_list, Matrix_comp

#This code is optimised to run in parallel, here the code can compute all of the mentionned above quantites for one specific mass but different coupling points at the same time.
if __name__ == "__main__":
    index = np.arange(0, len(coup), 1)
    p = Pool(12)
    res = p.map(Get_output, index)
    p.close()
    icoup = 0
    for [idx1,idx2,idx3,Mat_comp] in res:
        Output_array = np.array([idx1, idx2, idx3, Mat_comp]).T
        np.save(path_save[icoup], Output_array)
        icoup += 1
