import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Pool
from matplotlib.patches import Rectangle

sys.path.append("/Users/theo/Desktop/ALP_Simulation/New_code/Modules")

File_path = "/Users/theo/Desktop/ALP_Simulation/New_code/Ressources/mesons/"

Adress = "/Users/theo/Desktop/ALP_Simulation/New_code/Results/Thesis_Plots/"
matplotlib.rcParams.update({'font.size': 24})
from various import *
from br_validation import *
from llpmodel import LLPModel
from createllp import CreateLLP
from eventcounter import MyEventCounter
from llpgenerator import LLPGenerator



Neutral_Pion = [[],[],[]]
Kaon_Long = [[],[],[]]
Neutral_B = [[],[],[]]
Charged_Kaon = [[],[],[]]

p_NP, w_NP = convert_list_to_momenta(File_path + "LHC_111_log.txt", mass = 0.135)
p_KL, w_KL = convert_list_to_momenta(File_path + "LHC_130_log.txt", mass = 0.495)
p_NB, w_NB = convert_list_to_momenta(File_path + "LHC_5_log.txt", mass = 5.279)
p_CK, w_CK = convert_list_to_momenta(File_path + "LHC_321_log.txt", mass = 0.495)

BINX = 100
BINY = 80
map = "jet"

Neutral_Pion[0], Neutral_Pion[1], Neutral_Pion[2]= convert_to_hist_list(p_NP, w_NP, bin_x = BINX, bin_y = BINY, do_return = True, do_plot = False)
Kaon_Long[0], Kaon_Long[1], Kaon_Long[2] = convert_to_hist_list(p_KL, w_KL, bin_x = BINX, bin_y = BINY, do_return = True, do_plot = False)
Neutral_B[0], Neutral_B[1], Neutral_B[2] = convert_to_hist_list(p_NB, w_NB, bin_x = BINX, bin_y = BINY, do_return = True, do_plot = False)
Charged_Kaon[0], Charged_Kaon[1], Charged_Kaon[2] = convert_to_hist_list(p_CK, w_CK, bin_x = BINX, bin_y = BINY, do_return = True, do_plot = False)

sum_w_NP = 0
sum_w_KL = 0
sum_w_NB = 0
sum_w_CK = 0

tot_w_KL = 0
tot_w_NB = 0
tot_w_CK = 0

for i in range(len(Neutral_Pion[0])):
    if Neutral_Pion[0][i] < np.log10(0.1/480): sum_w_NP += Neutral_Pion[2][i]
for i in range(len(Kaon_Long[0])):
    tot_w_KL += Kaon_Long[2][i]
    if Kaon_Long[0][i] < np.log10(0.1/480): sum_w_KL += Kaon_Long[2][i]
for i in range(len(Neutral_B[0])):
    tot_w_NB += Neutral_B[2][i]
    if Neutral_B[0][i] < np.log10(0.1/480): sum_w_NB += Neutral_B[2][i]
for i in range(len(Charged_Kaon[0])):
    tot_w_CK += Charged_Kaon[2][i]
    if Charged_Kaon[0][i] < np.log10(0.1/480): sum_w_CK += Charged_Kaon[2][i]


# // code for the contour plot i presume, not sure
# FASER_Primakoff = np.genfromtxt("/Users/theo/Desktop/ALP_Simulation/New_code/Ressources/ALP-W/bounds/limits_FASER-Primakoff.txt", delimiter = " ")
# print(FASER_Primakoff.T[0])
#
# Data_gg = FASER_Primakoff.T[1]*0.22 #(Coupling to gamma/gamma)
# Chosen_pnt = [[3*10**-2, 3.8*10**-2, 6*10**-2, 7.6*10**-2, 1*10**-1, 1.6*10**-1, 2.1*10**-1, 3.1*10**-1], [2*10**-3, 1*10**-3, 5*10**-4, 2.7*10**-4, 1*10**-4, 4*10**-5, 2*10**-5, 4.1*10**-6]]
# lower_bound = FASER_Primakoff.T[1]*0.5
# upper_bound = FASER_Primakoff.T[1]*1.5
#
#
# blank = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
# plt.figure(figsize=(10,8))
# plt.subplot(1,2,1)
# plt.plot(FASER_Primakoff.T[0], Data_gg, color = "red", label="Primakoff gg-coupling scale")
# # plt.plot(FASER_Primakoff.T[0], FASER_Primakoff.T[1], color = "black", label="Primakoff WW-coupling scale")
# plt.plot(Chosen_pnt[0], Chosen_pnt[1], color = "orange", label = "Chosen points WW-coupling scale")
# # plt.plot(FASER_Primakoff.T[0], upper_bound, color = "blue")
# # plt.plot(FASER_Primakoff.T[0], lower_bound, color = "cyan")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Mass [GeV]")
# plt.ylabel("Coupling_gg [Gev^-1]")
# # plt.legend()
# # plt.xlim([10**-2, 1])
# # plt.ylim([10**-6, 10**-2])
# plt.subplot(1,2,2)
# plt.plot(FASER_Primakoff.T[0], FASER_Primakoff.T[1], color = "black", label="Primakoff WW-coupling scale")
# plt.plot(Chosen_pnt[0], Chosen_pnt[1], color = "orange", label = "Chosen points WW-coupling scale")
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel("Mass [GeV]")
# plt.ylabel("Coupling_WW [Gev^-1]")
# # plt.legend()
# plt.show()

# // ends here

# plt.figure(figsize=(10,8))
# plt.hist2d(Neutral_Pion[0], Neutral_Pion[1], weights = Neutral_Pion[2], bins = [BINX,BINY], range = [[-5,0],[0,4]], norm=matplotlib.colors.LogNorm(), cmap = map)
# plt.axvline(np.log10(0.1/480), label = "FASER1", color = "black", linestyle="--")
# plt.xlabel(r"log$_{10}$($\theta$)")
# plt.ylabel(r"log$_{10}$(p)")
# plt.ylim(0,4.1)
# plt.title(r"Spectrum for $\pi^0$ from IP    " + "  Effective XS: "+ str(round(sum_w_NP/(10**9), 2)) + " [mb]")
# plt.colorbar()
# plt.legend()
# # plt.savefig(Adress + r"Spectrum for $\pi^0$ from IP.png",  dpi=500)
# plt.show()



plt.figure(figsize=(12,9.6))
plt.subplots_adjust(left=0.09, bottom=0.09, right=1, top=0.93, wspace=0.2, hspace=0.2)
plt.hist2d(Kaon_Long[0], Kaon_Long[1], weights = Kaon_Long[2], bins = [BINX,BINY], range = [[-5,0],[0,4]], norm=matplotlib.colors.LogNorm(), cmap = map)
plt.axvline(np.log10(0.1/480), label = "FASER", color = "black", linestyle="--")
plt.xlabel(r"Angle wrt. beam axis log$_{10}$($\theta$) [rad]")
plt.ylabel(r"Momentum log$_{10}$(p) [GeV]")
plt.ylim(0,4.1)
plt.title(r"Spectrum of $K^0_L$ from the ATLAS IP")
plt.text(np.log10(0.1/480), 0.1, "FASER", fontsize=18,color="black",rotation=-90)
plt.text(-2.75,3.95, "Effective Cross Section: "+ str(round(sum_w_KL/(10**9), 2)) + " [mb]", fontsize=20,color="black",rotation=0)
plt.text(-2.63,3.8, "Total Cross Section: "+ str(round(tot_w_KL/(10**9), 2)) + " [mb]", fontsize=20,color="black",rotation=0)
plt.colorbar()
plt.savefig(Adress + r"Spectrum for $K^0_L$ from IP.png",  dpi=500)
plt.show()

plt.figure(figsize=(12,9.6))
plt.subplots_adjust(left=0.09, bottom=0.09, right=1, top=0.93, wspace=0.2, hspace=0.2)
plt.hist2d(Neutral_B[0], Neutral_B[1], weights = Neutral_B[2], bins = [BINX,BINY], range = [[-5,0],[0,4]], norm=matplotlib.colors.LogNorm(), cmap = map)
plt.axvline(np.log10(0.1/480), label = "FASER", color = "black", linestyle="--")
plt.xlabel(r"Angle wrt. beam axis log$_{10}$($\theta$) [rad]")
plt.ylabel(r"Momentum log$_{10}$(p) [GeV]")
plt.ylim(0,4.1)
plt.title(r"Spectrum of $B^0$ from the ATLAS IP")
plt.text(np.log10(0.1/480), 0.1, "FASER", fontsize=18,color="black",rotation=-90)
plt.text(-3,3.95, "Effective Cross Section: "+ str(round(sum_w_NB/(10**9), 6)) + " [mb]", fontsize=20,color="black",rotation=0)
plt.text(-2.42,3.8, "Total Cross Section: "+ str(round(tot_w_NB/(10**9), 2)) + " [mb]", fontsize=20,color="black",rotation=0)
plt.colorbar()
plt.savefig(Adress + r"Spectrum for $B^0$ from IP.png",  dpi=500)
plt.show()

plt.figure(figsize=(12,9.6))
plt.subplots_adjust(left=0.09, bottom=0.09, right=1, top=0.93, wspace=0.2, hspace=0.2)
plt.hist2d(Charged_Kaon[0], Charged_Kaon[1], weights = Charged_Kaon[2], bins = [BINX,BINY], range = [[-5,0],[0,4]], norm=matplotlib.colors.LogNorm(), cmap = map)
plt.axvline(np.log10(0.1/480), label = "FASER", color = "black", linestyle="--")
plt.xlabel(r"Angle wrt. beam axis log$_{10}$($\theta$) [rad]")
plt.ylabel(r"Momentum log$_{10}$(p) [GeV]")
plt.ylim(0,4.1)
plt.title(r"Spectrum of $K^\pm$ from the ATLAS IP")
plt.text(np.log10(0.1/480), 0.1, "FASER", fontsize=20,color="black",rotation=-90)
plt.text(-2.66,3.95, "Effective Cross Section: "+ str(round(sum_w_CK/(10**9), 2)) + " [mb]", fontsize=20,color="black",rotation=0)
plt.text(-2.64,3.8, "Total Cross Section: "+ str(round(tot_w_CK/(10**9), 2)) + " [mb]", fontsize=20,color="black",rotation=0)
plt.colorbar()
plt.savefig(Adress + r"Spectrum for $K^\pm$ from IP.png",  dpi=500)
plt.show()
