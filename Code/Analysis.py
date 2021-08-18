import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from multiprocessing import Pool
from mpl_toolkits.axes_grid1 import ImageGrid

sys.path.append("/Users/theo/Desktop/ALP_Simulation/New_code/Modules")

from various import *
from br_validation import *
from llpmodel import LLPModel
from createllp import CreateLLP
from eventcounter import MyEventCounter
from llpgenerator import LLPGenerator

masses = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] #in GeV so from 1 MeV to 1 GeV

mass_1, mass_2 = 0.001, 1
coup_1, coup_2, coup_3, coup_4 = 0.0001, 0.001, 0.01, 1
coup = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3]
min_1, max_1 = 10**-11,10**8
min_2, max_2 = 10**-11,10**8

bin_x, bin_y = 200, 80
data_11 = np.load("test/ALP-W_LHC_m_"+str(mass_1)+"_c_"+str(coup[0])+".npy")
data_12 = np.load("test/ALP-W_LHC_m_"+str(mass_1)+"_c_"+str(coup[1])+".npy")
data_13 = np.load("test/ALP-W_LHC_m_"+str(mass_1)+"_c_"+str(coup[2])+".npy")
data_14 = np.load("test/ALP-W_LHC_m_"+str(mass_1)+"_c_"+str(coup[3])+".npy")
data_21 = np.load("test/ALP-W_LHC_m_"+str(mass_1)+"_c_"+str(coup[4])+".npy")
data_22 = np.load("test/ALP-W_LHC_m_"+str(mass_1)+"_c_"+str(coup[5])+".npy")
data_23 = np.load("test/ALP-W_LHC_m_"+str(mass_1)+"_c_"+str(coup[6])+".npy")
data_24 = np.load("test/ALP-W_LHC_m_"+str(mass_1)+"_c_"+str(coup[7])+".npy")


plt.figure(figsize=(20,10))
plt.subplots_adjust(left=0.035, bottom=0.06, right=0.99, top=0.95, wspace=0.18, hspace=0.2)
plt.subplot(2,4,1)
plt.hist2d(x=data_11[0],y=data_11[1],weights=data_11[2],bins=[bin_x,bin_y],range=[[-5,0],[0,4]],vmin=min_1, vmax=max_1,norm=matplotlib.colors.LogNorm(), cmap="inferno")
plt.xlabel(r"log10($\theta$)")
plt.ylabel(r"log10(p)")
plt.title("Mass = " + str(mass_1) + " GeV coup = " + str(coup[0]))
# plt.axvline(np.log10(np.arctan(0.1/480)), color = "black", label = "FASER - 10cm")
# plt.legend()
# plt.colorbar()

plt.subplot(2,4,2)
plt.hist2d(x=data_12[0],y=data_12[1],weights=data_12[2],bins=[bin_x,bin_y],range=[[-5,0],[0,4]],vmin=min_1, vmax=max_1,norm=matplotlib.colors.LogNorm(), cmap="inferno")
plt.xlabel(r"log10($\theta$)")
plt.ylabel(r"log10(p)")
plt.title("Mass = "+str(mass_1) + " GeV coup = " + str(coup[1]))
# plt.axvline(np.log10(np.arctan(0.1/480)), color = "black", label = "FASER - 10cm")
# plt.legend()
# plt.colorbar()

plt.subplot(2,4,3)
plt.hist2d(x=data_13[0],y=data_13[1],weights=data_13[2],bins=[bin_x,bin_y],range=[[-5,0],[0,4]],vmin=min_1, vmax=max_1,norm=matplotlib.colors.LogNorm(), cmap="inferno")
plt.xlabel(r"log10($\theta$)")
plt.ylabel(r"log10(p)")
plt.title("Mass = "+str(mass_1) + " GeV coup = " + str(coup[2]))
# plt.axvline(np.log10(np.arctan(0.1/480)), color = "black", label = "FASER - 10cm")
# plt.legend()
# plt.colorbar()

plt.subplot(2,4,4)
plt.hist2d(x=data_14[0],y=data_14[1],weights=data_14[2],bins=[bin_x,bin_y],range=[[-5,0],[0,4]],vmin=min_1, vmax=max_1,norm=matplotlib.colors.LogNorm(), cmap="inferno")
plt.xlabel(r"log10($\theta$)")
plt.ylabel(r"log10(p)")
plt.title("Mass = "+str(mass_1) + " GeV coup = " + str(coup[3]))
# plt.axvline(np.log10(np.arctan(0.1/480)), color = "black", label = "FASER - 10cm")
# plt.legend()
plt.colorbar()



plt.subplot(2,4,5)
plt.hist2d(x=data_21[0],y=data_21[1],weights=data_21[2],bins=[bin_x,bin_y],range=[[-5,0],[0,4]],vmin=min_2, vmax=max_2,norm=matplotlib.colors.LogNorm(), cmap="inferno")
plt.xlabel(r"log10($\theta$)")
plt.ylabel(r"log10(p)")
plt.title("Mass = " + str(mass_1) + " GeV coup = " + str(coup[4]))
# plt.axvline(np.log10(np.arctan(0.1/480)), color = "black", label = "FASER - 10cm")
# plt.legend()
# plt.colorbar()

plt.subplot(2,4,6)
plt.hist2d(x=data_22[0],y=data_22[1],weights=data_22[2],bins=[bin_x,bin_y],range=[[-5,0],[0,4]],vmin=min_2, vmax=max_2,norm=matplotlib.colors.LogNorm(), cmap="inferno")
plt.xlabel(r"log10($\theta$)")
plt.ylabel(r"log10(p)")
plt.title("Mass = "+str(mass_1) + " GeV coup = " + str(coup[5]))
# plt.axvline(np.log10(np.arctan(0.1/480)), color = "black", label = "FASER - 10cm")
# plt.legend()
# plt.colorbar()

plt.subplot(2,4,7)
plt.hist2d(x=data_23[0],y=data_23[1],weights=data_23[2],bins=[bin_x,bin_y],range=[[-5,0],[0,4]],vmin=min_2, vmax=max_2,norm=matplotlib.colors.LogNorm(), cmap="inferno")
plt.xlabel(r"log10($\theta$)")
plt.ylabel(r"log10(p)")
plt.title("Mass = "+str(mass_1) + " GeV coup = " + str(coup[6]))
# plt.axvline(np.log10(np.arctan(0.1/480)), color = "black", label = "FASER - 10cm")
# plt.legend()
# plt.colorbar()

plt.subplot(2,4,8)
plt.hist2d(x=data_24[0],y=data_24[1],weights=data_24[2],bins=[bin_x,bin_y],range=[[-5,0],[0,4]],vmin=min_2, vmax=max_2,norm=matplotlib.colors.LogNorm(), cmap="inferno")
plt.xlabel(r"log10($\theta$)")
plt.ylabel(r"log10(p)")
plt.title("Mass = "+str(mass_1) + " GeV coup = " + str(coup[7]))
# plt.axvline(np.log10(np.arctan(0.1/480)), color = "black", label = "FASER - 10cm")
# plt.legend()
plt.colorbar()
# plt.savefig("Results_Mass_" + str(mass_1)+".png",  dpi=1000, quality=1000, optimize=True, progressive=True)
plt.show()
