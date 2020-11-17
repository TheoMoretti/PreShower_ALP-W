import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from multiprocessing import Pool, Lock
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib import rc

matplotlib.rcParams.update({'font.size': 22})
plt.rcParams["font.family"] = "Bitstream Vera Sans Mono"
rc('text', usetex=True)

#Load the data from the convolution of the 3D matrix and the efficiencies from the Geant4 simulations of the detector.
All = np.load("output_20201111_1452.npy", fix_imports = True)

#The list of masses for which the efficiency has been computed
masses = [1.02000000e-001, 1.14000000e-001, 1.26000000e-001, 1.43000000e-001, 1.58000000e-001, 1.76000000e-001, 2.00000000e-001, 2.06000000e-001,
            2.18000000e-001, 2.22000000e-001, 2.32000000e-001, 2.46000000e-001, 2.47000000e-001, 2.62000000e-001, 2.74000000e-001, 2.88000000e-001,
            3.06000000e-001, 3.17000000e-001, 3.41000000e-001, 3.49000000e-001, 3.78000000e-001, 3.81000000e-001, 4.08000000e-001, 4.17000000e-001,
            4.45000000e-001, 4.62000000e-001, 4.84000000e-001, 5.08000000e-001, 5.61000000e-001, 6.12000000e-001, 6.76000000e-001, 7.31000000e-001,
            8.15000000e-001, 8.91000000e-001, 9.86000000e-001, 1.09000000e-000, 1.18000000e-000, 1.28000000e-000, 1.39000000e-000, 1.49000000e-000,
            1.60000000e-000, 1.71000000e-000, 1.82000000e-000, 1.93000000e-000]

#The list of couplings for which the efficiency has been computed
couplings = [8.08000000e-004, 6.84000000e-004, 5.45000000e-004, 4.26000000e-004, 3.16000000e-004, 2.33000000e-004,
            1.81000000e-004, 1.45000000e-004, 1.12000000e-004, 8.97000000e-005, 6.32000000e-005, 5.09000000e-005,
            3.67000000e-005, 2.94000000e-005, 2.83000000e-005, 2.31000000e-005, 1.97000000e-005, 1.78000000e-005,
            1.52000000e-005, 1.44000000e-005, 1.08000000e-005, 1.06000000e-005, 8.22000000e-006, 7.52000000e-006,
            5.52000000e-006, 4.08000000e-006, 2.83000000e-006, 2.11000000e-006, 1.56000000e-006, 1.16000000e-006,
            8.23000000e-007, 6.07000000e-007, 4.71000000e-007]


print(len(masses))
print(len(couplings))

# print(len(All))

#Initialize the arrays in which the data will be appended, each corresponds to a certain separation and has the number of events for each mass and coupling point from the 2 lists above.
evt90_tot, evt3k_tot = np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings)))
evt90_c200, evt90_c300, evt90_c500, evt90_c1k, evt90_c2k = np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings)))
evt3k_c200, evt3k_c300, evt3k_c500, evt3k_c1k, evt3k_c2k = np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings))), np.ndarray(shape = (len(masses), len(couplings)))
evt90_tot.fill(0)
evt3k_tot.fill(0)
evt90_c200.fill(0)
evt90_c300.fill(0)
evt90_c500.fill(0)
evt90_c1k.fill(0)
evt90_c2k.fill(0)
evt3k_c200.fill(0)
evt3k_c300.fill(0)
evt3k_c500.fill(0)
evt3k_c1k.fill(0)
evt3k_c2k.fill(0)

#Appending the efficiencies to a matrix with one axis for the masses, the other for the couplings and order them according to the order of the mass and couplings lists above.
the_count = 0
for i in range(len(masses)):
    for j in range(len(couplings)):
        for k in range(len(All)):
            if float(All[k][1]) == masses[i] and float(All[k][0]) == couplings[j]:
                # print("it works")
                # print(masses[i])
                # print(couplings[j])
                evt90_tot[i][j] = All[k][2]
                evt3k_tot[i][j] = All[k][3]
                evt90_c200[i][j] = All[k][4]
                evt90_c300[i][j] = All[k][5]
                evt90_c500[i][j] = All[k][6]
                evt90_c1k[i][j] = All[k][7]
                evt90_c2k[i][j] = All[k][8]
                evt3k_c200[i][j] = All[k][9]
                evt3k_c300[i][j] = All[k][10]
                evt3k_c500[i][j] = All[k][11]
                evt3k_c1k[i][j] = All[k][12]
                evt3k_c2k[i][j] = All[k][13]
                the_count += 1


def readfile(filename):
    list_of_lists = []
    with open(filename) as f:
        for line in f:
            if line[0]=="#":continue
            inner_list = [float(elt.strip()) for elt in line.split( )]
            list_of_lists.append(inner_list)
    return np.array(list_of_lists)


plt.figure(figsize=(17,13))
plt.subplots_adjust(left=0.1, bottom=0.08, right=0.95, top=0.96, wspace=0.2, hspace=0.2)

bounds=["SN1987","E137","NuCal","LEP","E949_displ","NA62_1","NA62_2","KOTO","KTEV","NA6264","E949_prompt","CDF","PbPb",]

limits=[["Belle2-3gamma", "dashed", "royalblue",   0],
        ["KOTO-2gamma",   "dashed", "cyan",        0],
        ["KOTO-4gamma",   "dashed", "blue",        0],
        ["NA62-0gamma1",  "dashed", "dodgerblue",  0],
        ["NA62-0gamma2",  "dashed", "dodgerblue",  0],
        ["NA62-2gamma",   "dashed", "deepskyblue", 0],
        ["LHC",           "dashed", "teal",        0],]
zorder=-100
for bound in bounds:
    bound_file=readfile("/Users/theo/Desktop/ALP_Simulation/New_code/Ressources/bounds/bounds_"+bound+".txt")
    plt.fill(bound_file.T[0], bound_file.T[1], color="gainsboro",zorder=zorder)
    plt.plot(bound_file.T[0], bound_file.T[1], color="dimgray"  ,zorder=zorder,lw=1)
    zorder +=1

plt.text(1, 6.7*10**-4, "LEP", fontsize=12,color="dimgray",rotation=0)
plt.text(0.065, 7.5*10**-4, "CDF", fontsize=12,color="dimgray",rotation=-12)
plt.text(0.200, 5*10**-4, "KTEV", fontsize=12,color="dimgray",rotation=90)
plt.text(0.235, 5*10**-4, "NA62", fontsize=12,color="dimgray",rotation=90)
plt.text(0.270, 5*10**-4, "+NA48/2", fontsize=12,color="dimgray",rotation=90)
plt.text(0.065, 9.0*10**-5, "E949", fontsize=12,color="dimgray",rotation=-9)
plt.text(0.090, 3.4*10**-5, "KOTO", fontsize=12,color="dimgray",rotation=9)
plt.text(0.065, 9.2*10**-6, "NA62", fontsize=12,color="dimgray",rotation=2)
plt.text(0.065, 3.0*10**-6, "E949", fontsize=12,color="dimgray",rotation=-5)
plt.text(0.100, 1.2*10**-6, "E137", fontsize=12,color="dimgray",rotation=-8)
plt.text(0.100, 1.9*10**-7, "SN1987", fontsize=12,color="dimgray",rotation=25)


#Plot the contour lines for each of the different separations and the data with efficiency == 1 for each luminosity (either 90 fb^-1 or 3 ab^-1)
plt.contour(masses, couplings, evt90_tot.T, levels = [3], extend = "max", colors =["#f8481c"])
plt.contour(masses, couplings, evt90_c200.T, levels = [3], extend = "max", colors =["black"], linestyles = ["solid"])
plt.contour(masses, couplings, evt90_c300.T, levels = [3], extend = "max", colors =["black"], linestyles = ["dotted"])
plt.contour(masses, couplings, evt90_c500.T, levels = [3], extend = "max", colors =["black"], linestyles = ["dashed"])
plt.contour(masses, couplings, evt90_c1k.T, levels = [3], extend = "max", colors =["black"], linestyles = ["dashdot"])

plt.contour(masses, couplings, evt3k_tot.T, levels = [3], extend = "max", colors =["#75bbfb"])
plt.contour(masses, couplings, evt3k_c200.T, levels = [3], extend = "max", colors =["dimgray"], linestyles = ["solid"])
plt.contour(masses, couplings, evt3k_c300.T, levels = [3], extend = "max", colors =["dimgray"], linestyles = ["dotted"])
plt.contour(masses, couplings, evt3k_c500.T, levels = [3], extend = "max", colors =["dimgray"], linestyles = ["dashed"])
plt.contour(masses, couplings, evt3k_c1k.T, levels = [3], extend = "max", colors =["dimgray"], linestyles = ["dashdot"])



plt.xlim(6*10**-2,2)
plt.ylim(10**-7,0.002)
plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$m_a$ [GeV]")
plt.ylabel("$g_{AWW}$ [GeV$^{-1}$]")

#Manually create the legend since the function "contour" doesn't have a parameter for labelling the contours.
plt.plot([10**-10], [10**-10], color="#75bbfb", linestyle = "solid",label=r"\textbf{FASER} L$_{int}$ = 3 ab$^{-1}$")
plt.plot([10**-10], [10**-10], color="dimgray", linestyle = "solid",label=r"      $\delta$ $\geq$ 200 $\mu m$")
plt.plot([10**-10], [10**-10], color="dimgray", linestyle = "dotted",label=r"     $\delta$ $\geq$ 300 $\mu m$")
plt.plot([10**-10], [10**-10], color="dimgray", linestyle = "dashed",label=r"     $\delta$ $\geq$ 500 $\mu m$")
plt.plot([10**-10], [10**-10], color="dimgray", linestyle = "dashdot", label=r"       $\delta$ $\geq$ 1000 $\mu m$")

plt.plot([10**-10], [10**-10], color="#f8481c", linestyle = "solid",label=r"\textbf{FASER} L$_{int}$ = 90 fb$^{-1}$")
plt.plot([10**-10], [10**-10], color="black", linestyle = "solid",label=r"      $\delta$ $\geq$ 200 $\mu m$")
plt.plot([10**-10], [10**-10], color="black", linestyle = "dotted",label=r"     $\delta$ $\geq$ 300 $\mu m$")
plt.plot([10**-10], [10**-10], color="black", linestyle = "dashed",label=r"     $\delta$ $\geq$ 500 $\mu m$")
plt.plot([10**-10], [10**-10], color="black", linestyle = "dashdot", label=r"       $\delta$ $\geq$ 1000 $\mu m$")

plt.legend(loc=[0.52, 0.69], ncol=2, title="", fancybox=True, prop={'size': 17.5})
plt.savefig("Reach_plot.pdf",  dpi=1000)
plt.show()


# plt.xlim(6*10**-2,1.5)
# plt.ylim(10**-7,0.002)
# plt.xscale("log")
# plt.yscale("log")
# plt.xlabel(r"$m_a$ [GeV]")
# plt.ylabel("$g_{AWW}$ [GeV$^{-1}$]")
#
#
# plt.plot([10**-10], [10**-10], color="#75bbfb", linestyle = "solid",label=r"\textbf{FASER} L$_{int}$ = 3 ab$^{-1}$")
# # plt.plot([10**-10], [10**-10], color="dimgray", linestyle = "solid",label=r"      $\delta$ $\geq$ 200 $\mu m$")
# # plt.plot([10**-10], [10**-10], color="dimgray", linestyle = "dotted",label=r"     $\delta$ $\geq$ 300 $\mu m$")
# # plt.plot([10**-10], [10**-10], color="dimgray", linestyle = "dashed",label=r"     $\delta$ $\geq$ 500 $\mu m$")
# # plt.plot([10**-10], [10**-10], color="dimgray", linestyle = "dashdot", label=r"       $\delta$ $\geq$ 1000 $\mu m$")
#
# plt.plot([10**-10], [10**-10], color="#f8481c", linestyle = "solid",label=r"\textbf{FASER} L$_{int}$ = 90 fb$^{-1}$")
# plt.plot([10**-10], [10**-10], color="black", linestyle = "solid",label=r"      $\delta$ $\geq$ 200 $\mu m$")
# plt.plot([10**-10], [10**-10], color="black", linestyle = "dotted",label=r"     $\delta$ $\geq$ 300 $\mu m$")
# plt.plot([10**-10], [10**-10], color="black", linestyle = "dashed",label=r"     $\delta$ $\geq$ 500 $\mu m$")
# plt.plot([10**-10], [10**-10], color="black", linestyle = "dashdot", label=r"       $\delta$ $\geq$ 1000 $\mu m$")
#
# new_mass = [1.02*10**-1, 1.26*10**-1, 1.58*10**-1, 2*10**-1, 2.46*10**-1, 3.06*10**-1, 3.78*10**-1, 4.45*10**-1, 1.14*10**-1, 1.43*10**-1, 1.76*10**-1, 2.22*10**-1, 2.74*10**-1, 3.41*10**-1, 4.08*10**-1, 4.84*10**-1, 2.06*10**-1, 2.18*10**-1, 2.32*10**-1, 2.47*10**-1, 2.62*10**-1, 2.88*10**-1, 3.17*10**-1, 3.49*10**-1, 3.81*10**-1, 4.17*10**-1, 4.62*10**-1, 5.08*10**-1, 5.61*10**-1, 6.12*10**-1, 6.76*10**-1, 7.31*10**-1, 8.15*10**-1, 8.91*10**-1, 9.86*10**-1, 1.09, 1.18, 1.28, 1.39, 1.49, 1.6, 1.71, 1.82, 1.93]
# new_coup = [8.08*10**-4, 6.84*10**-4,5.45*10**-4, 3.16*10**-4, 1.81*10**-4, 1.12*10**-4, 6.32*10**-5, 3.67*10**-5, 1.97*10**-5, 1.08*10**-5, 4.26*10**-4, 2.33*10**-4, 1.45*10**-4, 8.97*10**-5, 5.09*10**-5, 2.83*10**-5, 1.52*10**-5, 8.22*10**-6, 2.94*10**-5, 2.31*10**-5, 1.78*10**-5, 1.44*10**-5, 1.06*10**-5, 7.52*10**-6, 5.52*10**-6, 4.08*10**-6, 2.83*10**-6, 2.11*10**-6, 1.56*10**-6, 1.16*10**-6, 8.23*10**-7, 6.07*10**-7, 4.71*10**-7]
#
# # for i in range(len(new_mass)):
# #     plt.axvline(new_mass[i], color = "black")
# #
# # for j in range(len(new_coup)):
# #     plt.axhline(new_coup[j], color = "black")
#
#
# plt.legend(loc=[0.7, 0.64], ncol=1, title="", fancybox=True, prop={'size': 19})
# plt.savefig("Reach_plot.pdf",  dpi=1000)
# plt.show()
