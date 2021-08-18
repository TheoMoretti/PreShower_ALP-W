import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from multiprocessing import Pool
import matplotlib as mpl
from matplotlib.patches import Rectangle

sys.path.append("/Users/theo/Desktop/ALP_Simulation/New_code/Modules")

from various import *
from br_validation import *
from llpmodel import LLPModel
from createllp import CreateLLP
from eventcounter import MyEventCounter
from llpgenerator import LLPGenerator

# IMPORTANT NOTE: The coupling for the input files has to be 1 so that it has no effect in the previous computations on the weight of the events and hence the results of the simulations done in what follows.
c = 299792458
radius=0.1 #in meter
length=3.5 #in meter
offset=0 #in meter
luminosity=150 #in fb^-1
nsample = 40
model = "ALP-W"
generator = "LHC"
mass = 0.421
coupling = 1*10**-5
Adress="/Users/theo/Desktop/ALP_Simulation/New_code/Results/Thesis_Plots/"

def do_convolution():
    self=MyEventCounter(model="ALP-W",generator="LHC",mass=mass,radius=radius, length=length, offset=offset, lumi=luminosity,nphisample=nsample,logcoupmin=np.log10(coupling),logcoupmax=np.log10(coupling),ncoup=1)
    #Load Flux file
    coupling_file = 1
    # /Users/theo/Desktop/ALP_Simulation/New_code/data/Raw_ALP_Spectra/ALPW_scale/ALP-W_LHC_m_0.3_c_1.npy
    # /Users/theo/Desktop/ALP_Simulation/New_code/data/Raw_ALP_Decay/ALPW_scale/Mass=0.03_Coup=2x10^-3
    # filename="data/Log/"+self.model+"_"+self.generator+"_m_"+str(self.mass)+"_c_"+str(coupling_file)+"sample1000_log_bin1500x1500.npy"
    filename = "/Users/theo/Desktop/ALP_Simulation/New_code/data/Raw_ALP_Spectra/Log/ALP-W_LHC_m_0.1_c_1sample1000_log_bin1500x1500.npy"
    particles_llp, weights_llp=convert_list_to_momenta_sampled_log(filename=filename,mass=self.mass,filetype="npy",nsample=self.nphisample,preselectioncut=self.preselectioncuts) #self.preselectioncuts

    for p,w in zip(particles_llp,weights_llp):
        #check if event passes
        if not self.event_passes(p): continue
        #weight of this event
        weight_event = w*self.lumi

        #loop over couplings
        for icoup,coup in enumerate(self.couplings):
            #add event weight
            ctau=self.ctaus[icoup]
            dbar=ctau*p.p/self.mass
            prob_decay=math.exp(-self.distance/dbar)-math.exp(-(self.distance+self.length)/dbar)
            self.nsignals[icoup]+=weight_event* coup**2 * prob_decay * self.br
            self.stat_t[icoup].append(p.pt/p.pz)
            self.stat_e[icoup].append(p.e)
            self.stat_w[icoup].append(weight_event* coup**2 *prob_decay * self.br)

    return self.couplings, self.ctaus, self.nsignals, self.stat_e, self.stat_w, self.stat_t

def decay_alp(mass,energy):

    #Randomly choose angles of decay in the ALP rest frame
    costh = random.uniform(-1,1)
    phi = random.uniform(-math.pi,math.pi)

    #4-momentum of p1 and p2 in ALP rest frame
    pz = mass/2. * costh
    py = mass/2. * math.sqrt(1-costh*costh) * np.sin(phi)
    px = mass/2. * math.sqrt(1-costh*costh) * np.cos(phi)
    p1 = LorentzVector( px, py, pz,mass/2)
    p2 = LorentzVector(-px,-py,-pz,mass/2)

    #Boost the decay products in lab frame
    xxx=math.sqrt(energy**2-mass**2)
    p0 = LorentzVector(0,0,math.sqrt(energy**2-mass**2),energy)
    p1_=p1.boost(-1*p0.boostvector)
    p2_=p2.boost(-1*p0.boostvector)

    return p1_, p2_

def Get_separation(z, gamma_1, gamma_2, weight, Asym, Less_e):
    norm_g1, norm_g2 = np.sqrt((gamma_1.px**2 + gamma_1.py**2 + gamma_1.pz**2)), np.sqrt((gamma_2.px**2 + gamma_2.py**2 + gamma_2.pz**2))
    theta_1, theta_2 = np.arccos((gamma_1.pz)/norm_g1), np.arccos((gamma_2.pz)/norm_g2)
    Sep = (np.tan(theta_1) + np.tan(theta_2))*(3.5-z)*1000
    if Asym == True:
        Asym = (np.abs(gamma_1.e-gamma_2.e))/(gamma_1.e+gamma_2.e)
    else: Asym = 0
    if Less_e == True:
        Min = np.minimum(gamma_1.e,gamma_2.e)
    else: Min = 0
    return Asym, Min, Sep, weight

# file = open("/Users/theo/Desktop/ALP_Simulation/New_code/Ressources/output.hepmc","w")
# file.write("HepMC::Version 2.06.09 \n")
# file.write("HepMC::IO_GenEvent-START_EVENT_LISTING \n")

data_x, data_y, data_z, data_e, data_w = [], [], [], [], []
energy_gamma_1, energy_gamma_2, diff_energy_gamma = [], [], []
Data_Asym, Data_e_min_gamma, Data_Sep, Data_w_sep = [], [], [], []
energy_gamma_1_sep, energy_gamma_2_sep = [], []
Data_weight_Sep_cond = []

coups, ctaus, nsigs, energies, weights, angles = do_convolution()

for i, (angle, energy, weight) in enumerate(zip(angles[0], energies[0], weights[0])):
    #Event Info
    # file.write("E \n")
    # file.write("N 1 \"0\" \n")
    # file.write("U GEV MM \n")
    # file.write("C "+str(weight)+" 0 \n")

    #Vertex
    phi = random.uniform(-math.pi, math.pi)
    pos_x, pos_y, pos_z = angle*480*np.cos(phi), angle*480*np.sin(phi), random.uniform(0,length)
    pos_t = c*np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
    data_x.append(pos_x)
    data_y.append(pos_y)
    data_z.append(pos_z)
    data_e.append(energy)
    data_w.append(weight)

    #Particles
    # particles = decay_alp(mass, energy)
    gamma_1, gamma_2 = decay_alp(mass,energy)

    energy_gamma_1.append(gamma_1.e)
    energy_gamma_2.append(gamma_2.e)

    diff_energy_gamma.append(np.absolute(gamma_1.e - gamma_2.e))

    Asym, Min, Sep, w_sep = Get_separation(pos_z, gamma_1, gamma_2, weight, True, True)
    Data_Asym.append(Asym)
    Data_e_min_gamma.append(Min)
    Data_Sep.append(Sep)
    Data_w_sep.append(w_sep)

    if Sep>0.2:
        Data_weight_Sep_cond.append(weight)
        energy_gamma_1_sep.append(gamma_1.e)
        energy_gamma_2_sep.append(gamma_2.e)


    ### Writting the HepMC output file ###
    # file.write("V - 1")
    # file.write(str(pos_x)+" ")
    # file.write(str(pos_y)+" ")
    # file.write(str(pos_z)+" ")
    # file.write(str(pos_t)+" ")
    # file.write("0 2 0 \n")

    # for particle in particles:
    #     file.write("P " + str(i+1) + " 22 ")
    #     file.write(str(particle.px) + " ")
    #     file.write(str(particle.py) + " ")
    #     file.write(str(particle.pz) + " ")
    #     file.write(str(particle.e) + " ")
    #     file.write("0 1 0 0 0 0 \n")

# file.close()

# data_e_log=[np.log10(e) for e in data_e]

hist_spc, x_edges_spc, y_edges_spc = np.histogram2d(data_x,data_y,weights=data_w,bins=[80,80], range=[[-0.15,0.15],[-0.15,0.15]])
def Get_better_radial(j, x=data_x, y=data_y, w=data_w, xedg=x_edges_spc, yedg=y_edges_spc):
    # print(j)
    data_r, data_w = 0, 0
    for i in range(len(xedg)-1):
        if np.sqrt(x[j]**2 + y[j]**2) < np.sqrt(xedg[i+1]**2 + yedg[i+1]**2) and np.sqrt(x[j]**2 + y[j]**2) > np.sqrt(xedg[i]**2 + yedg[i]**2):
            data_r=np.sqrt(x[j]**2+y[j]**2)
            if data_r != 0:data_w=w[j]/data_r
    return data_r, data_w

hist_dst, x_edges_dst, y_edges_dst = np.histogram2d(data_z, data_e, weights=data_w, bins =[80,50], range=[[0,length],[0,5000]])
def Decay_dist_sum(j, z=data_z, e=data_e, w=data_w, zedg=x_edges_dst, eedg=y_edges_dst):
    # print(j)
    data_e, data_w = 0, 0
    for i in range(len(eedg)-1):
        if e[j] < eedg[i+1] and e[j] > eedg[i]:
            data_w = w[j]
            data_e = e[j]
    return data_e, data_w
nb_sep = 3
sep_lim = [[0.2,0.3],[0.5,0.6],[0.5,3]]

def Select_event_sep_energy(j, eg_1 = Data_Asym, sep = Data_Sep , weight = Data_w_sep, sep_lim=sep_lim, nb_sep=nb_sep, e_lim=None):
    data_e, data_s, data_w = [0,0,0], [0,0,0], [0,0,0]
    for i in range(nb_sep):
        if sep_lim[i] is None and e_lim is None:
            data_e[i], data_s[i], data_w[i] = eg_1[j], sep[j], weight[j]

        elif sep_lim is None and e_lim is not None:
            if eg_1[j] > e_lim[i][0] and eg_1[j] < e_lim[i][1]:
                data_e[i], data_s[i], data_w[i] = eg_1[j], sep[j], weight[j]

        elif sep_lim is not None and e_lim is None:
            if sep[j] > sep_lim[i][0] and sep[j] < sep_lim[i][1]:
                data_e[i], data_s[i], data_w[i] = eg_1[j], sep[j], weight[j]
        else:
            if sep[j] > sep_lim[i][0] and sep[j] < sep_lim[i][1] and eg_1[j] > e_lim[i][0] and eg_1[j] < e_lim[i][1]:
                data_e[i], data_s[i], data_w[i] = eg_1[j], sep[j], weight[j]

    return data_e, data_s, data_w

Data_r, Data_w_spc = [], []
Data_e, Data_w_dst = [], []

idx_spc = np.arange(0,len(data_x),1)
idx_dst = np.arange(0, len(data_e),1)
idx_sep = np.arange(0, len(Data_Sep),1)

if __name__ == "__main__":
    p = Pool(12)

    # print("Started for spatial distribution")
    # res_spc=p.map(Get_better_radial, idx_spc)
    # for [r,w] in res_spc:
    #     Data_r.append(r)
    #     Data_w_spc.append(w)
    # print("Finished for spatial distribution")

    print("Started for ALP energy spectrum")
    res_dst = p.map(Decay_dist_sum, idx_dst)
    for [e,w] in res_dst:
        Data_e.append(e)
        Data_w_dst.append(w)
    print("Finished for ALP energy spectrum")

    print("Started for energy or separation projection")
    res_sep = p.map(Select_event_sep_energy, idx_sep)
    for i in range(nb_sep):
        Data_e_select, Data_sep_select, Data_w_select = [], [], []
        for [e,s,w] in res_sep:
            Data_e_select.append(e[i])
            Data_sep_select.append(s[i])
            Data_w_select.append(w[i])
        plt.figure(figsize=(16,9))
        plt.hist(Data_e_select, weights = Data_w_select, color = "red", range = [0,1], bins = 30)
        plt.xlabel("Asymmetry of the energy of the Gammas")
        plt.ylabel("a.u.")
        # plt.savefig(Adress+"Distribution_btw"+str(sep_lim[i][0])+"and"+str(sep_lim[i][1])+".png", dpi=1000,  optimize=True, progressive=True)
        plt.close()
    print("Finished for energy or separation projection")


################################################################    PLOTTING REGION    ################################################################


# plt.subplots_adjust(left=0.06, bottom=0.08, right=0.96, top=0.95, wspace=0.18, hspace=0.2)

################    Plot 2D Distributions   ################
blank = Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)
matplotlib.rcParams.update({'font.size': 22})
# # plt.subplot(1,3,1)
plt.figure(figsize=(14,11))
plt.hist2d(data_x, data_y ,weights = data_w, range = [[-0.15,0.15],[-0.15,0.15]], bins = [60,60], cmap="viridis")
plt.xlabel("x [m]")
plt.ylabel("y [m]")
# plt.title("ALP decay position at PS")     #r"$m_a="+str(mass)+"$ GeV, $g_{aWW}="+str(coupling)+"/$GeV"
plt.axvline(x=0, color = "black")
plt.axhline(y=0, color = "black")
plt.colorbar()
# plt.legend([blank], ["Number of events " + str(round(np.sum(data_w)))])
plt.savefig(Adress+"ALP_decay_position_at_PS_smear40.png",  dpi=500)
plt.show()
# plt.close()
# #
# # plt.subplot(1,3,2)
# plt.figure(figsize=(16,10))
# plt.hist2d(data_z,data_e,weights=data_w,range=[[0,length],[0,2500]],bins=[80,50], cmap="viridis")
# plt.xlabel("z [m]")
# plt.ylabel("ALP energy [GeV]")
# plt.title("Decay distance vs. Energy of ALP")
# plt.colorbar()
# plt.legend([blank], ["Number of entries: " + str(round(np.sum(data_w)))])
# plt.show()
# # plt.savefig(Adress+"Decay_dist_vs_energy_ALP.png",  dpi=1000,  optimize=True, progressive=True)
# plt.close()
#
# # plt.subplot(1,3,3)
# plt.figure(figsize=(10,8))
# plt.hist2d(energy_gamma_1, energy_gamma_2, weights=data_w,range=[[0,1750],[0,1750]], bins=[50,50], cmap="viridis")
# plt.xlabel("Energy of Gamma 1 [GeV]")
# plt.ylabel("Energy of Gamma 2 [GeV]")
# plt.title("2D distribution of photons energy ")
# plt.colorbar()
# plt.legend([blank], ["Number of entries: " + str(round(np.sum(data_w)))])
# plt.show()
# # plt.savefig(Adress+"2D_distribution_energy_gamma.png",  dpi=1000,  optimize=True, progressive=True)
# plt.close()
#
# plt.figure(figsize=(10,8))
# plt.hist2d(energy_gamma_1_sep, energy_gamma_2_sep, weights=Data_weight_Sep_cond,range=[[0,1750],[0,1750]], bins=[50,50], cmap="viridis")
# plt.xlabel("Energy of Gamma 1 [GeV]")
# plt.ylabel("Energy of Gamma 2 [GeV]")
# plt.title(r"2D distribution of photons energy - sep>200$\mu$m")
# plt.colorbar()
# plt.legend([blank], ["Number of entries: " + str(round(np.sum(Data_weight_Sep_cond)))])
# plt.show()
# # plt.savefig(Adress+"2D_distribution_energy_gamma_Sep_cond.png",  dpi=1000,  optimize=True, progressive=True)
# plt.close()
#
# plt.figure(figsize=(10,8))
# plt.hist2d(Data_Asym, Data_Sep, weights = Data_w_sep, range = [[0,1],[0.2,1]], bins = [20,20])
# plt.xlabel("Asymmetry of the photon pair energy")
# plt.ylabel("Separation [mm]")
# plt.title("Separation of photons vs. Energy asymmetry")
# plt.colorbar()
# plt.legend([blank], ["Number of entries: " + str(round(np.sum(Data_w_sep)))])
# plt.show()
# # plt.savefig(Adress+"2D_distribution_Asym_Sep.png",  dpi=1000,  optimize=True, progressive=True)
# plt.close()
#
# plt.figure(figsize=(10,8))
# plt.hist2d(Data_e_min_gamma, Data_Sep, weights = Data_w_sep, range = [[0,2000],[0.2,1]], bins = [20,20])
# plt.xlabel("Energy of less energetic gamma [GeV]")
# plt.ylabel("Separation [mm]")
# plt.title("Separation of photons vs. Minimal energy")
# plt.colorbar()
# plt.legend([blank], ["Number of entries: " + str(round(np.sum(Data_w_sep)))])
# # plt.show()
# plt.savefig(Adress+"2D_distribution_e_min_Sep.png",  dpi=1000,  optimize=True, progressive=True)
# plt.close()

################    Plot 1D Distributions   ################

# plt.subplot(2,3,1)
# plt.figure(figsize=(16,9))
# plt.hist(Data_r, weights=Data_w_spc,range=[0,0.15], bins=80, color="red", label = "Number of entries: " + str(round(np.sum(Data_w_spc))))
# plt.xlabel("radius [m]")
# plt.ylabel("a.u.")
# plt.title("Radial distribution")
# plt.legend()
# plt.show()
# # plt.savefig(Adress+"Radial_distribution_decay_pos.png",  dpi=1000,  optimize=True, progressive=True)
# plt.close()

# plt.subplot(2,3,2)
plt.figure(figsize=(16,9))
plt.hist(Data_e, weights = Data_w_dst, range=[0,5000], bins = 100, color="red", label = "Number of entries: " + str(round(np.sum(Data_w_dst))))
plt.xlabel("Energy of ALP [GeV]")
plt.ylabel("Number of events")
# plt.title("Energy distribution of ALPs")
# plt.legend()
plt.savefig(Adress+"Energy_spectrum_ALP_smear40.png",  dpi=500)
plt.show()
# plt.close()

# plt.subplot(2,3,3)
# plt.figure(figsize=(16,9))
# plt.hist(diff_energy_gamma,weights=data_w,range=[0,4000], bins = 80, color = "red", label = "Number of entries: " + str(round(np.sum(data_w))))
# plt.xlabel("Energy difference [GeV]")
# plt.ylabel("a.u.")
# plt.title("Difference of photon's energy")
# plt.legend()
# plt.show()
# # plt.savefig(Adress+"Gamma_e_diff_distribution.png",  dpi=1000,  optimize=True, progressive=True)
# plt.close()
#
# # plt.subplot(2,3,4)
#
#
# # plt.subplot(2,3,5)
# plt.figure(figsize=(16,9))
# plt.hist(Data_Sep, weights = Data_w_sep, range = [0,1], bins = 100, color = "red",  label = "Number of entries: " + str(round(np.sum(Data_w_sep))))
# plt.xlabel("Separation [mm]")
# plt.ylabel("a.u.")
# plt.title("Distribution of separation of photons")
# plt.legend()
# plt.show()
# # plt.savefig(Adress+"Photon_sep_distribution.png",  dpi=1000,  optimize=True, progressive=True)
# plt.close()


# plt.subplot(2,3,6)
# plt.hist(data_w, range = [0,np.max(data_w)], bins = 100, color = "red")
# plt.xlabel("Weights")
# plt.ylabel("a.u.")
# plt.yscale("log")
# plt.title("Weights distribution")







# plt.savefig("Distributions.png",  dpi=1000,  optimize=True, progressive=True)
