import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from multiprocessing import Pool
import matplotlib as mpl
from matplotlib.patches import Rectangle
from operator import itemgetter

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
luminosity=90 #in fb^-1
nsample = 50
model = "ALP-W"
generator = "LHC"
mass = 0.12
coupling = 3.65*10**-4
Coupling_name = "3,65x10^-4"
# Adress="/Users/theo/Desktop/ALP_Simulation/New_code/data/Raw_ALP_Decay/Mass=3x10^-2_Coup=2x10^-3/"
Adress="/Users/theo/Desktop/ALP_Simulation/New_code/data/Peppe_plot/Raw_ALP_Decay/"

def do_convolution():
    self=MyEventCounter(model="ALP-W",generator="LHC",mass=mass,radius=radius, length=length, offset=offset, lumi=luminosity,nphisample=nsample,logcoupmin=np.log10(coupling),logcoupmax=np.log10(coupling),ncoup=0)
    #Load Flux file
    coupling_file = 1
    # filename="data/Raw_ALP_Spectra/Log/"+self.model+"_"+self.generator+"_m_"+str(self.mass)+"_c_"+str(coupling_file)+"sample1000_log_bin1500x1500.npy"
    filename="data/Peppe_plot/Raw_ALP_Spectra/"+self.model+"_"+self.generator+"_m_"+str(self.mass)+"_c_"+str(coupling_file)+".npy"

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

# def unweight_events(energies, angles, weights, number):
#     #initialize arrays and random number generator
#     random.seed()
#     unweighted_angles=[]
#     unweighted_energies=[]
#     unweighted_weights=[]
#     total_weight = np.sum(weights)  #The cumulative distribution function CDF(x) from the PDF(x)
#     event_weight = total_weight/float(number)
#
#     #unweighting
#     for irand in range(number):
#         stopweight = random.uniform(0,1)*total_weight  #Find the various values for which stopweight/total_weight = R (random number in [0,1])
#         partid, weightsum = 0, 0
#         while weightsum < stopweight:
#             weightsum+=weights[partid]
#             partid+=1
#         unweighted_angles.append(angles[partid])
#         unweighted_energies.append(energies[partid])
#         unweighted_weights.append(event_weight)
#
#     return  unweighted_energies, unweighted_angles, unweighted_weights

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
    return Asym, Min, Sep, weight, gamma_1.e, gamma_2.e

# file = open(Adress + "Output_sorted.hepmc","w")
# file.write("HepMC::Version 2.06.09 \n")
# file.write("HepMC::IO_GenEvent-START_EVENT_LISTING \n")






data_x, data_y, data_z, data_e, data_w = [], [], [], [], []
mom_x, mom_y, mom_z = [], [], []
energy_gamma_1, energy_gamma_2, diff_energy_gamma = [], [], []
Data_Asym, Data_e_min_gamma, Data_Sep, Data_w_sep, Data_e1_sep, Data_e2_sep = [], [], [], [], [], []

coups, ctaus, nsigs, energies, weights, angles = do_convolution()

# sum = 0
# tot = np.sum(weights[0])
# count = 0
# energy_unw, angle_unw, weight_unw = [], [], []
# energy_unw, angle_unw, weight_unw = unweight_events(energies[0], angles[0], weights[0], 5000)

for i, (angle, energy, weight) in enumerate(zip(angles[0], energies[0], weights[0])):

    #Event Info
    # file.write("E \n")
    # file.write("N 1 \"0\" \n")
    # file.write("U GEV MM \n")
    # file.write("C "+str(weight)+" 0 \n")

    #Vertex
    phi = random.uniform(-math.pi, math.pi)
    pos_x, pos_y, pos_z = angle*480*np.cos(phi), angle*480*np.sin(phi), random.uniform(0,length)
    pos_t = c*np.sqrt(pos_x**2 + pos_y**2)
    data_x.append(pos_x)
    data_y.append(pos_y)
    data_z.append(pos_z)
    data_e.append(energy)
    data_w.append(weight)

    #Particles
    particles = decay_alp(mass, energy)
    gamma_1, gamma_2 = decay_alp(mass,energy)

    energy_gamma_1.append(gamma_1.e)
    energy_gamma_2.append(gamma_2.e)
    mom_x.append([gamma_1.px, gamma_2.px])
    mom_y.append([gamma_1.py, gamma_2.py])
    mom_z.append([gamma_1.pz, gamma_2.pz])

    diff_energy_gamma.append(np.absolute(gamma_1.e - gamma_2.e))

    Asym, Min, Sep, w_sep, e1_Sep, e2_Sep = Get_separation(pos_z, gamma_1, gamma_2, weight, True, True)
    Data_Asym.append(Asym)
    Data_e_min_gamma.append(Min)
    Data_Sep.append(Sep)
    Data_w_sep.append(w_sep)
    Data_e1_sep.append(e1_Sep)
    Data_e2_sep.append(e2_Sep)


    ## Writting the HepMC output file ###
    # file.write("V - 1")
    # file.write(str(pos_x)+" ")
    # file.write(str(pos_y)+" ")
    # file.write(str(pos_z)+" ")
    # file.write(str(pos_t)+" ")
    # file.write("0 2 0 \n")
    #
    # for particle in particles:
    #     file.write("P " + str(i+1) + " 22 ")
    #     file.write(str(particle.px) + " ")
    #     file.write(str(particle.py) + " ")
    #     file.write(str(particle.pz) + " ")
    #     file.write(str(particle.e) + " ")
    #     file.write("0 1 0 0 0 0 \n")

# file.close()

# print("The weight of each event is: " + str(weight))

filename_1 = Adress + "Data_ALP_raw/" + "Mass_" + str(mass) + "Coup_" + Coupling_name + ".npy"
filename_2 = Adress + "Data_Gamma_raw/" + "Mass_" + str(mass) + "Coup_" + Coupling_name +".npy"
filename_3 = Adress + "Data_Gamma_analysed/" + "Mass_" + str(mass) + "Coup_" + Coupling_name +".npy"
np.save(filename_1, np.array([data_x, data_y, data_z, data_e, data_w]))
np.save(filename_2, np.array([mom_x, mom_y, mom_z, energy_gamma_1, energy_gamma_2, diff_energy_gamma], dtype="object"))
np.save(filename_3, np.array([Data_Asym, Data_e_min_gamma, Data_Sep, Data_w_sep, Data_e1_sep, Data_e2_sep]))
