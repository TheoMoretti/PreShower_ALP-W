import numpy as np
import sys
import argparse
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from multiprocessing import Pool
import matplotlib as mpl
from matplotlib.patches import Rectangle
from operator import itemgetter

#Insert the path of the different modules (classes with functions)
sys.path.append("/Users/theo/Desktop/ALP_Simulation/New_code/Modules")

from various import *
from br_validation import *
from llpmodel import LLPModel
from createllp import CreateLLP
from eventcounter import MyEventCounter
from llpgenerator import LLPGenerator


# IMPORTANT NOTE: The coupling for the input files has to be 1 so that it has no effect in the previous computations on the Cross Section of the events and hence the results of the simulations done in what follows.

#Specify the parameters of the FASER detector
c = 299792458
radius=0.1 #in meter
length=3.5 #in meter, length of the decay volume.
offset=0 #in meter, offset from the Interaction Point line of sight.
luminosity=90 #in fb^-1, integrated luminosity for the run considered.
nsample = 25 #This parameter is used to smear the discretized distributions from the code generating the ALP spectra.
model = "ALP-W"
generator = "LHC"

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


Couplings = []
Couplings_names = []

#The code is run over different samples of couplings for limitation of the copmuting power (16Go of RAM), the number od elements in each list can be increased with better computing power or smaller value for the "nsample" variable.
if args.index == 1:
    Couplings = Couplings_1
    Couplings_names = Couplings_names_1

if args.index == 2:
    Couplings = Couplings_2
    Couplings_names = Couplings_names_2

if args.index == 3:
    Couplings = Couplings_3
    Couplings_names = Couplings_names_3

if args.index == 4:
    Couplings = Couplings_4
    Couplings_names = Couplings_names_4

if args.index == 5:
    Couplings = Couplings_5
    Couplings_names = Couplings_names_5

if args.index == 6:
    Couplings = Couplings_6
    Couplings_names = Couplings_names_6

mass = args.mass

#Path for the input files which are the ALP spectra for a specific mass.
Adress="/Users/theo/Desktop/ALP_Simulation/New_code/data/Discovery_potential/Raw_ALP_Decay/"

def do_convolution(coupling):
    self=MyEventCounter(model="ALP-W",generator="LHC",mass=mass,radius=radius, length=length, offset=offset, lumi=luminosity,nphisample=nsample,logcoupmin=np.log10(coupling),logcoupmax=np.log10(coupling),ncoup=0)
    #Load Flux file
    coupling_file = 1
    filename="data/Discovery_potential/Raw_ALP_Spectra/"+self.model+"_"+self.generator+"_m_"+str(self.mass)+"_c_"+str(coupling_file)+".npy"
    #Converting the 3 input lists into 2 lists with only momenta and Cross Section, to get a smoother distribution, the distribution in momenta is smeared "nsample" times.
    particles_llp, weights_llp=convert_list_to_momenta_sampled_log(filename=filename,mass=self.mass,filetype="npy",nsample=self.nphisample,preselectioncut=self.preselectioncuts) #self.preselectioncuts
    for p,w in zip(particles_llp,weights_llp):
        #check if event passes in the sense if the events are energetic enough to survive until the FASER detector but also inside the acceptance of the detector.
        if not self.event_passes(p): continue
        #weight of this Monte Carlo event
        weight_event = w*self.lumi

        #loop over couplings (Here its done only once to parallelize the process, see further).
        for icoup,coup in enumerate(self.couplings):
            #add event weight and copmute the relevant quantities to the ALPs
            ctau=self.ctaus[icoup]
            dbar=ctau*p.p/self.mass
            prob_decay=math.exp(-self.distance/dbar)-math.exp(-(self.distance+self.length)/dbar)
            self.nsignals[icoup]+=weight_event* coup**2 * prob_decay * self.br          #Computing the total number of events over the loop
            self.stat_t[icoup].append(p.pt/p.pz)                                        #Append the angles of ALP with respect to Interaction Point line of sight
            self.stat_e[icoup].append(p.e)                                              #Append the energy of the ALP
            self.stat_w[icoup].append(weight_event* coup**2 *prob_decay * self.br)      #Append the number of events associated to each Monte Carlo event from the ALP spectra.

    return self.couplings, self.ctaus, self.nsignals, self.stat_e, self.stat_w, self.stat_t

#Function which decays the ALP in their rest frame to 2 photons and boosts it back in the lab frame, returns the momentum of the 2 photons.
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

#Function which returns 6 lists with respectively, the asymmetry in energy of the 2 photons, the enegy of the less energetic photon, the separation at the end of the decay volume of the 2 photons, number of events and energy of each photon.
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

#This functions uses the 3 previously defined functions and uses them to read the ALP spectra, associate for each Minte Carlo event the number of events associated, decay the ALP into 2 photons and compute many quantities relative to the decay, 3 lists of lists as output give these quantities:
#1st list: position x, position y, position z, energy of ALP and number of events.
#2nd list: momentum x, momentum y, momentum z, energy of photon 1, energy of photon 2, energy difference of the 2 photons.
#3rd list: Asymmetry in energy of the photons, energy of less energetic photon, separation between the 2 photons, number of events, energy of photon 1, energy of photon 2.
def Get_output(coup):

    data_x, data_y, data_z, data_e, data_w = [], [], [], [], []
    mom_x, mom_y, mom_z = [], [], []
    energy_gamma_1, energy_gamma_2, diff_energy_gamma = [], [], []
    Data_Asym, Data_e_min_gamma, Data_Sep, Data_w_sep, Data_e1_sep, Data_e2_sep = [], [], [], [], [], []

    coups, ctaus, nsigs, energies, weights, angles = do_convolution(coup)


    for i, (angle, energy, weight) in enumerate(zip(angles[0], energies[0], weights[0])):

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

    return coup, [data_x, data_y, data_z, data_e, data_w], [mom_x, mom_y, mom_z, energy_gamma_1, energy_gamma_2, diff_energy_gamma], [Data_Asym, Data_e_min_gamma, Data_Sep, Data_w_sep, Data_e1_sep, Data_e2_sep]

#This code is optimised to run in parallel, here the code can compute all of the mentionned above quantites for one specific mass but different coupling points at the same time.
if __name__ == "__main__":
    p = Pool(12)
    res = p.map(Get_output, Couplings)
    idx = 0
    for [coup, [data_x, data_y, data_z, data_e, data_w], [mom_x, mom_y, mom_z, energy_gamma_1, energy_gamma_2, diff_energy_gamma], [Data_Asym, Data_e_min_gamma, Data_Sep, Data_w_sep, Data_e1_sep, Data_e2_sep]] in res:
        #The 3 lists mentionned above are saved in 3 different output files in the "npy" format to avoid reading issues from another device.
        filename_1 = Adress + "Data_ALP_raw/" + "Mass_" + str(mass) + "Coup_" + Couplings_names[idx] + ".npy"
        filename_2 = Adress + "Data_Gamma_raw/" + "Mass_" + str(mass) + "Coup_" + Couplings_names[idx] +".npy"
        filename_3 = Adress + "Data_Gamma_analysed/" + "Mass_" + str(mass) + "Coup_" + Couplings_names[idx] +".npy"
        idx+=1
        np.save(filename_1, np.array([data_x, data_y, data_z, data_e, data_w]))
        np.save(filename_2, np.array([mom_x, mom_y, mom_z, energy_gamma_1, energy_gamma_2, diff_energy_gamma], dtype="object"))
        np.save(filename_3, np.array([Data_Asym, Data_e_min_gamma, Data_Sep, Data_w_sep, Data_e1_sep, Data_e2_sep]))
