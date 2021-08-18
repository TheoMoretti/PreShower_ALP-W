import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import gridspec as gs
from multiprocessing import Pool
import matplotlib as mpl

sys.path.append("/Users/theo/Desktop/ALP_Simulation/New_code/Modules")

from various import *
from br_validation import *
from llpmodel import LLPModel
from createllp import CreateLLP
from eventcounter import MyEventCounter
from llpgenerator import LLPGenerator

# IMPORTANT NOTE: The coupling for the input files has to be 1 so that it has no effect in the previous copmutations on the weight of the events and hence the results of the simulations done in what follows.

radius=0.1 #in meter
length=1.5 #in meter
offset=0 #in meter
luminosity=150 #in fb^-1
model = "ALP-W"
generator = "LHC"
masses = [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
coup = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3]
coup = np.logspace(-6,-3,60)
masses_1 = np.logspace(-3,-2,60)
masses_2 = np.logspace(-2,-1,60)
masses_3 = np.logspace(-1,0,60)

def do_convolution(mass):
    self=MyEventCounter(model="ALP-W",generator="LHC",mass=mass,radius=radius, length=length, offset=offset, lumi=luminosity,logcoupmin=-6,logcoupmax=-3,ncoup=60)
    #Load Flux file
    coupling = 1
    filename="test_cont/"+self.model+"_"+self.generator+"_m_"+str(self.mass)+"_c_"+str(coupling)+".npy"
    particles_llp,weights_llp=convert_list_to_momenta(filename=filename,mass=self.mass,filetype="npy",nsample=self.nphisample,preselectioncut=self.preselectioncuts) #self.preselectioncuts

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
            #print(weight_event)
            self.nsignals[icoup]+=weight_event* coup**2 * prob_decay * self.br
            self.stat_e[icoup].append(p.e)
            self.stat_w[icoup].append(weight_event* coup**2 *prob_decay * self.br)

    return self.couplings, self.ctaus, self.nsignals, self.stat_e, self.stat_w

data1 = [[],[]]
data2 = [[],[]]
data3 = [[],[]]

if __name__ == "__main__":
    p = Pool(12)
    res_1=p.map(do_convolution, masses_1)
    res_2=p.map(do_convolution, masses_2)
    res_3=p.map(do_convolution, masses_3)
    for coups_1, ctaus_1, nsigs_1, energies_1, weights_1 in res_1:
        data1[0].append(nsigs_1)
        data1[1].append(ctaus_1)
    for coups_2, ctaus_2, nsigs_2, energies_2, weights_2 in res_2:
        data2[0].append(nsigs_2)
        data2[1].append(ctaus_2)
    for coups_3, ctaus_3, nsigs_3, energies_3, weights_3 in res_3:
        data3[0].append(nsigs_3)
        data3[1].append(ctaus_3)

m_edges_1, m_edges_2, m_edges_3 = np.array(masses_1), np.array(masses_2), np.array(masses_3)
c_edges = np.array(coup)

m_centers_1, m_centers_2, m_centers_3 = (m_edges_1[:-1] + m_edges_1[1:]) / 2, (m_edges_2[:-1] + m_edges_2[1:]) / 2, (m_edges_3[:-1] + m_edges_3[1:]) / 2
c_centers = (c_edges[:-1] + c_edges[1:]) / 2
# print(m_centers)
list_m1, list_m2, list_m3 = [], [], []
list_c1, list_c2, list_c3 = [], [], []
list_w1, list_w2, list_w3 = [], [], []
list_d1, list_d2, list_d3 = [], [], []

for it,t in enumerate(m_centers_1):
    for ip,p in enumerate(c_centers):
        list_m1.append(np.log10(m_centers_1[it]))
        list_c1.append(np.log10(c_centers[ip]))
        list_w1.append(data1[0][it][ip])
        list_d1.append(data1[1][it][ip])

for it,t in enumerate(m_centers_2):
    for ip,p in enumerate(c_centers):
        list_m2.append(np.log10(m_centers_2[it]))
        list_c2.append(np.log10(c_centers[ip]))
        list_w2.append(data2[0][it][ip])
        list_d2.append(data2[1][it][ip])

for it,t in enumerate(m_centers_3):
    for ip,p in enumerate(c_centers):
        list_m3.append(np.log10(m_centers_3[it]))
        list_c3.append(np.log10(c_centers[ip]))
        list_w3.append(data3[0][it][ip])
        list_d3.append(data3[1][it][ip])


fig, axes = plt.subplots(figsize=(20,10),nrows=2, ncols=3)
fig.subplots_adjust(left=0.035, bottom=0.06, right=0.98, top=0.95, wspace=0.205, hspace=0.2)
im1 = axes.flat[0].hist2d(x=list_m1,y=list_c1,weights=list_w1,bins=[len(masses_1)-1,len(coup)-1], cmap = "inferno", vmin = 10**-10, vmax = 10**5, range=[[-3,-2],[-6,-3]],norm=matplotlib.colors.LogNorm())
im1 = axes.flat[1].hist2d(x=list_m2,y=list_c2,weights=list_w2,bins=[len(masses_2)-1,len(coup)-1], cmap = "inferno", vmin = 10**-10, vmax = 10**5, range=[[-2,-1],[-6,-3]],norm=matplotlib.colors.LogNorm())
im1 = axes.flat[2].hist2d(x=list_m3,y=list_c3,weights=list_w3,bins=[len(masses_3)-1,len(coup)-1], cmap = "inferno", vmin = 10**-10, vmax = 10**5, range=[[-1,0],[-6,-3]],norm=matplotlib.colors.LogNorm())
im2 = axes.flat[3].hist2d(x=list_m1,y=list_c1,weights=list_d1,bins=[len(masses_1)-1,len(coup)-1], cmap = "inferno", vmin = 10**-6, vmax = 10**6, range=[[-3,-2],[-6,-3]],norm=matplotlib.colors.LogNorm())
im2 = axes.flat[4].hist2d(x=list_m2,y=list_c2,weights=list_d2,bins=[len(masses_2)-1,len(coup)-1], cmap = "inferno", vmin = 10**-6, vmax = 10**6, range=[[-2,-1],[-6,-3]],norm=matplotlib.colors.LogNorm())
im2 = axes.flat[5].hist2d(x=list_m3,y=list_c3,weights=list_d3,bins=[len(masses_3)-1,len(coup)-1], cmap = "inferno", vmin = 10**-6, vmax = 10**6, range=[[-1,0],[-6,-3]],norm=matplotlib.colors.LogNorm())
cax1,kw1 = mpl.colorbar.make_axes([ax for ax in axes.flat[0:3]], fraction = 0.05)
cax2,kw2 = mpl.colorbar.make_axes([ax for ax in axes.flat[3:6]], fraction = 0.05)
plt.colorbar(im1[3], cax=cax1, shrink=1, pad = 0.05)
plt.colorbar(im2[3], cax=cax2, shrink=1, pad = 0.05)
for i in range(len(axes.flat)):
    axes.flat[i].set_xlabel("Mass in [GeV]")
    axes.flat[i].set_ylabel("Coupling")
    if i in [0,1,2]: axes.flat[i].set_title("Number of events distribution")
    else : axes.flat[i].set_title("Decay distance distribution")

plt.show()
