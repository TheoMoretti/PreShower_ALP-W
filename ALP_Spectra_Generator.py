import numpy as np
import sys
import matplotlib
from matplotlib import pyplot as plt
from multiprocessing import Pool

#Insert the path of the different modules (classes with functions)
sys.path.append("/Users/theo/Desktop/ALP_Simulation/New_code/Modules")

from various import *
from br_validation import *
from llpmodel import LLPModel
from createllp import CreateLLP
from eventcounter import MyEventCounter
from llpgenerator import LLPGenerator

#Define the masses for which one want to get the ALP spectra from Interaction Point.
Masses_1 = [1.02*10**-1, 1.26*10**-1, 1.58*10**-1, 2*10**-1, 2.46*10**-1, 3.06*10**-1, 3.78*10**-1, 4.45*10**-1]
Masses_2 = [1.14*10**-1, 1.43*10**-1, 1.76*10**-1, 2.22*10**-1, 2.74*10**-1, 3.41*10**-1, 4.08*10**-1, 4.84*10**-1]
Masses_3 = [2.06*10**-1, 2.18*10**-1, 2.32*10**-1, 2.47*10**-1, 2.62*10**-1]
Masses_4 = [2.88*10**-1, 3.17*10**-1, 3.49*10**-1, 3.81*10**-1, 4.17*10**-1]
Masses_5 = [4.62*10**-1, 5.08*10**-1, 5.61*10**-1, 6.12*10**-1, 6.76*10**-1]
Masses_6 = [7.31*10**-1, 8.15*10**-1, 8.91*10**-1, 9.86*10**-1, 1.09, 1.18]
Masses_7 = [1.28, 1.39, 1.49, 1.6, 1.71, 1.82, 1.93]

#The coupling here has to be set to 1 and nothing else, the contribution of the coupling to the Cross Section of ALP will be accounted further (See code "Get_distributions").
coupling = 1

def get_LLP_spectrum(mass, model = "ALP-W", coup = coupling, num_sample=100, generator="LHC", channels=None):
    #The variable num_sample accounts for the MC sampling of the angles for the decay in the rest frame of the mesons coming from Interaction Point.
    """
    function that gives the ALP spectrum in terms of log10(theta) and log10(p)
    with corresponding weights for different LLP masses
    """
    print("Computing for mass = " + str(mass))
    creater = CreateLLP(model=model,mass=mass,coup=coup)
    #output
    momenta_lab=[]
    weights_lab=[]

    #Getting the mesons Particle ID and masses from which the ALP can be a decay product.
    meson_pids, meson_masses  = creater.llpmodel.get_mesons()

    #Loop over initial states
    for meson_pid,meson_mass in zip(meson_pids, meson_masses):
        weight_sum, weight_sum_f=0,0
        if channels is not None:
            if not meson_pid in channels: continue
            print ("Executing:", self.mass, meson_pid)
        elif model == "ALP-W":
            _,m0,m1,m2,m3=creater.llpmodel.br_into_llp(meson_pid)
            if m3==None: m3=0
            if m0 <= m1+m2+m3: continue
            #Getting the spectra of the mesons (Cross section) from Interaction Point in the log10(theta) and log10(p) plane.
            input_file="Ressources/Input/mesons/"+generator+"_"+str(meson_pid)+"_log.txt"
            #Converting the 3 input lists into 2 lists with only momenta and Cross Section.
            meson_momenta, meson_weights = convert_list_to_momenta(input_file,meson_mass)
            #Decay the mesons in their rest frame and boost the momentum of the ALP produces in the decay, back in the lab frame.
            if meson_pid==5:
                llp_momenta,llp_branching = creater.decay_meson_in_restframe(meson_pid,num_sample*2)
            else:
                llp_momenta,llp_branching = creater.decay_meson_in_restframe(meson_pid,num_sample)
            for p_meson,w_meson in zip(meson_momenta, meson_weights):
                #Associate to each ALP event its Cross Section:
                w_decay = creater.get_decay_prob(meson_pid, p_meson)
                for p_llp,w_lpp in zip(llp_momenta,llp_branching):
                    p_llp_lab=p_llp.boost(-1.*p_meson.boostvector)
                    momenta_lab.append(p_llp_lab)
                    weights_lab.append(w_meson*w_lpp*w_decay)
                    weight_sum+=w_meson*w_lpp*w_decay
                    # if p_llp_lab.pz>100 and p_llp_lab.pt/np.abs(p_llp_lab.pz)<0.1/480.: weight_sum_f+=w_meson*w_lpp*w_decay
                    # weight_sum_f+=w_meson*w_lpp*w_decay
        print(meson_pid,"{:.2e}".format(weight_sum),"{:.2e}".format(weight_sum_f))

    return [momenta_lab,weights_lab,mass,weight_sum_f]

#This code is optimized in a parallel way, one can compute the spectra for different masses at the same time.
if __name__ == "__main__":
    p = Pool(12)
    res = p.map(get_LLP_spectrum, Masses_7)
    for [p,w,mass,w_tot] in res:
        model = "ALP-W"
        generator = "LHC"
        #The bins here respectively coresponds to the bins for the angles axis and momentum axis.
        #The edeges of the axis is a fixed parameter of the function "convert_to_hist_list" but can be changed in the Module "Various".
        bin_x, bin_y = 500, 800
# p,w,mass,weight_sum_f = get_LLP_spectrum(Mass)
        name="/Users/theo/Desktop/ALP_Simulation/New_code/data/Discovery_potential/Raw_ALP_Spectra/"+model+"_"+generator+"_m_"+str(mass)+"_c_"+str(coupling)+ ".npy"
        #The function "convert_to_hist_list" builds a 2D histogram out of the momenta and weights in the log10(theta) and log10(p) plane
        convert_to_hist_list(p,w,bin_x, bin_y, filename = name, do_plot = False, do_return = False)
        print("File saved for mass =  " + str(mass) + " and coupling = " +str(coupling))

#The files for each mass point are then saved in the "npy" format to avoid reading issues on another device.
#Only 3 lists are saved in the output file, the momenta, the angle and the Cross Section of each bin of the histogram (in log10 scale).
