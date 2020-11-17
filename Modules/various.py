import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import math
import random
from scipy.ndimage.filters import gaussian_filter
#import pylhe
#import gzip
#import shutil
from skhep.math.vectors import LorentzVector, Vector3D

# unzip file
#def gunzip_file(filepath):
#    with gzip.open(filepath+'.gz', 'rb') as f_in:
#        with open(filepath, 'wb') as f_out:
#            shutil.copyfileobj(f_in, f_out)

#convert LHEReader particle into LorentzVector particle
def get_p(particle):
    p=LorentzVector(particle['px'],particle['py'],particle['pz'],particle['e'])
    return p

#function that reads a table in a .txt file and converts it to a numpy array
def readfile(filename):
    list_of_lists = []
    with open(filename) as f:
        for line in f:
            if line[0]=="#":continue
            inner_list = [float(elt.strip()) for elt in line.split( )]
            list_of_lists.append(inner_list)
    return np.array(list_of_lists)

# convert a table into input for contour plot
def table2contourinput(data,idz=2):
    ntotal=len(data)
    ny=sum( 1 if d[0]==data[0][0] else 0 for d in data)
    nx=sum( 1 if d[1]==data[0][1] else 0 for d in data)
    xval = [data[ix*ny,0] for ix in range(nx)]
    yval = [data[iy,1] for iy in range(ny)]
    zval = [ [ data[ix*ny+iy,idz] for iy in range(ny) ] for ix in range(nx)]
    return np.array(xval),np.array(yval),np.array(zval)

# function that converts input file into meson spectrum
def convert_list_to_momenta(filename,mass,filetype="txt",nsample=1,preselectioncut=None):
    if filetype=="txt":
        list_logth, list_logp, list_xs = readfile(filename).T
    elif filetype=="npy":
        list_logth, list_logp, list_xs = np.load(filename)
    else:
        print ("ERROR: cannot rtead file type")
    particles=[]
    weights  =[]
    for logth,logp,xs in zip(list_logth,list_logp, list_xs):

        if xs   < 10.**-6: continue
        p  = 10.**logp
        th = 10.**logth

        if preselectioncut is not None:
            if not eval(preselectioncut): continue

        en = math.sqrt(p**2+mass**2)
        pz = p*np.cos(th)
        pt = p*np.sin(th)

        for n in range(nsample):
            phi= random.uniform(-math.pi,math.pi)

            px = pt*np.cos(phi)
            py = pt*np.sin(phi)
            p=LorentzVector(px,py,pz,en)

            particles.append(p)
            weights.append(xs/float(nsample))

    return particles,weights

# convert list of momenta to 2D histogram, and plot
def convert_to_hist_list(momenta,weights, bin_x, bin_y, do_plot=False, filename=None, do_return=False):

    t_edges = np.logspace(-5,0,num=bin_x+1)
    p_edges = np.logspace( 0,4,num=bin_y+1)

    tx = [np.arctan(mom.pt/mom.pz) for mom in momenta]
    px = [mom.p for mom in momenta]

    w, t_edges, p_edges = np.histogram2d(tx, px, weights=weights,  bins=(t_edges, p_edges))

    t_centers = (t_edges[:-1] + t_edges[1:]) / 2
    p_centers = (p_edges[:-1] + p_edges[1:]) / 2

    list_t = []
    list_p = []
    list_w = []

    for it,t in enumerate(t_centers):
        for ip,p in enumerate(p_centers):
            list_t.append(np.log10 ( t_centers[it] ) )
            list_p.append(np.log10 ( p_centers[ip] ) )
            list_w.append(w[it][ip])

    if do_plot:
        plt.hist2d(x=list_t,y=list_p,weights=list_w,bins=[100,100],range=[[-5,0],[0,4]],norm=matplotlib.colors.LogNorm() )
        plt.xlabel(r"log10($\theta$)")
        plt.ylabel(r"log10(p)")
        plt.colorbar()
        plt.show()

    if filename is not None:
        np.save(filename,[list_t,list_p,list_w])

    if do_return:
        return list_t,list_p,list_w


def convert_to_hist_list_lin(momenta,weights, bin_x, bin_y, filename=None, do_plot = False, do_return = False):

    t_edges = np.linspace(10**-5,1,num=bin_x+1)
    p_edges = np.linspace(1,10**4,num=bin_y+1)

    tx = [np.arctan(mom.pt/mom.pz) for mom in momenta]
    px = [mom.p for mom in momenta]

    w, t_edges, p_edges = np.histogram2d(tx, px, weights=weights,  bins=(t_edges, p_edges))

    t_centers = (t_edges[:-1] + t_edges[1:]) / 2
    p_centers = (p_edges[:-1] + p_edges[1:]) / 2

    list_t = []
    list_p = []
    list_w = []

    for it,t in enumerate(t_centers):
        for ip,p in enumerate(p_centers):
            list_t.append(t_centers[it])
            list_p.append(p_centers[ip])
            list_w.append(w[it][ip])

    if do_plot:
        plt.hist2d(x=list_t,y=list_p,weights=list_w,bins=[500,500],range=[[10**-5,1],[1,10**4]],norm=matplotlib.colors.LogNorm() )
        plt.xlabel(r"log10($\theta$)")
        plt.ylabel(r"log10(p)")
        # plt.xscale("log")
        plt.yscale("log")
        plt.colorbar()
        plt.show()

    if filename is not None:
        np.save(filename,[list_t,list_p,list_w])

    if do_return:
        return list_t,list_p,list_w

def convert_list_to_momenta_sampled_log(filename,mass,filetype="txt",nsample=1,preselectioncut=None,):
    if filetype=="txt":
        list_logth, list_logp, list_xs = readfile(filename).T
    elif filetype=="npy":
        list_logth, list_logp, list_xs = np.load(filename)
    else:
        print ("ERROR: cannot rtead file type")
    particles=[]
    weights  =[]
    for logth,logp,xs in zip(list_logth,list_logp, list_xs):

        if xs   < 10.**-6: continue
        p  = 10.**logp
        th = 10.**logth
        if preselectioncut is not None:
            if not eval(preselectioncut): continue

        for n in range(nsample):
            phi= random.uniform(-math.pi,math.pi)
            fth = np.random.normal(1, 0.05, 1)[0]
            fp  = np.random.normal(1, 0.05, 1)[0]

            th_sm=10**(logth*fth)
            p_sm=10**(logp*fp)

            en = math.sqrt(p_sm**2+mass**2)
            pz = p_sm*np.cos(th_sm)
            pt = p_sm*np.sin(th_sm)
            px = pt*np.cos(phi)
            py = pt*np.sin(phi)
            part=LorentzVector(px,py,pz,en)

            particles.append(part)
            weights.append(xs/float(nsample))

    return particles,weights
