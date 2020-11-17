from various import *

class LLPModel:

    def __init__(self,model,mass,coupling):
        self.model = model
        self.mass  = mass
        self.coup  = coupling
        self.set_ctau()
        self.set_br()

    def read_and_interpolate(self,filename,mass):
        br=readfile(filename)
        return np.interp(mass, br.T[0],br.T[1])

    def get_mesons(self):
        if self.model in ["ALP-W"]:
            pids = [5,130,321]

        masses = [None if pid is 0 else self.masses(pid) for pid in pids]
        return pids, masses

    def set_ctau(self):

        #ctau
        ctau=self.read_and_interpolate("/Ressources/ctau_ALP-W.txt",self.mass)
        self.ctau = ctau / self.coup / self.coup

    def set_br(self):
        if self.model == "ALP-W":
            self.decays ={
                "gamma_gamma":     ["full",  [22, 22]                   , "0" ],
            }
            label = "alp_W"

        self.branching = {}
        for channel in self.decays:
            this_br = self.read_and_interpolate("/Ressources/bfrac_alp_W_gamma_gamma.txt",self.mass)
            self.branching[channel] = this_br

    def masses(self,pid):
        if   pid==111: return 0.135
        elif pid==221: return 0.547
        elif pid==321: return 0.495
        elif pid==310: return 0.495
        elif pid==130: return 0.495
        elif pid==211: return 0.135
        elif pid==331: return 0.957
        elif pid==5: return 5.279
        elif pid==23: return 91.2
        else: return None

    def kaellen(self,a,b,c):
        return a*a + b*b + c*c - 2*(a*b + a*c + b*c)

    def ps(self,m0,m1,m2):
        #phase space factor: 2p / m = sqrt(lamda)/m2
        if m1+m2>m0: return 0
        else: return np.sqrt(self.kaellen(m0*m0,m1*m1,m2*m2)) / (m0**2)

    def br_into_llp(self,pid):

        coup=self.coup
        mass=self.mass
        if self.model in ["ALP-W"]:
            if (pid==5):
                m0,m1,m2,m3 = self.masses(5),self.masses(321),mass,None
                ff1 = 0.33/(1-mass**2/38)   #Form Factor for K
                ff2 = 1.35/(1-mass**2/28)-1/(1-mass**2/37)  #Form Factor for K*
                if m0<m1+m2: br=0
                else: br = 10**5*coup*coup*(self.ps(m0,m1,m2)*ff1**2+self.ps(m0,m1,m2)**3*ff2**2)
            elif (pid==321):
                m0,m1,m2,m3 = self.masses(321),self.masses(211),mass,None
                if m0<m1+m2: br=0
                else: br = 10.53*coup*coup*self.ps(m0,m1,m2)
            elif (pid==130):
                m0,m1,m2,m3 = self.masses(130),self.masses(111),mass,None
                if m0<m1+m2: br=0
                else: br = 4.52*coup*coup*self.ps(m0,m1,m2)
            elif (pid==23):
                m0,m1,m2,m3 = self.masses(23),0,mass,None
                if m0<m1+m2: br=0
                else: br = 174.6*coup*coup

        return br,m0,m1,m2,m3
