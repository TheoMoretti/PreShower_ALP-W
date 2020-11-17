from various import *
from llpmodel import LLPModel

class MyEventCounter():

    def __init__(self,
            model, mass, generator="LHC",
            radius=1, length=5, offset=0, lumi=3000,
            logcoupmin=-8,logcoupmax=-3,ncoup=50, nphisample=2,
            channels=None
        ):

        #options
        self.logcoupmin=logcoupmin
        self.logcoupmax=logcoupmax
        self.nphisample=nphisample

        #Specify Model
        self.mass=mass
        self.model = model
        self.generator = generator

        #Specify detector
        self.distance = 480.
        self.length = length
        self.radius = radius
        self.offset = offset
        self.lumi = 1000.*lumi

        #setup different couplings to scan over
        self.setup_couplings(ncoup)

        #Presection cuts (using p and th as variables)
        self.preselectioncuts="th<0.01 and p>100"

        #Probability to have
        if channels is None:
            self.br=1.
        if channels is not None:
            self.br=0.
            llpmodel = LLPModel(self.model,self.mass,1)
            branchings = llpmodel.branching
            for channel in llpmodel.decays:
                if channel in channels: self.br+=branchings[channel]


    def setup_couplings(self,ncoup=50):
        self.couplings=[]
        self.ctaus=[]
        self.nsignals=[]
        self.stat_t=[]
        self.stat_e=[]
        self.stat_w=[]
        for coup in np.logspace(self.logcoupmin,self.logcoupmax, ncoup+1):
            llpmodel = LLPModel(self.model,self.mass,coup)
            self.couplings.append(1.*coup)
            self.ctaus.append(llpmodel.ctau)
            self.nsignals.append(0.)
            self.stat_t.append([])
            self.stat_e.append([])
            self.stat_w.append([])

    def rotate_event(self,p1,p0):
        #get axis of p0
        zaxis=Vector3D(0,0,1)
        rotaxis=zaxis.cross(p0.vector).unit()
        rotangle=zaxis.angle(p0.vector)

        #rotate:
        if rotangle!=0: p1=p1.rotate(rotangle,rotaxis)
        return p1

    def event_passes(self,momentum):

        #obtain 3-momentum
        p_llp=Vector3D(momentum.px,momentum.py,momentum.pz)

        #check of event passes
        pos_llp=float(self.distance/p_llp.z)*p_llp
        if type(pos_llp) is np.ndarray:
            pos_llp=Vector3D(pos_llp[0],pos_llp[1],pos_llp[2])
        if ((pos_llp.x+self.offset)**2 + (pos_llp.y)**2 > self.radius**2):
            return False
        else:
            return True
