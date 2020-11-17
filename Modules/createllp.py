from various import *
from llpmodel import LLPModel

class CreateLLP:

    def __init__(self,model, mass,coup):
        self.model = model
        self.mass = mass
        self.coup = coup
        self.llpmodel = LLPModel(model,mass,coup)

    def twobody_decay(
        self,
        p0,
        m0, m1, m2,
        phi, costheta):
        """
        function that decays p0 > p1 p2 and returns p1,p2
        """

        #get axis of p0
        zaxis=Vector3D(0,0,1)
        rotaxis=zaxis.cross(p0.vector).unit()
        rotangle=zaxis.angle(p0.vector)

        #energy and momentum of p2 in the rest frame of p0
        energy1   = (m0*m0+m1*m1-m2*m2)/(2.*m0)
        energy2   = (m0*m0-m1*m1+m2*m2)/(2.*m0)
        momentum1 = math.sqrt(energy1*energy1-m1*m1)
        momentum2 = math.sqrt(energy2*energy2-m2*m2)

        #4-momentum of p1 and p2 in the rest frame of p0
        en1 = energy1
        pz1 = momentum1 * costheta
        py1 = momentum1 * math.sqrt(1.-costheta*costheta) * np.sin(phi)
        px1 = momentum1 * math.sqrt(1.-costheta*costheta) * np.cos(phi)
        p1=LorentzVector(-px1,-py1,-pz1,en1)
        if rotangle!=0: p1=p1.rotate(rotangle,rotaxis)

        en2 = energy2
        pz2 = momentum2 * costheta
        py2 = momentum2 * math.sqrt(1.-costheta*costheta) * np.sin(phi)
        px2 = momentum2 * math.sqrt(1.-costheta*costheta) * np.cos(phi)
        p2=LorentzVector(px2,py2,pz2,en2)
        if rotangle!=0: p2=p2.rotate(rotangle,rotaxis)

        #boost p2 in p0 restframe
        p1_=p1.boost(-1.*p0.boostvector)
        p2_=p2.boost(-1.*p0.boostvector)
        return p1_,p2_

    def decay_meson_in_restframe_2body(self,meson_pid,num_s):
        """
        function that decays mesons in their rest frame into LLP, and returns
        the LLP momenta and corresponding branching fractions
        """
        #output list
        particles=[]
        weights=[]

        #get branching fractions
        br,m0,m1,m2,m3=self.llpmodel.br_into_llp(meson_pid)

        #create meson 4-vector
        p_meson=LorentzVector(0,0,0,m0)

        #MC sampling of angles
        for i in range(num_s):

            cos =random.uniform(-1.,1.)
            phi =random.uniform(-math.pi,math.pi)
            p_1,p_2=self.twobody_decay(p_meson,m0,m1,m2,phi,cos)

            particles.append(p_2)
            weights.append(br/num_s)

        return particles,weights

    def decay_meson_in_restframe(self,meson_pid,num_s=10000):

        #check how many final state particles
        _,_,_,_,m3=self.llpmodel.br_into_llp(meson_pid)

        if m3==None:
            return self.decay_meson_in_restframe_2body(meson_pid,num_s)


    def get_decay_prob(self,pid, momentum):

        # return 1 when decaying promptly
        if pid not in [321, 130, 310]: return 1

        # mass and lifetime
        if pid==130: ctau= 5.116*3    #[m]
        if pid==310: ctau= 0.008954*3 #[m]
        if pid==321: ctau= 1.238*3    #[m]

        # kinematics
        theta=math.atan(momentum.pt/momentum.pz)
        dbarz = ctau * momentum.pz / momentum.m
        dbart = ctau * momentum.pt / momentum.m

        #P to decay in beampipe
        factor=1
        if pid in [130, 310]:
            ltan = 140.
            ltas = 20.
            rpipe = 0.05
            if (theta < 0.017/ltas): factor= 1.- np.exp(- ltan/dbarz)
            elif (theta < 0.05/ltas): factor= 1.- np.exp(- ltas/dbarz)
            else: factor= 1.- np.exp(- rpipe /dbart)
        if pid in [321]:
            ltas = 20.
            rpipe = 0.05
            if (theta < 0.05/ltas): factor= 1.- np.exp(- ltas/dbarz)
            else: factor= 1.- np.exp(- rpipe /dbart)

        return factor
