import pythia8
from various import *
from llpmodel import LLPModel

class LLPGenerator():

    def __init__(self,model, setup="theory"):
        self.model = model
        self.setup = setup

    #specify Dark Photon model
    def specify_darkphoton(self,pythia,mass):

        pythia.readString("ProcessLevel:all = off");
        pythia.readString("PartonLevel:FSR = off");
        pythia.readString("111:mayDecay = off");
        pythia.readString("310:mayDecay = off");
        pythia.readString("130:mayDecay = off");

        #define LLP model
        llpmodel = LLPModel(self.model, mass=mass, coupling=1)
        channels, branching = llpmodel.decays, llpmodel.branching

        #Decay into Hadrons
        if mass <= 1.50001:

            #use Y(1s)
            pythia.readString("553:m0="+str(mass));

            command="oneChannel"
            for channel in channels:

                # ignore 'other' channel and decay into quarks
                if channels[channel][0] == "large": continue
                if channels[channel][1] is None: continue

                # bRatio
                bRatio = str(branching[channel])

                # products
                products = ""
                for pid in channels[channel][1]: products = products + str(pid) + " "

                # meMode
                meMode =channels[channel][2]

                # add to pythia
                pythia.readString("553:"+command+" = 1 "+bRatio+" "+meMode + " "+products)
                command="addChannel"

            return pythia, 553

        else:

            #use Z'
            pythia.readString("32:m0="+str(mass));

            command="oneChannel"
            for channel in channels:
                # ignore decay into hadrons
                if channels[channel][0] == "small": continue

                # bRatio
                bRatio = str(branching[channel])

                # products
                products = ""
                for pid in channels[channel][1]: products = products + str(pid) + " "

                # meMode
                meMode =channels[channel][2]

                # add to pythia
                pythia.readString("32:"+command+" = 1 "+bRatio+" "+meMode + " "+products)
                command="addChannel"

            return pythia, 32


    #specify Dark Photon model
    def specify_darkhiggs(self,pythia,mass):

        pythia.readString("ProcessLevel:all = off");
        pythia.readString("PartonLevel:FSR = off");
        pythia.readString("111:mayDecay = off");
        pythia.readString("310:mayDecay = off");
        pythia.readString("130:mayDecay = off");

        #define LLP model
        llpmodel = LLPModel(self.model, mass=mass, coupling=1)
        channels, branching = llpmodel.decays, llpmodel.branching

        #Decay into Hadrons
        if mass <= 2.0001:

            #use etab0(1P)
            pythia.readString("10551:m0="+str(mass));

            command="oneChannel"
            for channel in channels:

                # ignore 'other' channel and decay into quarks
                if channels[channel][0] == "large": continue
                if channels[channel][1] is None: continue

                # bRatio
                bRatio = str(branching[channel])

                # products
                products = ""
                for pid in channels[channel][1]: products = products + str(pid) + " "

                # meMode
                meMode =channels[channel][2]

                # add to pythia
                pythia.readString("10551:"+command+" = 1 "+bRatio+" "+meMode + " "+products)
                command="addChannel"

            return pythia, 10551

        else:

            #use Higgs
            pythia.readString("25:m0="+str(mass));

            command="oneChannel"
            for channel in channels:
                # ignore decay into hadrons
                if channels[channel][0] == "small": continue

                # bRatio
                bRatio = str(branching[channel])

                # products
                products = ""
                for pid in channels[channel][1]: products = products + str(pid) + " "

                # meMode
                meMode =channels[channel][2]

                # add to pythia
                pythia.readString("25:"+command+" = 1 "+bRatio+" "+meMode + " "+products)
                command="addChannel"

            return pythia, 25

    #specify ALP-W model
    def specify_alpw(self,pythia,mass):

        pythia.readString("ProcessLevel:all = off");
        pythia.readString("PartonLevel:FSR = off");

        #define LLP model
        llpmodel = LLPModel(self.model, mass=mass, coupling=1)
        channels, branching = llpmodel.decays, llpmodel.branching

        #Decay into Hadrons
        if mass <= 1:

            #use etab0(1P)
            pythia.readString("10551:m0="+str(mass));

            command="oneChannel"
            for channel in channels:

                # ignore 'other' channel and decay into quarks
                if channels[channel][0] == "large": continue
                if channels[channel][1] is None: continue

                # bRatio
                bRatio = str(branching[channel])

                # products
                products = ""
                for pid in channels[channel][1]: products = products + str(pid) + " "

                # meMode
                meMode =channels[channel][2]

                # add to pythia
                pythia.readString("10551:"+command+" = 1 "+bRatio+" "+meMode + " "+products)
                command="addChannel"

            return pythia, 10551

        else:

            #use Higgs
            pythia.readString("25:m0="+str(mass));

            command="oneChannel"
            for channel in channels:
                # ignore decay into hadrons
                if channels[channel][0] == "small": continue

                # bRatio
                bRatio = str(branching[channel])

                # products
                products = ""
                for pid in channels[channel][1]: products = products + str(pid) + " "

                # meMode
                meMode =channels[channel][2]

                # add to pythia
                pythia.readString("25:"+command+" = 1 "+bRatio+" "+meMode + " "+products)
                command="addChannel"

            return pythia, 25

    #specify Dark Photon model
    def specify_darkphoton_pythia(self,pythia,mass):
        pythia.readString("Zprime:universality=on");
        pythia.readString("32:m0="+str(mass));

        pythia.readString("Zprime:vd=-0.3333");
        pythia.readString("Zprime:vu=0.6666");
        pythia.readString("Zprime:ve=-1");
        pythia.readString("Zprime:vnue=0");

        pythia.readString("Zprime:ad=0");
        pythia.readString("Zprime:au=0");
        pythia.readString("Zprime:ae=0");
        pythia.readString("Zprime:anue=0");

        pythia.readString("ProcessLevel:all = off");
        pythia.readString("PartonLevel:FSR = off");
        pythia.readString("111:mayDecay = off");
        pythia.readString("310:mayDecay = off");
        pythia.readString("130:mayDecay = off");

        return pythia, 32

    #specify Dark Higgs model
    def specify_darkhiggs_pythia(self,pythia,mass):
        pythia.readString("54:m0="+str(mass));

        pythia.readString("Sdm:vf=1");
        pythia.readString("Sdm:af=0");
        pythia.readString("Sdm:vX=0");
        pythia.readString("Sdm:aX=0");

        pythia.readString("ProcessLevel:all = off");
        pythia.readString("PartonLevel:FSR = off");
        pythia.readString("111:mayDecay = off");
        pythia.readString("310:mayDecay = off");
        pythia.readString("130:mayDecay = off");

        return pythia, 54

    # function that simulates `nevent` dark photon decays for dark photon mass `mass`
    def simulate_events(self,mass, nevent=1000, print_first_event=False,print_partile_data = False,outputfile=None):

        #specify particle
        px,py,pz,en = 0,0,0,mass
        status,col,acol,scale,pol = 2,0,0,0,9.

        #initialize pythia
        pythia = pythia8.Pythia()

        if self.model=="DarkPhoton" and self.setup=="theory":
            pythia, pid =self.specify_darkphoton(pythia=pythia,mass=mass)
        if self.model=="DarkHiggs" and self.setup=="theory":
            pythia, pid =self.specify_darkhiggs(pythia=pythia,mass=mass)
        if self.model=="DarkPhoton" and self.setup=="pythia":
            pythia, pid =self.specify_darkphoton_pythia(pythia=pythia,mass=mass)
        if self.model=="DarkHiggs"  and self.setup=="pythia":
            pythia, pid =self.specify_darkhiggs_pythia(pythia=pythia,mass=mass)
        if self.model=="ALP-W":
            pythia, pid =self.specify_alpw(pythia=pythia,mass=mass)
        if print_partile_data: print (pythia.particleData)
        pythia.init()

        # Begin event loop. Generate event. Skip if error. List first one.
        events = []
        for iEvent in range(0, nevent):
            pythia.event.reset()
            pythia.event.append(pid, status, col, acol, px, py, pz, en, mass, scale, pol)
            pythia.next()
            if print_first_event and iEvent==0: print(pythia.event)

            #Loop over particles in event. Find pions
            event = []
            for part in pythia.event:
                if part.status()>0:
                    event.append([part.id(),part.px(),part.py(),part.pz(),part.e()])
            events.append(event)

        if outputfile is not None:
            np.save(outputfile,events)

        return events

    # function that extracts branching fractions
    def extract_br(self,events):
        nevent = float(len(events))
        branching_fraction={}
        for event in events:
            final_state=[particle[0] for particle in event]
            final_state=list(np.sort(final_state))
            if str(final_state) in branching_fraction.keys():
                branching_fraction[str(final_state)] += 1./nevent
            else:
                branching_fraction[str(final_state)] = 1./nevent
        return branching_fraction

    # function that scans over the mass and obtains the branching fraction
    def br_scan(self,massmin=0.105, massmax=1.95, nmass = 40, nevent=1000):
        branching_fractions=[]
        for mass in np.linspace(massmin, massmax, nmass):
            events=self.simulate_events(mass=mass,nevent=nevent)
            bf=self.extract_br(events)
            branching_fractions.append([mass,bf])
        return np.array(branching_fractions)


    # scan over mass and claculate BR
    def scan_br(self, massmin=0.01, massmax=2.0, nmass=40, nevent=1000):

        # Simulate BR
        data=self.br_scan(massmin=massmin, massmax=massmax,nmass=nmass, nevent=nevent)
        np.save("files/results/brscan_"+self.model+".npy",data)
