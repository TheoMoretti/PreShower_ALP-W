from various import *
from llpmodel import LLPModel

# plot branching fractions
def br_validation_plot(model, massmin=0.01, massmax=2.0):

    # initiate plot
    fig, ax = plt.subplots(figsize=(9,7))

    # Define Model
    refmodel = LLPModel(model, mass=1, coupling=1)
    channels = refmodel.decays

    # Get Theory prediction
    masses = np.linspace(massmin, massmax, 400)
    branchings_theory = []
    for mass in masses:
        llpmodel = LLPModel(model, mass=mass, coupling=1)
        branchings_theory.append(llpmodel.branching)

    for channel in channels:
        br = [ branching[channel] for branching in branchings_theory ]
        if channels[channel][0] == "large": continue
        if channel == "other_hadrons":
            ax.plot(masses, br, label=channel, lw=1, color="black")
        else:
            ax.plot(masses, br, label=channel, lw=1)

    # Load simulated BR
    data = np.load("files/results/brscan_"+model+".npy")
    masses=data.T[0]
    branchings_sim=data.T[1]

    #add BR
    for channel in channels:
        if channels[channel][0]=="large": continue
        if channel is "other_hadrons": continue
        br_sim = [br[str(channels[channel][1])] if str(channels[channel][1]) in br else 0 for br in branchings_sim]
        ax.scatter(masses, br_sim, s=10)

    # other channels
    this_br=[]
    for br in branchings_sim:
        br_other=0
        for c in br:
            found=False
            for channel in channels:
                if channels[channel][0]=="large": continue
                if channel is "other_hadrons": continue
                if c==str(channels[channel][1]): found=True
            if not found: br_other+=br[c]
        this_br.append(br_other)
    ax.scatter(masses, this_br, c="black", s=10)

    #finish plot
    ax.legend(frameon=False)
    ax.set_ylim(0,1)
    ax.grid(True)

    return fig
