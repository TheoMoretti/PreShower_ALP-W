# PreShower_ALP-W
The aim of this code is to study the generation of Axion Like Particles (ALP) from the the Interaction Point of the ATLAS experiment using the LHC at CERN. This code also aims at studying the decays of such particles into 2 photons at the precise location where the FASER experiment is.

The project is separated into 2 main parts:
 * The first part of the project is a Jupyter Notebook which allows one to pick a masS point from the set of already generated (INSERT NUMBER HERE) different mass points for the ALP-W model. The code will then allow one to look at the different aspects of the ALP inside FASER but also to look as the different aspects of the di-photons decay products from the ALP inside FASER for any value of the coupling one wants to have in the ALP-W model.
 
 The possibility to also generate any mass point requires one to download the associated code and generate the mass point by himself since it requires specific packages like Pythia which are not disponible on the Jupyter Notebook.
 
  * The second part of the project allows one to create the what is here called a "Reach plot" which is nothing more than the contour line in the (Mass - Coupling) parameter plane for which the number of events inside FASER is greater than 3. Furthermore, is one wants to get the number of events convoluted with the efficiencies of the Pre Shower from the GEANT4 simulations to get the contour for different separations between the photons and for different luminosities.


This repository is based on the code written by Felix Kling for the simulations of the ALP-W events generated from Interaction Point at the ATLAS experiment at CERN.
