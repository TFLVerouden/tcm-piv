"""
Compare the PIV with literature
"""

import os
import numpy as np
import sys


# Add the functions directory to the path and import Gupta
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'functions'))
import Gupta2009 as Gupta

def Gupta_model(Gender, weight, length):
    """
    Functions imports the gender, weight and length and returns the modeled cough according to Gupta2009
    Args:
        Gender(str): "Male" or "Female" 
        weight (float): Weight of cougher in kg
        length (float): Length of cougher in m
        

    Returns:
        Flowrate (L/s):  The flow rate in L/s of the modelled cough
        Time (np.array): The time array for the cough, same length as the flowrate
    """


    Tau = np.linspace(0,10,101)

    PVT, CPFR, CEV = Gupta.estimator(Gender,weight, length)

    cough_E = Gupta.M_model(Tau,PVT,CPFR,CEV)
    Time = Tau*PVT
    Flowrate = cough_E*CPFR
    mask = Time <0.15
    Time= Time[mask]
    Flowrate = Flowrate[mask]

    return Flowrate,Time