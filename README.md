# DES-OPF-Design
This repository includes an MILP model to design a Distributed Energy System, which can be combined with NLP formulations for balanced AC optimal power flow (OPF) to form an MINLP model overall. 
Please use t9_main.py to run MILP, NLP and MINLP models.

Note that the results from the OPF class (voltage magnitudes, angles, active/reactive powers, and currents)
are all returned in p.u. Please use the bases of these to convert them to SI units. 

**Please cite as**:
De Mel IA, Klymenko O V., Short M. *Levels of Approximation for the Optimal Design of Distributed Energy Systems.* Comput Aided Chem Eng 2021;50:1403–8. https://doi.org/10.1016/B978-0-323-88506-5.50216-3.
