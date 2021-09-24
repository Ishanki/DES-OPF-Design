# DES-OPF-Design
This repository includes an MILP model to design a Distributed Energy System, which can be combined with NLP formulations for balanced AC optimal power flow (OPF) to form an MINLP model overall. \
Please use _t9_main.py_ to run MILP, NLP and/or MINLP models. 

Note that the results from the OPF class (voltage magnitudes, angles, active/reactive powers, and currents)
are all returned in p.u. Please use the bases of these to convert them to SI units. 

**Preprint (completed results and analysis)**:\
I. De Mel, O. V. Klymenko, and M. Short, “Investigating the Consolidation of the Optimal Design of Distributed Energy Systems with Optimal Power Flow,” Sep. 2021, \
Available: https://arxiv.org/abs/2109.11228v1

**Preliminary analysis - Conference Proceedings**:\
I. A. De Mel, O. V. Klymenko, and M. Short, “Levels of Approximation for the Optimal Design of Distributed Energy Systems,” Comput. Aided Chem. Eng., vol. 50, pp. 1403–1408, Jan. 2021. \
https://doi.org/10.1016/B978-0-323-88506-5.50216-3.

# License
Copyright (c) 2020, Ishanki De Mel. GPL-3.
