# -*- coding: utf-8 -*-
"""
Created on Sat Oct 31 16:30:56 2020

@author: ishan
"""
from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from pandas import ExcelWriter

class DES_OPF(object):
    def __init__(self, house, ft, gen,\
                 df_grid, df_network, slack, nodes, bus_connectivity):
        self.house = house #house names
        #self.df = df #dataframe for electricity
        #self.dfh = dfh #dataframe for heating
        #self.days = days #number of days in season
        #self.interval = interval #duration of time interval
        self.ft = ft
        self.gen = gen
        #self.irrad = irrad #irradiance data
        #self.dfi = dfi #dataframe for irradiance
        #self.df_scalars = df_scalars #dataframe with all the parameters
        #self.df_roof = df_roof
        #self.elec_house = elec_house
        #self.heat_house = heat_house
        #self.df_batteries = df_batteries
        #self.df_volume = df_volume
        #self.battery = battery #battery types available
        self.df_grid = df_grid
        self.df_network = df_network
        self.slack = slack
        self.nodes = nodes
        self.bus_connectivity = bus_connectivity
        
    def OPF(self):
        
        model = ConcreteModel()
        
        model.n = Set(initialize = self.nodes)
        model.m = Set(initialize = model.n)
        #model.periods2 = len(list(self.df[1])) #used to create a RangeSet for model.p
        model.t = RangeSet(self.ft, doc= 'periods/timesteps')
        
        resistance = {}
        reactance = {}
        
        for k,v in self.bus_connectivity.items():
            #distance[v] = self.df_network['Length (km)'][k-1]
            resistance[v] = self.df_network['Resistance (p.u.)'][k-1]
            reactance[v] = self.df_network['Reactance (p.u.)'][k-1]
        #print(distance)
        #print(resistance)
        #print(reactance)
        
        model.line_R = Param(model.n, model.m, initialize = resistance, default=0)
        model.line_X = Param(model.n, model.m, initialize = reactance, default=0)
        
        V_BASE = self.df_grid.iat[0,1] #'nominal voltage (V)'
        PF =  self.df_grid.iat[1,1] #doc = 'power factor'
        V_UB = self.df_grid.iat[2,1] #doc = 'network voltage upper bound (V)'
        V_LB = self.df_grid.iat[3,1] #doc = 'network voltage lower bound (V)'
        FREQ = self.df_grid.iat[4,1] #doc = 'network frequency (Hz)'
        S_BASE = self.df_grid.iat[5,1] #doc = 'base apparent power (kVA)'
        I_MAX = self.df_grid.iat[6,1] #doc = 'line current (A)'
        I_BASE = S_BASE/V_BASE
        
        '''Constructing the admittance matrix'''
        def branch_series_admittance(model,n,m):
            if (model.line_R[n,m]**2+model.line_X[n,m]**2) !=0:
                return complex((model.line_R[n,m]/(model.line_R[n,m]**2+model.line_X[n,m]**2)),\
                           -(model.line_X[n,m]/(model.line_R[n,m]**2+model.line_X[n,m]**2)))
            else:
                return 0
        model.y_series = Param(model.n,model.m, initialize = branch_series_admittance)
        # model.y_series.pprint()
        
        y_series_mags_sqr = {}
        for k, v in model.y_series.items():
            if v != 0:
                y_series_mags_sqr[k] = v.real**2 + v.imag**2
            else:
                y_series_mags_sqr[k] = 0
        model.y_branch_magsqr = Param(model.n, model.m, initialize = y_series_mags_sqr)
        # model.y_branch_magsqr.pprint()
        
        Y = {}
        for k,v in model.y_series.items():
            Y[k] = v
        #print(Y)
        
        diags = {}
        for n in self.nodes:
            diags[n] = (n,n)
        #print('*******')
        #print(diags)
        
        non_diags = {}
        for n in self.nodes:
            for k in self.nodes:
                if n != k:
                    non_diags[n,k] = 0
        #print('**---*****')
        #print(non_diags)
        
        y_diags = {}
        for k, (v1,v2) in diags.items():
            y_diags[k,k] = sum(v for (k1,k2),v in Y.items() if k==k1 or k==k2)
        #print("DIAGONALS")
        #print(y_diags)
                
        y_nd = {}
        for (kk1,kk2),v in non_diags.items():
            y_nd[kk1,kk2] = - sum(v for (k1,k2),v in Y.items() if kk1==k1 and kk2==k2) - \
                sum(v for (k1,k2),v in Y.items() if kk1==k2 and kk2==k1)
        #print("NON-DIAGONALS")
        #print(y_nd)
            
        Admittance = {**y_diags,**y_nd}
        #print("********MERGED woohoo!*******")
        #print(Admittance)
        
        Conductance = {}
        Susceptance = {}
        for k, v in Admittance.items():
            Conductance[k]= v.real
            Susceptance[k]= v.imag
        #print("---Conductance---")
        #print(Conductance)
        #print("---Susceptance---")
        #print(Susceptance)
        
        model.G = Param(model.n,model.m, initialize = Conductance)
        model.B = Param(model.n,model.m, initialize = Susceptance)
        
        # =============================================================================
        # Variables and Constraints
        # =============================================================================
        
        model.V = Var(model.n, model.t, bounds = (None,None), doc = 'node voltage', initialize = 1)
        model.P = Var(model.n, model.t,  bounds = (None,None), doc = 'node active power', initialize = 0)
        model.Q = Var(model.n, model.t,  bounds = (None,None), doc = 'node reactive power', initialize = 0)
        model.theta = Var(model.n, model.t, bounds = (None,None), doc = 'voltage angle', initialize = 0)
        model.current_sqr = Var(model.n, model.m, model.t, bounds = (None,None), initialize=0)
        
        def P_balance(model,n,t):
            return model.P[n,t] == model.V[n,t]*sum(model.V[m,t]*((model.G[n,m]*cos(model.theta[n,t]-model.theta[m,t])) \
                                         + (model.B[n,m]*sin(model.theta[n,t]-model.theta[m,t]))) for m in model.m)
        model.C1 = Constraint(model.n,model.t, rule = P_balance)
        
        def Q_balance(model,n,t):
            return model.Q[n,t] ==  model.V[n,t]*sum(model.V[m,t]*((model.G[n,m]*sin(model.theta[n,t]-model.theta[m,t])) \
                                         - (model.B[n,m]*cos(model.theta[n,t]-model.theta[m,t]))) for m in model.m)
        model.C2 = Constraint(model.n,model.t, rule = Q_balance)
        
        def V_bus_upper(model,n,t):
            if n != self.slack:
                return model.V[n,t] <= V_UB/V_BASE
                #return model.V[n,t] <= 1.05
            else:
                return model.V[n,t] == 1
        model.C3 = Constraint(model.n,model.t, rule = V_bus_upper)
        
        def V_bus_lower(model,n,t):
            if n != self.slack:
                return model.V[n,t] >= V_LB/V_BASE
                #return model.V[n,t] >= 0.95
            else:
                return model.V[n,t] == 1
        model.C4 = Constraint(model.n,model.t, rule = V_bus_lower)
        
        def theta_upper(model,n,t):
            if n != self.slack:
                return model.theta[n,t] <= np.pi
            else:
                return model.theta[n,t] == 0
        model.C5 = Constraint(model.n,model.t, rule = theta_upper)
        
        def theta_lower(model,n,t):
            if n != self.slack:
                return model.theta[n,t] >= (-np.pi)
            else:
                return model.theta[n,t] == 0
        model.C6 = Constraint(model.n,model.t, rule = theta_lower)
        
# =============================================================================
        def Non_gen_P(model,n,t):
            if n !=self.slack and n not in self.gen:
                return model.P[n,t] == 0
            else:
                return Constraint.Skip
        model.C7 = Constraint(model.n,model.t, rule = Non_gen_P)
        
        def Non_gen_Q(model,n,t):
            if n != self.slack and n not in self.gen:
                return model.Q[n,t] == 0
            else:
                return Constraint.Skip
        model.C8 = Constraint(model.n,model.t, rule = Non_gen_Q)
# =============================================================================
        
        # New current constraint using series branch admittance
        def current_equality(model,n,m,t):
            if value(model.y_branch_magsqr[n,m]) != 0:
                return model.current_sqr[n,m,t] == ((((model.V[n,t]*cos(model.theta[n,t])) - (model.V[m,t]*cos(model.theta[m,t])))**2 \
                    + ((model.V[n,t]*sin(model.theta[n,t])) - (model.V[m,t]*sin(model.theta[m,t])))**2) \
                    * model.y_branch_magsqr[n,m])
            else:
                return Constraint.Skip
        model.C11 = Constraint(model.n, model.m, model.t, rule=current_equality)
        
        def current_constraint(model,n,m,t):
            if value(model.y_branch_magsqr[n,m]) != 0:
                return model.current_sqr[n,m,t] <= (I_MAX/I_BASE)**2
            else:
                return Constraint.Skip
        model.C12 = Constraint(model.n, model.m, model.t, rule=current_constraint)
            
        
        return model