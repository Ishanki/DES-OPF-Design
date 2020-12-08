from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from pandas import ExcelWriter

'''
Created by I De Mel in April 2020 

Updates:
07/09/2020 
    > Operational cost of PVs and batteries 
    > Capital costs divided by 4 to adjust for the seasonal blocks
    > Removed boiler lower bound
    > Added carbon cost to grid electricity
22/10/2020
    
'''

class Residential_Seasonal_DES(object):
    def __init__(self, house, df, days, interval, ft, \
                 irrad, df_scalars, df_roof, elec_house, heat_house,df_batteries,\
                     df_volume, battery):
        self.house = house #house names
        self.df = df #dataframe for electricity
        #self.dfh = dfh #dataframe for heating
        self.days = days #number of days in season
        self.interval = interval #duration of time interval
        self.ft = ft
        self.irrad = irrad #irradiance data
        #self.dfi = dfi #dataframe for irradiance
        self.df_scalars = df_scalars #dataframe with all the parameters
        self.df_roof = df_roof
        self.elec_house = elec_house
        self.heat_house = heat_house
        self.df_batteries = df_batteries
        self.df_volume = df_volume
        self.battery = battery #battery types available
        
    def DES_MILP(self):
        
        model = ConcreteModel()
        
        model.i = Set(initialize= self.house, doc= 'residential users')
        model.periods2 = len(list(self.df[1])) #used to create a RangeSet for model.p
        model.t = RangeSet(model.periods2, doc= 'periods')
        model.t_night = RangeSet(1,13, doc = 'night periods eligible for night tariff')
        model.t_day = RangeSet(14,48, doc = 'day periods eligible for day tariff')
        model.c = Set(initialize = self.battery, doc='types of batteries available')
        
        house_num = list(range(1,(len(self.house))+1))
        
        print("data is now loading into loops")

        model.E_load = Param(model.i, model.t, initialize = self.elec_house, doc = 'electricity load')
        model.H_load = Param(model.i, model.t, initialize = self.heat_house, doc = 'heating load')
        #print(value(model.E_load['h1',34]))
        #print(value(model.H_load['h2',34]))
        
        
        model.Irradiance = Param(model.t, initialize = self.irrad, doc = 'solar irradiance')
        
        ra = {}
        for n, h in zip(range(len(self.house)), self.house):
            ra[h] = self.df_roof.iat[n,1]
        
        model.max_roof_area = Param(model.i, initialize = ra, doc = 'maximum roof surface area available')
        
        model.r = Param(initialize = self.df_scalars.iat[0,1], doc = 'interest rate')
        model.n = Param(initialize = self.df_scalars.iat[1,1], doc = 'project lifetime')
        model.price_grid = Param(initialize = self.df_scalars.iat[2,1], doc = 'electricity price in £ per kWh')
        model.price_gas = Param(initialize = self.df_scalars.iat[3,1], doc = 'price of gas in £ per kWh')
        model.carbon_grid = Param(initialize = self.df_scalars.iat[4,1], doc = 'carbon intensity of grid electricity in kg/kWh')
        model.cc_PV = Param(initialize = self.df_scalars.iat[5,1], doc = 'capital cost of PV in £ per panel (1.75 m2)')
        model.n_PV = Param(initialize = self.df_scalars.iat[6,1], doc = 'efficiency of the PV')
        model.oc_fixed_PV = Param(initialize = self.df_scalars.iat[7,1], doc = 'fixed operational cost of PV in £ per kW per year')
        model.oc_var_PV = Param(initialize = self.df_scalars.iat[8,1], doc = 'variable operational cost of PV in £ per kWh')
        model.SEG = Param(initialize = self.df_scalars.iat[9,1], doc = 'SEG Tariff for exporting in £ per kWh')
        model.cc_b = Param(initialize = self.df_scalars.iat[10,1], doc = 'capital cost of boiler per kWh')
        model.n_b = Param(initialize = self.df_scalars.iat[11,1], doc = "thermal efficiency of the boiler")
        model.panel_area = Param(initialize = self.df_scalars.iat[12,1], doc = 'surface area per panel in m2')
        model.panel_capacity = Param(initialize = self.df_scalars.iat[13,1], doc = 'rated capacity per panel in kW')
        model.max_capacity_PV = Param(initialize = self.df_scalars.iat[14,1], doc = 'maximum renewables capacity the DES is allowed to have as per SEG tariffs')
        model.c_carbon = Param(initialize = self.df_scalars.iat[15,1], doc = 'fake carbon cost for the grid')
        FIT_tariff = value(self.df_scalars.iat[16,1]) #'FIT generation tariff')
        pg_night = self.df_scalars.iat[19,1]
        pg_day = self.df_scalars.iat[20,1]
        
        '''battery parameters'''
        print("processing battery parameters....")
        vol_av = {}
        for n, h in zip(range(len(self.house)), self.house):
            vol_av[h] = self.df_volume.iat[n+1,1]
        #print(vol_av)
            
        model.VA = Param(model.i, initialize = vol_av, doc = 'maximum volume available to install battery')
        
        RTE1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            RTE1[c] = self.df_batteries.iat[n,1]
        #print(RTE1)
        
        model.RTE = Param(model.c, initialize = RTE1, doc='round trip efficiency')
        
        VED1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            VED1[c] = self.df_batteries.iat[n,2]
        #print(VED1)
        
        model.VED = Param(model.c, initialize = VED1, doc= 'volumetric energy density')
        
        mDoD1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            mDoD1[c] = self.df_batteries.iat[n,3]
        #print(mDoD1)
        
        model.max_DoD = Param(model.c, initialize = mDoD1, doc='max depth of discharge')

        mSoC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            mSoC1[c] = self.df_batteries.iat[n,4]
        #print(mSoC1)
        
        model.max_SoC = Param(model.c, initialize = mSoC1, doc='max state of charge')
        
        CCC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            CCC1[c] = self.df_batteries.iat[n,5]
        #print(CCC1)
        
        model.cc_storage = Param(model.c, initialize = CCC1, doc='capacity capital cost (£/kWh)')
        
        OMC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            OMC1[c] = self.df_batteries.iat[n,6]
        #print(OMC1)
        
        model.om_storage = Param(model.c, initialize = OMC1, doc='operational and maintenance cost (£/kW/y)')
        
        NEC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            NEC1[c] = self.df_batteries.iat[n,7]
        #print(NEC1)
        
        model.n_ch = Param(model.c, initialize = NEC1, doc='charging efficiency')
        
        NED1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            NED1[c] = self.df_batteries.iat[n,8]
        #print(NED1)
        
        model.n_disch = Param(model.c, initialize = NED1, doc='discharging efficiency')
        
        print("....battery parameters successfully processed.")    
        
        
        '''calculating capital recovery factor'''
        def capital_recovery_factor(model):
            return (model.r * ((1 + model.r)**model.n.value))/(((1 + model.r)**model.n.value)-1)
        model.CRF = Param(initialize = capital_recovery_factor)
        print("this is CRF:")
        print(model.CRF.value)
        
        '''Positive variables'''
        model.E_grid = Var(model.i, model.t, within = NonNegativeReals, doc = 'electricity imported from the grid in kW')
        model.E_PV_used = Var(model.i, model.t, within = NonNegativeReals, doc = 'electricity generated from the PV used at the house')
        model.E_PV_sold = Var(model.i, model.t, within = NonNegativeReals, doc = 'electricity generated from the PV sold to grid')
        model.H_b = Var(model.i, model.t, within = NonNegativeReals, doc = 'heat generated by boiler')
        model.annual_inv_PV = Var(within = NonNegativeReals, doc = 'investment cost of PV')
        model.annual_inv_B = Var(within = NonNegativeReals, doc = 'investment cost of boiler')
        model.annual_cost_grid = Var(within = NonNegativeReals, doc = 'cost of purchasing electricity from the grid')
        model.cost_night = Var(within=NonNegativeReals, doc = 'annual cost of buying electricity during the night')
        model.cost_day = Var(within=NonNegativeReals, doc = 'annual cost of buying electricity during the day')
        model.annual_income= Var(within = NonNegativeReals, doc = 'Income from selling electricity from PVs to the grid')
        model.annual_oc_PV = Var(within = NonNegativeReals, doc = 'total opex of PVs')
        model.annual_oc_b = Var(within = NonNegativeReals, doc = 'total opex of boilers')
        #model.area_PV = Var(model.i, within = NonNegativeReals, doc = 'total PV area installed')
        model.max_H_b = Var(model.i, within = NonNegativeReals, doc = 'maximum heat generated by boilers')
        
        model.storage_cap = Var(model.i, model.c, within=NonNegativeReals, doc= 'installed battery capacity')
        model.E_stored = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'electricity stored')
        model.E_grid_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'electricity charged from grid')
        model.E_PV_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'electricity charged from PVs')
        model.E_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'total electricity charged')
        model.E_discharge = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'total electricity discharged from battery')
        model.volume = Var(model.i, model.c, within=NonNegativeReals, doc = 'volume of battery installed')
        #model.E_disch_used = Var(model.i,model.t, model.c, within=NonNegativeReals, doc= 'electricity that is discharged and consequently consumed')
        #model.E_disch_sold = Var(model.i,model.t, model.c, within=NonNegativeReals, doc= 'electricity that is discharged and consequently consumed')
        #model.cycle = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'battery cycle number')
        model.annual_inv_S = Var(within = NonNegativeReals, doc = 'annual investment cost of batteries')
        model.annual_oc_S = Var(within = NonNegativeReals, doc = 'annual opex of batteries')
        
        model.carbon_cost = Var(within = NonNegativeReals, doc = 'carbon cost calculations')
        model.FIT_gen = Var(within = NonNegativeReals, doc = 'Total FIT generation income')
        
        '''Binary & integer variables'''
        model.panels_PV = Var(model.i, within = NonNegativeIntegers, bounds = (0,5000), doc = 'number of panels of PVs installed')
        model.X = Var(model.i, model.t, within=Binary, doc = '0 if electricity is bought from the grid')
        #model.YB = Var(model.i, within= Binary, doc = '1 if boiler is selected for the residential area')
        #model.Z = Var(model.i, within = Binary, doc = '1 if solar panels are selected for the residential area')
        model.Q1 = Var(model.i,model.t, model.c,  within = Binary,initialize = 1, doc='1 if charging')
        model.Q2 = Var(model.i,model.t, model.c, within = Binary, initialize = 0, doc='1 if discharging')
        model.W = Var(model.i, model.c, within = Binary, initialize = 1, doc = '1 if battery is selected')
        
                
        # =============================================================================
        #               '''Constraints: General Electricity and Heating'''
        # =============================================================================
        '''Satisfying the demand with generated power'''
        def electricity_balance(model,i,t):
            return model.E_load[i,t] == model.E_grid[i,t] + model.E_PV_used[i,t] + sum(model.E_discharge[i,t,c] for c in model.c)
        model.eb_constraint = Constraint(model.i, model.t, rule = electricity_balance, 
                                         doc = 'electricity consumed equals electricity generated')
        
        def heat_balance(model,i,t):
            return model.H_load[i,t] == model.H_b[i,t]
        model.hb_constraint = Constraint(model.i, model.t, rule = heat_balance, 
                                         doc = 'heat consumed = heat generated')
        
        '''Ensuring that electricity generated is used first before sold to the grid '''
        def buying_from_grid(model,i,t):
            return model.E_grid[i,t] <= model.E_load[i,t] * (1 - model.X[i,t])
        model.bfg_constraint = Constraint(model.i, model.t, rule = buying_from_grid, 
                                          doc = 'electricity bought from grid <= electricity load * (1 - binary)')
        
        def selling_to_grid(model,i,t):
            return model.E_PV_sold[i,t] <= 100 * model.X[i,t]
        model.stg_constraint = Constraint(model.i, model.t, rule = selling_to_grid, 
                                          doc = 'electricity sold <=  some upper bound * binary')
        
                
        # =============================================================================
        #                           '''Constraints: PVs'''
        # =============================================================================
        '''Power balance for PVs'''
        def PV_generation(model,i,t):
            return model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c) <= model.panels_PV[i] * model.panel_area * model.Irradiance[t] * model.n_PV
        model.pvg_constraint = Constraint(model.i, model.t, rule = PV_generation, 
                                          doc = 'total electricity generated by PV <= PV area * Irradiance * PV efficiency')
        
        '''Electricity generated by PVs not exceeding rated capacity'''
        def PV_rated_capacity(model,i,t):
            return model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c) <= model.panels_PV[i] * model.panel_capacity
        model.pvrc_constraint = Constraint(model.i, model.t, rule = PV_rated_capacity, 
                                          doc = 'total electricity generated by PV <= installed PV area * capacity of each panel/surface area of each panel')
        
        '''Investment cost of PVs per year'''
        def PV_investment(model):
            return (sum(model.cc_PV * model.panels_PV[i] * model.CRF for i in model.i))/4 == model.annual_inv_PV
        model.pvi_constraint = Constraint(rule = PV_investment, 
                                          doc = 'sum for all residences(capital cost * surface area/1.75 * CRF, N.B. 1.75 is the surface area per panel')
        
        '''Operation and maintenance cost of PVs per year'''
        def PV_operation_cost(model):
            return sum(((model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c))* model.oc_var_PV * self.days * self.interval) for i in model.i for t in model.t) \
                + sum((model.panels_PV[i] * model.oc_fixed_PV * (1/365) * self.days * model.panel_capacity) for i in model.i) == model.annual_oc_PV
        model.pvoc_constraint = Constraint(rule = PV_operation_cost, doc = 'sum of variable and fixed costs')        
        
        '''Roof area limitation'''
        def maximum_roof_area(model,i):
            return model.panels_PV[i] * model.panel_area <= model.max_roof_area[i]
        model.mra_constraint = Constraint(model.i, rule = maximum_roof_area, 
                                          doc = 'total PV area installed cannot exceed max roof area at each residence')
        
        '''Capacity limitation imposed by Smart Export Guarantee'''
        def SEG_capacity_limitation(model):
            return sum(model.panels_PV[i] * model.panel_capacity for i in model.i) <= model.max_capacity_PV
        model.segcl_constraint = Constraint(rule = SEG_capacity_limitation, 
                                            doc = 'sum of PVs installed in all houses cannot exceed maximum capacity given by the tariff regulations')
        
        # =============================================================================
        #                       '''Constraints: Boilers'''
        # =============================================================================
        #see if there is a better way of doing this
        '''Maximum boiler capacity'''
        def maximum_boiler_capacity(model,i,t):
            return model.max_H_b[i] >= model.H_b[i,t]
        model.mbc_constraint = Constraint(model.i, model.t, rule = maximum_boiler_capacity,
                                          doc = 'maximum boiler capacity is the maximum heat generated by boiler')
        
        '''Investment cost of boiler per year'''
        def boiler_investment(model):
            return (sum(model.cc_b * model.max_H_b[i] * model.CRF for i in model.i))/4 == model.annual_inv_B
        model.bi_constraint = Constraint(rule = boiler_investment,
                                         doc = 'investment cost = sum for all residences(capital cost * boiler capacity * CRF)') 
        
        '''Operation and maintenance cost of boilers per year'''
        def boiler_operation_cost(model):
            return sum(model.H_b[i,t] * self.interval * (model.price_gas/model.n_b) * self.days  for i in model.i for t in model.t) == model.annual_oc_b
        model.boc_constraint = Constraint(rule = boiler_operation_cost,
                                          doc = 'for all residences, seasons and periods, the heat generated by boiler * fuel price/thermal efficiency * no.of days' )
        
        # '''Boiler lower bound'''
        # def boiler_lower_bound(model,i):
        #     return model.max_H_b[i] >= 10 * model.YB[i]
        # model.blb_constraint = Constraint(model.i, rule = boiler_lower_bound,
        #                                   doc = 'boiler lower bound')
        
        #Add boiler upper bound if necessary

        # =============================================================================
        #                       '''Constraints: Batteries'''
        # =============================================================================
        '''Only 1 type of battery can be installed at each house'''
        def battery_type(model,i):
            return sum(model.W[i,c] for c in model.c) <= 1
        model.SC0 = Constraint(model.i, rule = battery_type)
        
        def type_bigM(model,i,c):
            return model.storage_cap[i,c] <= 100 * model.W[i,c]
        model.SC0a = Constraint(model.i,model.c, rule = type_bigM)
        
        '''Installed battery capacity calculation'''
        def installed_battery_cap(model,i,c):
            return model.storage_cap[i,c] == model.volume[i,c] * model.VED[c]
        model.SC1a = Constraint(model.i, model.c, rule = installed_battery_cap,
                               doc = 'capacity = volume available * volumetric energy density/delta(t)*binary')
        
        def volume_limit(model,i):
            return sum(model.volume[i,c] for c in model.c) <= model.VA[i]
        model.SC1b = Constraint(model.i, rule = volume_limit, 
                                doc = 'volume of battery cannot exceed maximum volume available')
        
        '''Maximum SoC limitation'''
        def battery_capacity1(model,i,t,c):
            return model.E_stored[i,t,c] <= model.storage_cap[i,c] * model.max_SoC[c]
        model.SC2a = Constraint(model.i, model.t, model.c, rule = battery_capacity1,
                                         doc = 'the energy in the storage cannot exceed its capacity based on the volume available and maximum state of charge')
 
        '''Maximum DoD limitation'''
        def battery_capacity2(model,i,t,c):
            return model.E_stored[i,t,c] >= model.storage_cap[i,c] * (1-model.max_DoD[c])
        model.SC2b = Constraint(model.i, model.t, model.c, rule = battery_capacity2,
                                         doc = 'the energy in the storage has to be greater than or equal to its capacity based on the volume available and maximum depth of discharge')
        
        '''Battery storage balance'''
        def storage_balance(model,i,t,c):
            if t > 1:
                return model.E_stored[i,t,c] == model.E_stored[i,t-1,c] + (model.E_charge[i,t,c]*model.n_ch[c]*self.interval) - (model.E_discharge[i,t,c]*self.interval/model.n_disch[c])
            else:
                return model.E_stored[i,t,c] == (model.E_charge[i,t,c]*model.n_ch[c]*self.interval) - (model.E_discharge[i,t,c]*self.interval/model.n_disch[c])
            #Constraint.Skip
        model.SC3 = Constraint(model.i, model.t, model.c, rule = storage_balance,
                                         doc = 'Energy stored at the beginning of each time interval is equal unused energy stored + energy coming in - Energy discharged')
        
        def discharge_condition(model,i,t,c):
            if t > 1:
                return model.E_discharge[i,t,c]*self.interval <= model.E_stored[i,t-1,c]
            else:
                return Constraint.Skip
        model.SC4 = Constraint(model.i, model.t, model.c, rule = discharge_condition,
                                doc = 'electricity discharged cannot exceed the energy stored from the previous time step')
        
        def fixing_start_and_end1(model,i,t,c):
            return model.E_stored[i,1,c] == model.E_stored[i,self.ft,c]
        model.SC5 = Constraint(model.i, model.t, model.c, rule = fixing_start_and_end1,
                                  doc = 'battery capacity at the start and end of the time horizon must be the same')
        
        '''Total electricity used to charge battery'''
        def total_charge(model,i,t,c):
            return model.E_charge[i,t,c] ==  model.E_grid_charge[i,t,c] + model.E_PV_charge[i,t,c]
        model.SC6 = Constraint(model.i, model.t, model.c, rule = total_charge,
                                         doc = 'total charging power = charging electricity from PVs + grid')
        
        '''Maximum State of Charge (SoC) limitation'''
        def charge_lim(model,i,t,c):
            return model.E_charge[i,t,c]*self.interval <= model.storage_cap[i,c] * 0.2
        model.SC7 = Constraint(model.i, model.t, model.c, rule = charge_lim,
                                          doc = '###')
        
        
        '''Maximum Depth of Discharge (DoD) limitation'''
        def discharge_lim(model,i,t,c):
            return model.E_discharge[i,t,c]*self.interval <= model.storage_cap[i,c] * 0.2
        model.SC8 = Constraint(model.i, model.t, model.c, rule = discharge_lim,
                                          doc = '###')
        
        '''Ensuring that battery cannot charge and discharge at same time'''
        def charging_bigM(model,i,t,c):
            return model.E_charge[i,t,c] <= 100 * model.Q1[i,t,c]
        model.SC9a = Constraint(model.i, model.t, model.c,rule = charging_bigM,
                               doc = 'Q1 will be 1 if charging')
        
        def discharging_bigM(model,i,t,c):
            return model.E_discharge[i,t,c] <= 100 * model.Q2[i,t,c]
        model.SC9b = Constraint(model.i, model.t, model.c,rule = discharging_bigM,
                               doc = 'Q2 will be 1 if discharging')
        
        def chg_disch_binaries(model,i,t,c):
            return model.Q1[i,t,c] + model.Q2[i,t,c] == 1 
        model.SC9c = Constraint(model.i, model.t, model.c, rule = chg_disch_binaries, 
                                    doc = 'only 1 binary allowed to be 1 at a time')
        
        '''Investment Cost of Batteries per year'''
        def storage_investment(model):
            return sum(sum(model.storage_cap[i,c] * model.cc_storage[c] for c in model.c) * model.CRF/4 for i in model.i) == model.annual_inv_S
        model.SC10 = Constraint(rule = storage_investment,
                                         doc = 'investment cost = capacity (kWh) * cost per kWh * CRF')

        '''Operational and Maintenance cost of batteries per year'''
        def storage_operation_cost(model):
            return sum(sum(model.storage_cap[i,c] *(1/self.interval) * model.om_storage[c] for c in model.c) * (1/365) * self.days for i in model.i) == model.annual_oc_S
        model.SC11 = Constraint(rule = storage_operation_cost,
                                         doc = 'opex = capacity (kWh) * cost per kWh per year')
        
        # =============================================================================
        #                    '''Constraints: General Costs'''
        # =============================================================================
        
        '''total cost of buying electricity during the day'''
        def grid_day(model):
            return model.cost_day == sum(((model.E_grid[i,t] + sum(model.E_grid_charge[i,t,c] for c in model.c)) * self.interval * pg_day * self.days) for i in model.i for t in model.t_day)
        model.GCa = Constraint(rule = grid_day)
        
        '''total cost of buying electricity during the night'''
        def grid_day(model):
            return model.cost_night == sum(((model.E_grid[i,t] + sum(model.E_grid_charge[i,t,c] for c in model.c)) * self.interval * pg_night * self.days) for i in model.i for t in model.t_night)
        model.GCb = Constraint(rule = grid_day)
        
        '''Cost of buying electricity per year'''
        def annual_electricity_cost(model):
            return model.annual_cost_grid == model.cost_night + model.cost_day
        #sum(((model.E_grid[i,t] + model.E_grid_charge[i,t,c]) * self.interval * model.price_grid * self.days) for i in model.i for t in model.t for c in model.c)
        model.GC1 = Constraint(rule = annual_electricity_cost,
                                          doc = 'annual electricity cost = sum for all residences, seasons and periods(electricity bought at i,m,p * price')
        
        '''carbon cost for the grid'''
        def seasonal_carbon_cost(model):
            return model.carbon_cost == sum(((model.E_grid[i,t] + sum(model.E_grid_charge[i,t,c] for c in model.c)) * self.interval * model.carbon_grid * model.c_carbon * self.days) for i in model.i for t in model.t)
        model.GC2 = Constraint(rule = seasonal_carbon_cost)
        
        '''Income from selling electricity to the grid'''
        def income_electricity_sold(model):
            return model.annual_income == sum((model.E_PV_sold[i,t] * self.interval * model.SEG * self.days) for i in model.i for t in model.t)
        model.ies_constraint = Constraint(rule = income_electricity_sold,
                                         doc = 'income = sum for all residences, seasons and periods(electricity sold * smart export guarantee tariff')
        
        '''FIT generation'''
        def FIT(model):
            return model.FIT_gen == sum(((model.E_PV_used[i,t] + model.E_PV_sold[i,t] + sum(model.E_PV_charge[i,t,c] for c in model.c))* self.interval * FIT_tariff * self.days) for i in model.i for t in model.t)
        model.I1 = Constraint(rule = FIT)
        
        def objective_rule(model):
            return (model.annual_cost_grid + model.carbon_cost + model.annual_inv_PV + model.annual_inv_B + model.annual_inv_S + model.annual_oc_PV + model.annual_oc_b + model.annual_oc_S - model.annual_income - model.FIT_gen)
        model.objective = Objective(rule= objective_rule, sense = minimize, doc = 'objective function')


        return model



