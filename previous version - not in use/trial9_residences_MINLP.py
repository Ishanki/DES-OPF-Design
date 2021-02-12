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
31/20/2020
    > Battery capacities returned in kWh (constraints adjusted accordingly)
'''

U_BOUND = 100000

class Residential_Seasonal_DES(object):
    def __init__(self, house, df, days, interval, ft, \
                 irrad, df_scalars, df_roof, elec_house, heat_house, \
                     df_batteries, df_volume, battery, \
                         ):
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
    
        
    def DES_MINLP(self):
        
        model = ConcreteModel()
        
        model.i = Set(initialize= self.house, doc= 'residential users')
        #model.periods2 = len(list(self.df[1])) #used to create a RangeSet for model.p
        model.t = RangeSet(self.ft, doc= 'periods / timestamps')
        model.c = Set(initialize = self.battery, doc='types of batteries available')
        
        house_num = list(range(1,(len(self.house))+1))
        
        print("data is now loading into loops")

        model.E_load = Param(model.i, model.t, initialize = self.elec_house, doc = 'electricity load')
        model.H_load = Param(model.i, model.t, initialize = self.heat_house, doc = 'heating load')
        model.Irradiance = Param(model.t, initialize = self.irrad, doc = 'solar irradiance')
        
        ra = {}
        for n, h in zip(range(len(self.house)), self.house):
            ra[h] = self.df_roof.iat[n,1]
        
        model.max_roof_area = Param(model.i, initialize = ra, doc = 'maximum roof surface area available')
        
        irate = self.df_scalars.iat[0,1] #, doc = 'interest rate')
        nyears = self.df_scalars.iat[1,1] #, doc = 'project lifetime')
        price_grid = self.df_scalars.iat[2,1] #, doc = 'electricity price in £ per kWh')
        price_gas =self.df_scalars.iat[3,1] #, doc = 'price of gas in £ per kWh')
        carbon_grid = self.df_scalars.iat[4,1] #, doc = 'carbon intensity of grid electricity in kg/kWh')
        cc_PV = self.df_scalars.iat[5,1] #, doc = 'capital cost of PV in £ per panel (1.75 m2)')
        n_PV = self.df_scalars.iat[6,1] #, doc = 'efficiency of the PV')
        oc_fixed_PV = self.df_scalars.iat[7,1] #, doc = 'fixed operational cost of PV in £ per kW per year')
        oc_var_PV = self.df_scalars.iat[8,1] #, doc = 'variable operational cost of PV in £ per kWh')
        TEx = self.df_scalars.iat[9,1] #, doc = 'Tariff for exporting in £ per kWh')
        cc_b = self.df_scalars.iat[10,1] #, doc = 'capital cost of boiler per kWh')
        n_b = self.df_scalars.iat[11,1] #, doc = "thermal efficiency of the boiler")
        panel_area = self.df_scalars.iat[12,1] #, doc = 'surface area per panel in m2')
        panel_capacity = self.df_scalars.iat[13,1] #, doc = 'rated capacity per panel in kW')
        max_capacity_PV = self.df_scalars.iat[14,1] #, doc = 'maximum renewables capacity the DES is allowed to have as per SEG tariffs')
        c_carbon = self.df_scalars.iat[15,1]#, doc = 'fake carbon cost for the grid')
        TGen = self.df_scalars.iat[16,1] #, doc = 'generation tariff')
        PF = self.df_scalars.iat[17,1] # doc = 'PV inverter power factor')
        n_inv = self.df_scalars.iat[18,1] # doc = 'inverter efficiency')
        #GCA = self.df_scalars.iat[19,1] #doc = 'states whether batteries can be charged from the grid or not'
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
        print(mDoD1)
        
        model.max_DoD = Param(model.c, initialize = mDoD1, doc='max depth of discharge')

        mSoC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            mSoC1[c] = self.df_batteries.iat[n,4]
        print(mSoC1)
        
        model.max_SoC = Param(model.c, initialize = mSoC1, doc='max state of charge')
        
        CCC1 = {}
        for n, c in zip(range(len(self.battery)), self.battery):
            CCC1[c] = self.df_batteries.iat[n,5]
        print(CCC1)
        
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
        
        print("....battery parameters processed.")    
        
        
        CRF = (irate * ((1 + irate)**nyears))/(((1 + irate)**nyears)-1)
        print("this is CRF:")
        print(CRF)
        
        # =============================================================================
        #                   '''Variables and Initialisations'''
        # =============================================================================
        model.panels_PV = Var(model.i, within = NonNegativeIntegers, bounds = (0,5000), initialize = 16, 
                              doc = 'number of panels of PVs installed')
        
        EPVu_init = {}
        for i,t in zip(model.i,model.t):
            EPVu_init[i,t] = model.panels_PV[i].value * panel_area * model.Irradiance[t] * n_PV
        model.E_PV_used = Var(model.i, model.t, within = NonNegativeReals, initialize = EPVu_init,
                              doc = 'electricity generated from the PV used at the house')
        
        model.E_PV_sold = Var(model.i, model.t, within = NonNegativeReals, initialize = 0,
                              doc = 'electricity generated from the PV sold to grid')
        
        AIPV_init = (sum(cc_PV * model.panels_PV[i].value * CRF for i in model.i))/4
        model.annual_inv_PV = Var(within = NonNegativeReals, initialize = AIPV_init,
                                  bounds=(0,U_BOUND), doc = 'investment cost of PV')
        Egrid_init = {}
        for i,t in zip(model.i,model.t):
            Egrid_init[i,t] = model.E_load[i,t] - model.E_PV_used[i,t].value
        model.E_grid = Var(model.i, model.t, within = NonNegativeReals, initialize = Egrid_init,
                           doc = 'electricity imported from the grid in kW')
        
        model.E_PV_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, 
                                initialize = 0, doc= 'electricity charged from PVs')
    
        AOPV_init = {}
        AOPV_init = sum(model.E_PV_used[i,t].value * oc_var_PV * self.days * self.interval \
                        for i in model.i for t in model.t if model.E_PV_used[i,t].value != None) \
            + sum((model.panels_PV[i].value * oc_fixed_PV * (1/365) * self.days * panel_capacity) \
                        for i in model.i if model.E_PV_used[i,t].value != None)
        model.annual_oc_PV = Var(within = NonNegativeReals, initialize = AOPV_init,
                                 bounds=(0,U_BOUND), doc = 'total opex of PVs')
        
        EI_init = sum((model.E_PV_sold[i,t].value * self.interval * TEx * self.days) for i in model.i for t in model.t)
        model.export_income= Var(within = NonNegativeReals, initialize = EI_init,
                                 bounds=(0,U_BOUND), doc = 'Income from selling electricity from PVs to the grid')
        
        GI_init = sum(((model.E_PV_used[i,t].value)* self.interval * TGen * self.days) \
                      for i in model.i for t in model.t for c in model.c if model.E_PV_used[i,t].value != None)
        model.gen_income = Var(within = NonNegativeReals, initialize = GI_init,
                               bounds=(0,U_BOUND), doc = 'income from generating renewable energy')
        
        #model.area_PV = Var(model.i, within = NonNegativeReals, doc = 'total PV area installed')
        
        model.H_b = Var(model.i, model.t, within = NonNegativeReals, initialize = self.heat_house,
                        doc = 'heat generated by boiler')
        
        MHB_init = {}
        for i in model.i:
            MHB_init[i] = sum(model.H_b[i,t].value for t in model.t)/(self.ft - (self.ft/2))
        model.max_H_b = Var(model.i, within = NonNegativeReals, initialize = MHB_init,
                            doc = 'maximum heat generated by boilers')
        
        AIB_init = (sum(cc_b * model.max_H_b[i].value * CRF for i in model.i))/4 
        model.annual_inv_B = Var(within = NonNegativeReals, initialize = AIB_init,
                                 bounds=(0,U_BOUND), doc = 'investment cost of boiler')
        
        AOB_init = sum(model.H_b[i,t].value * self.interval * (price_gas/n_b) * self.days  for i in model.i for t in model.t)
        model.annual_oc_b = Var(within = NonNegativeReals, initialize = AOB_init,
                                bounds=(0,U_BOUND), doc = 'total opex of boilers')
        
        Vol_init = {}
        for i in (model.i):
            Vol_init[i] = model.VA[i]
        model.volume = Var(model.i, within=NonNegativeReals, initialize = Vol_init,
                           doc = 'volume of battery installed')
        
        SC_init = {}
        for i,c in zip(model.i,model.c):
            SC_init[i,c] = model.VA[i] * model.VED[c]
        model.storage_cap = Var(model.i, model.c, within=NonNegativeReals, initialize = SC_init,
                                doc= 'installed battery capacity in kWh')
        
        model.E_discharge = Var(model.i, model.t, model.c, within=NonNegativeReals, initialize = 0, 
                                doc= 'total electricity discharged from battery')
        
        EgCh_init = {}
        for i,t,c in zip(model.i,model.t,model.c):
            EgCh_init[i,t,c] = model.storage_cap[i,c].value * (1-model.max_DoD[c])/self.interval
        model.E_grid_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, initialize = EgCh_init,
                                  doc= 'electricity charged from grid')
        
        ECh_init = {}
        for i,t,c in zip(model.i,model.t,model.c):
            ECh_init[i,t,c] = model.E_grid_charge[i,t,c].value + model.E_PV_charge[i,t,c].value
        model.E_charge = Var(model.i, model.t, model.c, within=NonNegativeReals, initialize = ECh_init,
                             doc= 'total electricity charged')
        ES_init = {}
        for i,t,c in zip(model.i,model.t,model.c):
            ES_init[i,t,c] = (model.E_charge[i,t,c].value*model.n_ch[c]*self.interval) - (model.E_discharge[i,t,c].value*self.interval/model.n_disch[c])
        model.E_stored = Var(model.i, model.t, model.c, within=NonNegativeReals, initialize = ES_init,
                             doc= 'electricity stored')
        
        #model.E_disch_used = Var(model.i,model.t, model.c, within=NonNegativeReals, doc= 'electricity that is discharged and consequently consumed')
        #model.E_disch_sold = Var(model.i,model.t, model.c, within=NonNegativeReals, doc= 'electricity that is discharged and consequently consumed')
        #model.cycle = Var(model.i, model.t, model.c, within=NonNegativeReals, doc= 'battery cycle number')
        #AIS_init = sum(model.storage_cap[i,c].value * model.cc_storage[c] * CRF/4 for i in model.i for c in model.c)
        model.annual_inv_S = Var(within = NonNegativeReals, #initialize = AIS_init,
                                 bounds=(0,U_BOUND), doc = 'annual investment cost of batteries')
        
        AOS_init = sum(model.storage_cap[i,c].value *(1/self.interval) * model.om_storage[c] * (1/365) * self.days for i in model.i for c in model.c if model.storage_cap[i,c].value != None)
        model.annual_oc_S = Var(within = NonNegativeReals, initialize = AOS_init,
                                bounds=(0,U_BOUND), doc = 'annual opex of batteries')
        
        CC_init = sum(((model.E_grid[i,t].value + model.E_grid_charge[i,t,c].value) * self.interval * carbon_grid * c_carbon * self.days) \
                      for i in model.i for t in model.t for c in model.c if model.E_grid_charge[i,t,c].value != None)
        model.carbon_cost = Var(within = NonNegativeReals, initialize = CC_init,
                                bounds=(0,U_BOUND), doc = 'carbon cost calculations')
        
        InC_init = {}
        for i,t in zip(model.i,model.t):
            InC_init[i] = (model.panels_PV[i].value*panel_capacity)/PF
        model.inv_cap = Var(model.i, within = NonNegativeReals, initialize = InC_init,
                            doc = 'inverter capacity in kVA')
        
        SI_init = {}
        for i,t in zip(model.i,model.t):
            SI_init[i,t] = (model.E_PV_sold[i,t].value)/PF
                                     #   + model.E_AC_disch[i,t,c].value)/PF
        model.S_inv = Var(model.i, model.t, bounds = (None,None), initialize = SI_init
                          )
        
        PI_init = {}
        for i,t in zip(model.i,model.t):
            PI_init[i,t] = model.E_PV_sold[i,t].value
        model.P_inv = Var(model.i, model.t, within = NonNegativeReals, initialize = PI_init
                          )
        
        QI_init = {}
        for i,t in zip(model.i,model.t):
            QI_init[i,t] = sqrt((model.S_inv[i,t].value**2) - (model.P_inv[i,t].value**2) + 0.0001)
        model.Q_gen = Var(model.i, model.t, bounds=(None,None), initialize = QI_init,
                          doc = 'reactive power from PV inverters')
        
        ACG = sum(((model.E_grid[i,t].value + model.E_grid_charge[i,t,c].value) * self.interval * price_grid * self.days) \
                  for i in model.i for t in model.t for c in model.c if model.E_grid_charge[i,t,c].value != None)
        model.annual_cost_grid = Var(within = NonNegativeReals, bounds=(0,U_BOUND), doc = 'cost of purchasing electricity from the grid')
        
        '''Binary variables'''
        model.X = Var(model.i, model.t, within=Binary, initialize = 0, doc = '0 if electricity is bought from the grid')
        #model.YB = Var(model.i, within= Binary, doc = '1 if boiler is selected for the residential area')
        #model.Z = Var(model.i, within = Binary, initialize = 1, doc = '1 if solar panels are selected for the residential area')
        
        model.Q1 = Var(model.i,model.t, model.c,  within = Binary, initialize = 1, doc='1 if charging')
        model.Q2 = Var(model.i,model.t, model.c, within = Binary, initialize = 0, doc='1 if discharging')
        
        
        # =============================================================================
        #               '''Constraints: General Electricity and Heating'''
        # =============================================================================
        '''Satisfying the demand with generated power'''
        def electricity_balance(model,i,t):
            return model.E_load[i,t] == model.E_grid[i,t] + model.E_PV_used[i,t] + model.E_discharge[i,t,c]
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
            return model.E_PV_sold[i,t] <= 10000 * model.X[i,t]
        model.stg_constraint = Constraint(model.i, model.t, rule = selling_to_grid, 
                                          doc = 'electricity sold <=  some upper bound * binary')
        
        # =============================================================================
        #                           '''Constraints: PVs'''
        # =============================================================================
        '''Power balance for PVs'''
        def PV_generation(model,i,t,c):
            return model.E_PV_used[i,t] + model.E_PV_sold[i,t] + model.E_PV_charge[i,t,c] <= model.panels_PV[i] * panel_area * model.Irradiance[t] * n_PV
        model.PV1 = Constraint(model.i, model.t, model.c, rule = PV_generation, 
                                          doc = 'total electricity generated by PV <= PV area * Irradiance * PV efficiency')
        
        '''Electricity generated by PVs not exceeding rated capacity'''
        def PV_rated_capacity(model,i,t,c):
            return model.E_PV_used[i,t] + model.E_PV_sold[i,t] + model.E_PV_charge[i,t,c] <= model.panels_PV[i] * panel_capacity
        model.PV2 = Constraint(model.i, model.t, model.c, rule = PV_rated_capacity, 
                                          doc = 'total electricity generated by PV <= installed PV area * capacity of each panel/surface area of each panel')
        
        '''Investment cost of PVs per year'''
        def PV_investment(model):
            return (sum(cc_PV * model.panels_PV[i] * CRF for i in model.i))/4 == model.annual_inv_PV
        model.PV3 = Constraint(rule = PV_investment, 
                                          doc = 'sum for all residences(capital cost * surface area/1.75 * CRF, N.B. 1.75 is the surface area per panel')
        
        '''Operation and maintenance cost of PVs per year'''
        def PV_operation_cost(model):
            return sum(((model.E_PV_used[i,t] + model.E_PV_sold[i,t] + model.E_PV_charge[i,t,c])* oc_var_PV * self.days * self.interval) for i in model.i for t in model.t for c in model.c) \
                + sum((model.panels_PV[i] * oc_fixed_PV * (1/365) * self.days * panel_capacity) for i in model.i) == model.annual_oc_PV
        model.PV4 = Constraint(rule = PV_operation_cost, doc = 'sum of variable and fixed costs')        
        
        '''Roof area limitation'''
        def maximum_roof_area(model,i):
            return model.panels_PV[i] * panel_area <= model.max_roof_area[i]
        model.PV5 = Constraint(model.i, rule = maximum_roof_area, 
                                          doc = 'total PV area installed cannot exceed max roof area at each residence')
        
        '''Capacity limitation imposed by Smart Export Guarantee'''
        def capacity_limitation(model):
            return sum(model.panels_PV[i] * panel_capacity for i in model.i) <= max_capacity_PV
        model.PV6 = Constraint(rule = capacity_limitation, 
                                            doc = 'sum of PVs installed in all houses cannot exceed maximum capacity given by the tariff regulations')
        
        # =============================================================================
        #                       '''Constraints: Boilers'''
        # =============================================================================
        #see if there is a better way of doing this
        '''Maximum boiler capacity'''
        def maximum_boiler_capacity(model,i,t):
            return model.max_H_b[i] >= model.H_b[i,t]
        model.H2 = Constraint(model.i, model.t, rule = maximum_boiler_capacity,
                                          doc = 'maximum boiler capacity is the maximum heat generated by boiler')
        
        '''Investment cost of boiler per year'''
        def boiler_investment(model):
            return (sum(cc_b * model.max_H_b[i] * CRF for i in model.i))/4 == model.annual_inv_B
        model.H3 = Constraint(rule = boiler_investment,
                                         doc = 'investment cost = sum for all residences(capital cost * boiler capacity * CRF)') 
        
        '''Operation and maintenance cost of boilers per year'''
        def boiler_operation_cost(model):
            return sum(model.H_b[i,t] * self.interval * (price_gas/n_b) * self.days  for i in model.i for t in model.t) == model.annual_oc_b
        model.H4 = Constraint(rule = boiler_operation_cost,
                                          doc = 'for all residences, seasons and periods, the heat generated by boiler * fuel price/thermal efficiency * no.of days' )
        
# =============================================================================
#         '''Boiler lower bound'''
#         def boiler_lower_bound(model,i):
#             return model.max_H_b[i] >= 10 * model.YB[i]
#         model.blb_constraint = Constraint(model.i, rule = boiler_lower_bound,
#                                           doc = 'boiler lower bound')
# =============================================================================
        
        #Add boiler upper bound if necessary

        # =============================================================================
        #                       '''Constraints: Batteries'''
        # =============================================================================
        
        '''Installed battery capacity calculation'''
        def installed_battery_cap(model,i,c):
            return model.storage_cap[i,c] == model.volume[i] * (model.VED[c])
        model.SC1a = Constraint(model.i, model.c, rule = installed_battery_cap,
                               doc = 'capacity = volume available * volumetric energy density/delta(t)*binary')
        
        def volume_limit(model,i,c):
            return model.volume[i] <= model.VA[i]
        model.SC1b = Constraint(model.i,model.c, rule = volume_limit, 
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
            if t >= 2:
                return model.E_stored[i,t,c] == model.E_stored[i,t-1,c] + (model.E_charge[i,t,c]*model.n_ch[c]*self.interval) - (model.E_discharge[i,t,c]*self.interval/model.n_disch[c])
            else:
                return model.E_stored[i,t,c] == (model.E_charge[i,t,c]*model.n_ch[c]*self.interval) - (model.E_discharge[i,t,c]*self.interval/model.n_disch[c])
            #Constraint.Skip
        model.SC3 = Constraint(model.i, model.t, model.c, rule = storage_balance,
                                         doc = 'Energy stored at the beginning of each time interval is equal unused energy stored + energy coming in - Energy discharged')
        
        def discharge_condition(model,i,t,c):
            if t>=2:
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
            return sum(model.storage_cap[i,c] * model.cc_storage[c] * CRF/4 for i in model.i for c in model.c) == model.annual_inv_S
        model.SC10 = Constraint(rule = storage_investment,
                                         doc = 'investment cost = volume available * volume density * cost per kWh * CRF')

        '''Operational and Maintenance cost of batteries per year'''
        def storage_operation_cost(model):
            return sum(model.storage_cap[i,c] *(1/self.interval) * model.om_storage[c] * (1/365) * self.days for i in model.i for c in model.c) == model.annual_oc_S
        model.SC11 = Constraint(rule = storage_operation_cost,
                                         doc = 'opex = volume available * volume density * cost per kWh per year')
        
        
        # =============================================================================
        #                    '''Constraints: Inverter'''
        # =============================================================================
        '''inverter capacity calculation using power factor'''
        def icap(model,i,t,c):
            return model.inv_cap[i] == (model.panels_PV[i] * panel_capacity)/PF
        #(model.panels_PV[i] * panel_capacity)/PF
        model.INV0 = Constraint(model.i,model.t,model.c, rule = icap, 
                                doc = 'total active power output of inverter/power factor')
        
        '''Inverter limits'''
        def inv_lim(model,i,t):
            return (model.E_PV_used[i,t] + model.E_PV_sold[i,t]  + model.E_discharge[i,t,c])/PF <= model.inv_cap[i]
        model.INV1 = Constraint(model.i, model.t, rule = inv_lim)
        
        '''inverter active power'''
        def inv_act(model,i,t,c):
            return model.P_inv[i,t] == model.E_PV_sold[i,t]
        #model.E_PVAC_sold[i,t]
        #(model.E_PV_used[i,t] + model.E_PV_sold[i,t]  + model.E_discharge[i,t,c])
        model.INV5 = Constraint(model.i, model.t, model.c, rule = inv_act)
        
        '''inverter apparent power'''
        def inv_apparent(model,i,t,c):
            return model.S_inv[i,t] == model.E_PV_sold[i,t]/PF
        #model.E_PVAC_sold[i,t]/PF
        #(model.E_PV_used[i,t] + model.E_PV_sold[i,t]  + model.E_discharge[i,t,c])/PF
        model.INV4 = Constraint(model.i,model.t, model.c, rule = inv_apparent)
                
        '''reactive power generation'''
        def Inv_q_gen(model,i,t):
            return (model.Q_gen[i,t]**2) == (((model.P_inv[i,t])**2)*(1-(PF**2))/(PF**2))
        #sqrt((model.S_inv[i,t]**2) - (model.P_inv[i,t]**2) + 0.0001)
        model.INV6 = Constraint(model.i,model.t, rule = Inv_q_gen)
        
        # =============================================================================
        #                    '''Constraints: General Costs'''
        # =============================================================================
        
        model.t_night = RangeSet(1,13)
        model.t_day = RangeSet(14,48)
        model.cost_night = Var(within=NonNegativeReals)
        model.cost_day = Var(within=NonNegativeReals)
        
        '''total cost of buying electricity during the day'''
        def grid_day(model):
            return model.cost_day == sum(((model.E_grid[i,t] + model.E_grid_charge[i,t,c]) * self.interval * pg_day * self.days) for i in model.i for t in model.t_day for c in model.c)
        model.GCa = Constraint(rule = grid_day)
        
        '''total cost of buying electricity during the night'''
        def grid_night(model):
            return model.cost_night == sum(((model.E_grid[i,t] + model.E_grid_charge[i,t,c]) * self.interval * pg_night * self.days) for i in model.i for t in model.t_night for c in model.c)
        model.GCb = Constraint(rule = grid_night)
        
        '''Cost of buying electricity per year'''
        def annual_electricity_cost(model):
            return model.annual_cost_grid == model.cost_night + model.cost_day
        #sum(((model.E_grid[i,t] + model.E_grid_charge[i,t,c]) * self.interval * model.price_grid * self.days) for i in model.i for t in model.t for c in model.c)
        model.GC1 = Constraint(rule = annual_electricity_cost,
                                          doc = 'annual electricity cost = sum for all residences, seasons and periods(electricity bought at i,m,p * price')
        
        '''carbon cost for the grid'''
        def seasonal_carbon_cost(model):
            return model.carbon_cost == sum(((model.E_grid[i,t] + model.E_grid_charge[i,t,c]) * self.interval * carbon_grid * c_carbon * self.days) for i in model.i for t in model.t for c in model.c)
        model.GC2 = Constraint(rule = seasonal_carbon_cost)
        
        '''Income from selling electricity to the grid'''
        def income_electricity_sold(model):
            return model.export_income == sum((model.E_PV_sold[i,t] * self.interval * TEx * self.days) for i in model.i for t in model.t)
        model.GC3 = Constraint(rule = income_electricity_sold,
                                         doc = 'income = sum for all residences, seasons and periods(electricity sold * smart export guarantee tariff')
        
        '''generation income'''
        def gen_tariff(model):
            return model.gen_income == sum(((model.E_PV_used[i,t] + model.E_PV_sold[i,t] + model.E_PV_charge[i,t,c])* self.interval * TGen * self.days) for i in model.i for t in model.t for c in model.c)
        model.GC4 = Constraint(rule = gen_tariff)
        
# =============================================================================
#         model.Net_P = Var(model.i, model.t, bounds =(None,None))
#         def check_P(model,i,t,c):
#             return model.Net_P[i,t] == model.E_PVAC_sold[i,t] - model.E_grid[i,t] - model.E_grid_charge[i,t,c]
#         model.check = Constraint(model.i, model.t, model.c, rule = check_P)
# =============================================================================
        
        def objective_rule(model):
            return (model.annual_cost_grid + model.carbon_cost + model.annual_inv_PV + model.annual_inv_B + model.annual_inv_S + model.annual_oc_PV + model.annual_oc_b + model.annual_oc_S - model.export_income - model.gen_income)
        model.objective = Objective(rule= objective_rule, sense = minimize, doc = 'objective function')

        
        return model
    
