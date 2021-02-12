from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from time import perf_counter
from pandas import ExcelWriter
from t9_MINLP import ResidentialSeasonalDES
from trial9_DES_OPF import DES_OPF

'''
This is the run file for the model that inputs data to the
DES (MINLP) and OPF (NLP), and links them.
Outputs include design capacities, operational schedule across all 4 seasons,
power flows, and voltage/angles. 
    
    > If PF is not unity, please change Q_GEN constraint as it is curently set to 0
    > Big M is 100, no individual powers reach that value for this case study.
    > The models are tested using the DES network in Morvaj et al. (2016)
'''
m = ConcreteModel()

# =============================================================================
#                              '''Inputs here'''
# =============================================================================
house = ['A', 'B', 'C','D','E']
loads_file_name = "Loads_houses_morvaj.xls" #"summer.xls" #"Loads_houses.xls"
parameters_file_name = "Parameters.xlsx" #NOTE: slightly higher PV efficiency 
irrad_file_name = "Parameters.xlsx"
# Battery options given:
battery = ['LI']#,'SS'] # LI - lithium ion, SS - Sodium-Sulphur
# duration of time interval:
interval = 1  # 0.5 
# State the total amount of timestamps here:
m.timestamp = RangeSet(24) 
# Number of seasons, adjust the number accordingly:
m.S = RangeSet(4) 
# State the final time step e.g. 24 or 48:
ft = 24  #48
# Days in each season, 1-winter, 2-spring, 3-summer, 4-autumn:
d = {1:90, 2:92, 3:92, 4:91} 
# Results filename, looks incomplete but will attach the string of season later
results_file_name = "MILP_season" 
results_file_suffix = '.xlsx'

KEEP_BATTERY = 1
KEEP_PV = 1

# Switch OPF functionality off using 0 (when running MILP), else use 1:
# N.B. default solver is CPLEX
KEEP_OPF = 1

# To run NLP or MINLP instead of MILP (note KEEP_OPF must be equal to 1):
# N.B. default solver for NLP is CONOPT, MINLP is SBB
RUN_NLP = 0
RUN_MINLP = 1

# Grid related parameters (please ensure these are correct if KEEP_OPF = 1)
df_grid = pd.read_excel(parameters_file_name, sheet_name = "Grid") #required for linking constraints
S_base = df_grid.iat[5,1]  

#distribution network
slack = 1
nodes = [1,2,3,4,5,6,7]
bus_connectivity = {1:(1,2),
                    2:(2,3),
                    3:(3,4),
                    4:(3,5),
                    5:(5,6),
                    6:(6,7),
                    }

house_bus = {2:'A',
             4: 'B',
             5: 'C',
             6: 'D',
             7: 'E',
             }

gen = [2,4,5,6,7] #generating nodes apart from slack

# =============================================================================
#                           '''Data Processing'''
# =============================================================================
#this is here to easily loop the data imported from excel
house_num = list(range(1,(len(house))+1))

start = perf_counter()

# =============================================================================
# '''The residential block'''
# =============================================================================
def seasonal_data(m,s):

    m.df = {} #represents the dataframe for electricity loads
    m.dfh = {} #represents the dataframe for heating loads
    # Looping through the loads w.r.t each season s and house from excel
    for n, h in zip(house_num, house):
        sheet_n1 = ("Elec_"+str(n))
        m.df[n] = pd.read_excel(loads_file_name, sheet_name = sheet_n1)
        m.df[n].set_index(h, inplace = True)
        m.df[n] = m.df[n].iloc[s-1]
        sheet_n2 = ("Heat_"+str(n))
        m.dfh[n] = pd.read_excel(loads_file_name, sheet_name = sheet_n2)
        m.dfh[n].set_index(h, inplace = True)
        m.dfh[n] = m.dfh[n].iloc[s-1]
        string1 = "this is electricity for season "
        string2 = " for house "
        #print(string1 + str(s) + string2 + str(h))
        #print(m.df[n])
        #print(b.df[n].get(2))
        
    # Assigning loaded dataframes into dictionaries, now w.r.t house h and time t
    # print("data is now loading into loops")
    m.elec_house = {}
    m.heat_house = {}
    for n, h in zip(house_num, house):
        for t in (m.df[n].index):
            m.elec_house[h, t] = round(float(m.df[n][t]/interval),3) 
            m.heat_house[h, t] = round(float(m.dfh[n][t]/interval),3)
                
    #print(m.elec_house['h1', 34])
    #print(m.heat_house['h2',34]) 
    
    # Loading other time-dependent parameters
    m.dfi = pd.read_excel(irrad_file_name, sheet_name = "Irradiance")
    m.dfi.set_index("Irrad", inplace = True)
    m.dfi = m.dfi.iloc[s-1]
    string3 = "this is irradiance data for season "
    #print(string3 + str(s))
    #print(m.dfi)
    
    m.irrad = {}
    for t in m.dfi.index:
        m.irrad[t] = float(m.dfi[t])

    m.days = d[s]
    
    # These dataframes are fed directly into the ResidentialSeasonalDES class
    df_scalars = pd.read_excel(parameters_file_name, sheet_name = "Res_Scalars")
    #print(df_scalars)
    df_roof = pd.read_excel(parameters_file_name, sheet_name = "Roof_areas_res")
    df_batteries = pd.read_excel(parameters_file_name, sheet_name = "batteries")
    df_volume = pd.read_excel(parameters_file_name, sheet_name = "Stor_vol_res")
    df_grid = pd.read_excel(parameters_file_name, sheet_name = "Grid")
    df_network = pd.read_excel(parameters_file_name, sheet_name = "Network")
    
    # The object m.full_model is created from the ResidentialSeasonalDES class
    m.full_model = ResidentialSeasonalDES(house=house, 
                                      df=m.df, 
                                      days=m.days,
                                      interval=interval, 
                                      ft=ft, 
                                      irrad=m.irrad,
                                      df_scalars=df_scalars,
                                      df_roof=df_roof,
                                      elec_house=m.elec_house,
                                      heat_house=m.heat_house,
                                      df_batteries=df_batteries,
                                      df_volume=df_volume, 
                                      battery=battery,
                                      KEEP_BATTERY=KEEP_BATTERY,
                                      KEEP_PV=KEEP_PV,
                                      )
    
    # Assigning the DES_MINLP method in the full_model object to the Pyomo model m
    m = m.full_model.DES_MINLP()
    
    # Deactivating the individual objectives in each block
    m.objective.deactivate()  
    
    # This is the free variable for total cost which the objective minimises
    m.cost = Var(bounds = (None, None))
    # m.cost = Var(bounds = (None, 7000)) #Octeract bound changes when this is included
    
    #This is the objective function rule that combines the costs for that particular season
    def rule_objective(m):
        expr = 0
        expr += (m.annual_cost_grid + m.carbon_cost + m.annual_inv_PV + \
                 m.annual_inv_B + m.annual_inv_S + \
                     m.annual_oc_PV + m.annual_oc_b + m.annual_oc_S \
                         - m.export_income - m.gen_income)
        #expr += m.annual_cost_grid
        #expr += (sum(m.E_discharge[i,t,c] for i in m.i for t in m.t for c in m.c))
        return m.cost == expr
    m.obj_constr = Constraint(rule = rule_objective)

    
    # This function returns the model m to be used later within the code
    return m

# Assigning the function to a Block so that it loops through all the seasons
m.DES_res = Block(m.S, rule=seasonal_data)

# =============================================================================
# '''The OPF block'''
# =============================================================================

# A function to create the OPF block
def OPF_block(m,s):
    df_grid = pd.read_excel(parameters_file_name, sheet_name = "Grid")
    df_network = pd.read_excel(parameters_file_name, sheet_name = "Network")
    
    # Creating the opf_model object from the DES_OPF class
    m.opf_model = DES_OPF(house=house,  ft=ft, gen=gen, \
                                    df_grid=df_grid,\
                                    df_network=df_network,\
                                    slack=slack, nodes=nodes,\
                                    bus_connectivity=bus_connectivity)
    
    # Assigning the OPF method in opf_model to the Pyomo model m
    m = m.opf_model.OPF()
    
    # This function also returns the Pyomo model 
    return m

# =============================================================================
#                  '''Objective + Linking Constraints'''
# =============================================================================

# Creating a count and a dictionary to keep track of each season to aid the linking constraints
count = 0
m.map_season_to_count = dict()
m.first_season = None
for i in m.S:
    m.map_season_to_count[count] = i
    if count == 0:
        m.first_season = i
    count += 1

# Linking the capacities of all the DES technologies used within the model
def linking_PV_panels_residential_rule(m,season,house):
    previous_season = None
    if season == m.first_season:
        return Constraint.Skip
    else:
        for key, val in m.map_season_to_count.items():
            if val == season:
                previous_season = m.map_season_to_count[key-1]
                return m.DES_res[season].panels_PV[house] == m.DES_res[previous_season].panels_PV[house]
  
m.PV_linking_res = Constraint(m.S, house, rule = linking_PV_panels_residential_rule)

def boiler_capacities_residential_rule(m,season,house):
    previous_season = None
    if season == m.first_season:
        return Constraint.Skip
    else:
        for key, val in m.map_season_to_count.items():
            if val == season:
                previous_season = m.map_season_to_count[key-1]
                return m.DES_res[season].max_H_b[house] == m.DES_res[previous_season].max_H_b[house]

m.boiler_linking_res = Constraint(m.S, house, rule = boiler_capacities_residential_rule)

def battery_capacities_residential_rule(m,season,house,battery):
    previous_season = None
    if season == m.first_season:
        return Constraint.Skip
    else:
        for key, val in m.map_season_to_count.items():
            if val == season:
                previous_season = m.map_season_to_count[key-1]
                return m.DES_res[season].storage_cap[house,battery] == m.DES_res[previous_season].storage_cap[house,battery]

m.battery_linking_res = Constraint(m.S, house, battery, rule = battery_capacities_residential_rule)

def inverter_capacities_residential_rule(m,season,house):
    previous_season = None
    if season == m.first_season:
        return Constraint.Skip
    else:
        for key, val in m.map_season_to_count.items():
            if val == season:
                previous_season = m.map_season_to_count[key-1]
                return m.DES_res[season].inv_cap[house] == m.DES_res[previous_season].inv_cap[house]
m.inverter_linking_res = Constraint(m.S, house, rule = inverter_capacities_residential_rule)


#This is the objective function combining residential costs for all seasons
#m.obj = Objective(sense = minimize, expr=sum(b.cost for b in m.DES_res[:] for b in m.DES_com[:]))
m.obj = Objective(sense = minimize, expr=sum(b.cost for b in m.DES_res[:]))
#m.obj = Objective(sense = maximize, expr=sum(b.cost for b in m.DES_res[:]))

# =============================================================================
#                        '''SOLVE'''
# =============================================================================

# Deactivating some nonlinear constraints for the MILP intialisation
for s in m.S:
    m.DES_res[s].INV6.deactivate()
    
solver=SolverFactory('gams')
results = solver.solve(m, tee=True, 
                       solver = 'cplex', add_options=['option optcr=0.0001;'])
# m.pprint(filename='milp_check.txt')
# with open("MILP_withbatt_results.txt","w") as f:
#     f.write(str(results))
            

if KEEP_OPF == 1:
    
    # results = solver.solve(m, tee=True, solver = 'dicopt')
    
    # Declaring the OPF function as a block with respect to each season
    m.OPF_res = Block(m.S, rule=OPF_block)
    
    # Reactivating the nonlinear equations
    for s in m.S:
        m.DES_res[s].INV6.activate()
        # deactivating the current constraint if required
        # m.OPF_res[s].C12.deactivate()
        
    
    # Linking constraints to describe which house is connected to which node for active and reactive PFs
    def linking_blocks_P(m,season,house,time,node):
        for n,v in house_bus.items():
            if n == node and v == house:
                return (m.DES_res[season].E_PV_sold[house,time] - m.DES_res[season].E_grid[house,time] \
                    - sum(m.DES_res[season].E_grid_charge[house,time,battery] for battery in battery))/S_base == m.OPF_res[season].P[node,time]
            else:
                continue
        else:
            return Constraint.Skip
    m.Active_power_link = Constraint(m.S, house, m.timestamp, nodes, rule = linking_blocks_P)
    
    
    def linking_blocks_Q(m,season,house,time,node):
        for n,v in house_bus.items():
            if n == node and v == house:
                return m.DES_res[season].Q_gen[house,time]/S_base == m.OPF_res[season].Q[node,time] 
            else:
                continue
        else:
            return Constraint.Skip
    m.Reactive_power_link = Constraint(m.S, house, m.timestamp, nodes, rule = linking_blocks_Q)
    
    if RUN_NLP == 1:
        # Fixing binary variables and capacities for the NLP initialisation
        for s in m.S:
            for h in m.DES_res[s].i:
                # m.DES_res[s].inv_cap[h].fix()
                # m.DES_res[s].panels_PV[h].fix()
                # m.DES_res[s].max_H_b[h].fix()
                for t in m.DES_res[s].t:
                    m.DES_res[s].X[h,t].fix()
                    # Operation
                    # m.DES_res[s].E_PV_sold[h,t].fix()
                    for c in m.DES_res[s].c:
                        # m.DES_res[s].storage_cap[h,c].fix()
                        m.DES_res[s].Q[h,t,c].fix()
                        m.DES_res[s].W[h,c].fix()
                        # Operation
                        # m.DES_res[s].E_PV_charge[h,t,c].fix()
                    
    
        # solver = SolverFactory('ipopt')
        # options = {}
        # options['linear_solver'] = 'ma57'
        # results = solver.solve(m, options = options, tee=True,)
        solver = SolverFactory('gams')
        results = solver.solve(m, tee=True, solver = 'conopt')
        # with open("nlp_withbatt_results.txt","w") as f:
        #     f.write(str(results))
    
    
        # m.pprint(filename='NLPinit_check.txt')

    
    if RUN_NLP != 1 and RUN_MINLP == 1:
        solver=SolverFactory('gams')
        results = solver.solve(m, tee=True, solver = 'sbb', add_options=["GAMS_MODEL.nodlim = 5000;",'option optcr=0.001;'])
        # with open("minlp_results_withbatt.txt","w") as f:
        #     f.write(str(results))
        # results = solver.solve(m, tee=True, solver = 'dicopt')
        # solver = SolverFactory("octeract-engine")
        # results = solver.solve(m, tee = True, keepfiles=False)
        
# m.pprint(filename="NLP_check.txt")

# This returns the total run time (wall clock not CPU time)
stop = perf_counter()
ex_time = stop - start 

print("****Total time*****: ", ex_time )
print('')
print('')

# Some prints to sanity check the costs
annual_grid = sum(m.annual_cost_grid.value for m in m.DES_res[:])
print("annual_grid_cost = " + str(annual_grid))
day_grid = sum(m.cost_day.value for m in m.DES_res[:])
print("day_cost = " + str(day_grid))
night_grid = sum(m.cost_night.value for m in m.DES_res[:])
print("night_cost = " + str(night_grid))
carb_grid = sum(m.carbon_cost.value for m in m.DES_res[:])
print("grid_carbon_cost = " + str(carb_grid))
annual_PV_inv_cost = sum(m.annual_inv_PV.value for m in m.DES_res[:])
print("annual_PV_inv_cost = " + str(annual_PV_inv_cost))
annual_PV_op_cost = sum(m.annual_oc_PV.value for m in m.DES_res[:])
print("annual_PV_op_cost = " + str(annual_PV_op_cost))
annual_boiler_inv_cost = sum(m.annual_inv_B.value for m in m.DES_res[:])
print("annual_boiler_inv_cost = " + str(annual_boiler_inv_cost))
annual_boiler_op_cost = sum(m.annual_oc_b.value for m in m.DES_res[:])
print("annual_boiler_op_cost = " + str(annual_boiler_op_cost))
annual_batt_inv_cost = sum(m.annual_inv_S.value for m in m.DES_res[:])
print("annual_battery_inv_cost = " + str(annual_batt_inv_cost))
annual_batt_op_cost = sum(m.annual_oc_S.value for m in m.DES_res[:])
print("annual_battery_op_cost = " + str(annual_batt_op_cost))
annual_inc = sum(m.export_income.value for m in m.DES_res[:])
print("export_income = " + str(annual_inc))
annual_FIT = sum(m.gen_income.value for m in m.DES_res[:])
print("annual_FIT = " + str(annual_FIT))
print(' '*24)


grid_cost = {}
carb = {}
PV_inv = {}
PV_op = {}
B_I={}
B_O = {}
batt_inv={}
batt_op={}
einc ={}
ginc = {}

# for s in m.S:
#     season = str(s)
#     grid_cost[s] = m.DES_res[s].annual_cost_grid.value
#     print(season + "grid cost = "  + str(grid_cost[s]))
#     carb[s] = m.DES_res[s].carbon_cost.value
#     print(season + "carb cost = "  + str(carb[s]))
#     PV_inv[s] = m.DES_res[s].annual_inv_PV.value
#     print(season + "PV_inv = "  + str(PV_inv[s]))
#     PV_op[s] = m.DES_res[s].annual_oc_PV.value
#     print(season + "PV_op = "  + str(PV_op[s]))
#     B_I[s] = m.DES_res[s].annual_inv_B.value
#     print(season + "Boiler_inv = "  + str(B_I[s]))
#     B_O[s] = m.DES_res[s].annual_oc_b.value
#     print(season + "Boiler_op = "  + str(B_O[s]))
#     batt_inv[s] = m.DES_res[s].annual_inv_S.value
#     print(season + "Batt_inv = "  + str(batt_inv[s]))
#     batt_op[s] = m.DES_res[s].annual_oc_S.value
#     print(season + "Batt_op = "  + str(batt_op[s]))
#     einc[s] = m.DES_res[s].export_income.value
#     print(season + "Exp Income = "  + str(einc[s]))
#     ginc[s] = m.DES_res[s].gen_income.value
#     print(season + "Gen Income = "  + str(ginc[s]))
#     print('')
#     print('')
    

# =============================================================================
#     '''Converting results to pandas df and then exporting to Excel'''
# =============================================================================
    
for s in m.S:
    #for i,t in zip(house,m.timestamp):
    '''residential results'''
    E_grid_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_grid.items()}
    rdf_result1 = pd.DataFrame.from_dict(E_grid_res_data, orient="index", columns=["variable value"])
    
    panels_PV_res_data = {(i, v.name): value(v) for (i), v in m.DES_res[s].panels_PV.items()}
    rdf_result2 = pd.DataFrame.from_dict(panels_PV_res_data, orient="index", columns=["variable value"])
    
    E_PV_sold_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_PV_sold.items()}
    rdf_result3 = pd.DataFrame.from_dict(E_PV_sold_res_data, orient="index", columns=["variable value"])
    
    E_PV_used_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].E_PV_used.items()}
    rdf_result4 = pd.DataFrame.from_dict(E_PV_used_res_data, orient="index", columns=["variable value"])
    
    Max_H_b_res_data = {(i, v.name): value(v) for (i), v in m.DES_res[s].max_H_b.items()}
    rdf_result5 = pd.DataFrame.from_dict(Max_H_b_res_data, orient="index", columns=["variable value"])
    
    Storage_cap_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].storage_cap.items()}
    rdf_result8 = pd.DataFrame.from_dict(Storage_cap_res_data, orient="index", columns=["variable value"])
    
    Q_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].Q.items()}
    rdf_result9 = pd.DataFrame.from_dict(Q_res_data, orient="index", columns=["variable value"])
    
    Storage_volume_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].volume.items()}
    rdf_result14 = pd.DataFrame.from_dict(Storage_volume_res, orient="index", columns=["variable value"])
    
    X_res_data = {(i, t, v.name): value(v) for (i,t), v in m.DES_res[s].X.items()}
    rdf_result16 = pd.DataFrame.from_dict(X_res_data, orient="index", columns=["variable value"])
    
    type_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].W.items()}
    rdf_result17 = pd.DataFrame.from_dict(type_res_data, orient="index", columns=["variable value"])
    
    # New additions and OPF related exports
    Inv_cap_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].inv_cap.items()}
    rdf_result18 = pd.DataFrame.from_dict(Inv_cap_res, orient="index", columns=["variable value"])
    
    # PInv_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].P_inv.items()}
    # rdf_result19 = pd.DataFrame.from_dict(PInv_res_data, orient="index", columns=["variable value"])
    
    
    if KEEP_OPF == 1:
        Q_gen_res = {(i,t, v.name): value(v) for (i,t), v in m.DES_res[s].Q_gen.items()}
        rdf_result21 = pd.DataFrame.from_dict(Q_gen_res, orient="index", columns=["variable value"])
        
        V_res = {(n,t, v.name): value(v) for (n,t), v in m.OPF_res[s].V.items()}
        rdf_result22 = pd.DataFrame.from_dict(V_res, orient="index", columns=["variable value"])
        
        Theta_res = {(n,t, v.name): value(v) for (n,t), v in m.OPF_res[s].theta.items()}
        rdf_result23 = pd.DataFrame.from_dict(Theta_res, orient="index", columns=["variable value"])
        
        Q_res = {(n,t, v.name): value(v) for (n,t), v in m.OPF_res[s].Q.items()}
        rdf_result24 = pd.DataFrame.from_dict(Q_res, orient="index", columns=["variable value"])
        
        P_res = {(n,t, v.name): value(v) for (n,t), v in m.OPF_res[s].P.items()}
        rdf_result25 = pd.DataFrame.from_dict(P_res, orient="index", columns=["variable value"])
        
        I_res = {(n,m,t, v.name): value(v) for (n,m,t),v in m.OPF_res[s].current_sqr.items()}
        rdf_result26 = pd.DataFrame.from_dict(I_res, orient="index", columns=["variable value"])
        
    
    E_PVch_res = {}
    rdf_EPVch = {}
    E_stored_res = {}
    rdf_stored = {}
    E_gch_res = {}
    rdf_gridcharge = {}
    E_chg_res = {}
    rdf_chg = {}
    E_dsch_res = {}
    rdf_dsch = {}
    
    for bat_num in range(len(battery)):
        E_PVch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_PV_charge.items() if c == battery[bat_num]} 
        rdf_EPVch[bat_num] = pd.DataFrame.from_dict(E_PVch_res[bat_num], orient="index", columns=["variable value"])
        E_stored_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_stored.items() if c == battery[bat_num]}
        rdf_stored[bat_num] = pd.DataFrame.from_dict(E_stored_res[bat_num], orient="index", columns=["variable value"])
        E_gch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_grid_charge.items() if c == battery[bat_num]}
        rdf_gridcharge[bat_num] = pd.DataFrame.from_dict(E_gch_res[bat_num], orient="index", columns=["variable value"])
        E_chg_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_charge.items() if c == battery[bat_num]}
        rdf_chg[bat_num] = pd.DataFrame.from_dict(E_chg_res[bat_num], orient="index", columns=["variable value"])
        E_dsch_res[bat_num] = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_discharge.items() if c == battery[bat_num]}
        rdf_dsch[bat_num] = pd.DataFrame.from_dict(E_dsch_res[bat_num], orient="index", columns=["variable value"])
    
        
    Results_file_name1 = results_file_name+str(s)+results_file_suffix
    writer = ExcelWriter(Results_file_name1)
    rdf_result8.to_excel(writer, 'Res_storage_cap')
    rdf_result14.to_excel(writer, 'Res_stor_vol')
    rdf_result2.to_excel(writer, 'Res_Panels_PV')
    rdf_result5.to_excel(writer, 'Res_max_H_b')
    rdf_result1.to_excel(writer,'Res_E_grid')
    rdf_result3.to_excel(writer, 'Res_E_PV_sold')
    rdf_result4.to_excel(writer, 'Res_E_PV_used')
    
    for bat_num in range(len(battery)):
        for k, v in rdf_EPVch[bat_num].items():
            if sum(v) != 0:
                rdf_EPVch[bat_num].to_excel(writer, f"Res_E_PV_ch_{battery[bat_num]}")
        
        for k, v in rdf_stored[bat_num].items():
            if sum(v) != 0:
                rdf_stored[bat_num].to_excel(writer, f"Res_E_stored_{battery[bat_num]}")
        
        for k, v in rdf_gridcharge[bat_num].items():
            if sum(v) != 0:
                rdf_gridcharge[bat_num].to_excel(writer, f"Res_E_grd_ch_{battery[bat_num]}")
        
        for k, v in rdf_chg[bat_num].items():
            if sum(v) != 0:
                rdf_chg[bat_num].to_excel(writer, f"Res_E_charge_{battery[bat_num]}")
        
        for k, v in rdf_dsch[bat_num].items():
            if sum(v) != 0:
                rdf_dsch[bat_num].to_excel(writer, f"Res_E_disch_{battery[bat_num]}")
    
    
    if KEEP_OPF == 1:
        # rdf_result18.to_excel(writer, 'Inv_Cap')
        # rdf_result19.to_excel(writer, 'P_inv')
        rdf_result21.to_excel(writer, 'Q_gen')
        rdf_result22.to_excel(writer, 'V_OPF')
        rdf_result23.to_excel(writer, 'Angle_OPF')
        rdf_result25.to_excel(writer, 'P_node')
        rdf_result24.to_excel(writer, 'Q_node')
        rdf_result26.to_excel(writer, 'I_sqr_node')
        
    rdf_result9.to_excel(writer, 'Res_Q')
    rdf_result16.to_excel(writer, 'Res_X')
    

    
    writer.save()

# m.pprint(filename="full_model.txt")



