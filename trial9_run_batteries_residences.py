from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from time import perf_counter
from pandas import ExcelWriter
from trial9_residences_MINLP import Residential_Seasonal_DES
from trial9_DES_OPF import DES_OPF

'''
This is the run file for the model that inputs data to the
DES (MINLP) and OPF (NLP), and links them.
Outputs include design capacities, operational schedule across all 4 seasons,
power flows, and voltage/angles. 
    
'''
m = ConcreteModel()

# =============================================================================
#                              '''Inputs here'''
# =============================================================================
house = ['h1','h2','h3','h4','h5']
loads_file_name = "Loads_houses.xls" #"summer.xls"
building = ['b1', 'b2']
#commercial_loads_file_name = "Loads_buildings.xls"
parameters_file_name = "Parameters_cmbnd.xlsx"
'''REMOVE THIS AFTER TROUBLESHOOTING'''
irrad_file_name = "Parameters_cmbnd.xlsx"
battery = ['c1'] # c1 - lithium ion
interval = 0.5 #duration of time interval
m.timestamp = RangeSet(48) #state the total amount of timestamps here
ft = 48 #state the final time step e.g. 24 or 48
m.S = RangeSet(4) #number of seasons, adjust the number accordingly. 
d = {1:90, 2:92, 3:92, 4:91} #days in each season, 1-winter, 2-spring, 3-summer, 4-autumn
results_file_name = 'MINLP_season_' #looks incomplete but it will attach the string of season later
results_file_suffix = '.xlsx'
df_grid = pd.read_excel(parameters_file_name, sheet_name = "Grid") #required for linking constraints
S_base = df_grid.iat[5,1]  

#distribution network
slack = 0
nodes = [0,1,2,3,4,5]
bus_connectivity = {1:(0,1),
                    2:(1,2),
                    3:(2,3),
                    4:(3,4),
                    5:(4,5)}

house_bus = {1:'h1',
             2: 'h2',
             3: 'h3',
             4: 'h4',
             5: 'h5'}

gen = [1,2,3,4,5] #generating nodes apart from slack

# =============================================================================
#                           '''Data Processing'''
# =============================================================================
#this is here to easily loop the data imported from excel
house_num = list(range(1,(len(house))+1))
#print(house)
building_num = list(range(1,(len(building))+1))

start = perf_counter()

# =============================================================================
# '''The residential block'''
# =============================================================================
def seasonal_data(m,s):

    m.df = {} #represents the dataframe for electricity
    m.dfh = {} #represents the dataframe for heating
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
        
    print("data is now loading into loops")
    m.elec_house = {}
    m.heat_house = {}
    for n, h in zip(house_num, house):
        for t in (m.df[n].index):
            m.elec_house[h, t] = float((m.df[n][t]))/interval
            m.heat_house[h, t] = float((m.dfh[n][t]))/interval
                
    #print(m.elec_house['h1', 34])
    #print(m.heat_house['h2',34]) 
    
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

    df_scalars = pd.read_excel(parameters_file_name, sheet_name = "Res_Scalars")
    #print(df_scalars)
    df_roof = pd.read_excel(parameters_file_name, sheet_name = "Roof_areas_res")
    df_batteries = pd.read_excel(parameters_file_name, sheet_name = "batteries")
    df_volume = pd.read_excel(parameters_file_name, sheet_name = "Stor_vol_res")
    df_grid = pd.read_excel(parameters_file_name, sheet_name = "Grid")
    df_network = pd.read_excel(parameters_file_name, sheet_name = "Network")
    
    m.Int1 = Residential_Seasonal_DES(house=house, df=m.df, days=m.days, \
                                    interval=interval, ft=ft, irrad=m.irrad, \
                                    df_scalars=df_scalars, df_roof=df_roof, \
                                    elec_house=m.elec_house, \
                                    heat_house=m.heat_house,\
                                    df_batteries=df_batteries,\
                                    df_volume=df_volume,\
                                    battery=battery, \
                                    )
            
    m = m.Int1.DES_MINLP()
    #model = b.model
    print(m)
    
    m.objective.deactivate()  #deactivating the individual objectives in each block
    
    #this is the free variable for total cost
    m.cost = Var(bounds = (None, None))
    #m.cost = Var(bounds = (None, 10000))
    
    #This is the objective function rule that combines the costs of all the seasons
    def rule_objective(m):
        expr = 0
        expr += (m.annual_cost_grid + m.carbon_cost + \
                 m.annual_inv_PV + m.annual_inv_B + m.annual_inv_S + \
                 m.annual_oc_PV + m.annual_oc_b + m.annual_oc_S \
                 - m.export_income - m.gen_income)
        #expr += m.annual_cost_grid
        #expr += (sum(m.E_discharge[i,t,c] for i in m.i for t in m.t for c in m.c))
        return m.cost == expr
    m.obj_constr = Constraint(rule = rule_objective)
    
    for i in house:
        print(m.max_H_b[i].value)
    
    #this return is for the whole block function.
    return m

m.DES_res = Block(m.S, rule=seasonal_data)

# =============================================================================
# '''The OPF block'''
# =============================================================================

def OPF_block(m,s):
    df_grid = pd.read_excel(parameters_file_name, sheet_name = "Grid")
    df_network = pd.read_excel(parameters_file_name, sheet_name = "Network")
    
    m.OPF1 = DES_OPF(house=house,  ft=ft, gen=gen, \
                                    df_grid=df_grid,\
                                    df_network=df_network,\
                                    slack=slack, nodes=nodes,\
                                    bus_connectivity=bus_connectivity)
    
    m = m.OPF1.OPF()
    
    return m

# =============================================================================
#                  '''Objective + Linking Constraints'''
# =============================================================================

m.slack = Var(nodes, m.timestamp, bounds = (None,None), initialize = 0)


count = 0
m.map_season_to_count = dict()
m.first_season = None
for i in m.S:
    m.map_season_to_count[count] = i
    if count == 0:
        m.first_season = i
    count += 1

#Linking boiler capacities model.max_H_b[i] and number of panels model.panels_PV[i]
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


#This is the objective function combining residential and commercial costs
#m.obj = Objective(sense = minimize, expr=sum(b.cost for b in m.DES_res[:] for b in m.DES_com[:]))
m.obj = Objective(sense = minimize, expr=sum(b.cost for b in m.DES_res[:]))
#m.obj = Objective(sense = maximize, expr=sum(b.cost for b in m.DES_res[:]))

# =============================================================================
#                        '''SOLVE'''
# =============================================================================
# =============================================================================
# #m.DES_res[1].Q1.fix(1)
# #m.DES_res[1].Q2.fix(0)
# #m.DES_res[1].X.fix(0)
# m.DES_res[1].panels_PV.fix(16)
# solver=SolverFactory('ipopt')
# options ={}
# options['linear_solver'] = 'ma57'
# results = solver.solve(m, options = options, tee=True)
# 
# =============================================================================
#m.slack.pprint()

solver=SolverFactory('gams')
# #options = {}
results = solver.solve(m, tee=True,  add_options=["GAMS_MODEL.optfile = 1;",'option optcr=0.01;'], solver = 'dicopt')
#results = solver.solve(m, tee=True, solver = 'sbb', add_options=["GAMS_MODEL.nodlim = 5000;",'option optcr=0.01;'])
#results = solver.solve(m, tee=True, solver = 'couenne')
#results = solver.solve(m, tee=True, solver = 'sbb', add_options=["GAMS_MODEL.nodlim = 5000;",'option optcr=0.1;'])
m.OPF_res = Block(m.S, rule=OPF_block)

def linking_blocks_P(m,season,house,time,node,battery):
    for n,v in house_bus.items():
        if n == node and v == house:
            return (m.DES_res[season].E_PV_sold[house,time] - m.DES_res[season].E_grid[house,time] \
                - m.DES_res[season].E_grid_charge[house,time,battery])/S_base == m.OPF_res[season].P[node,time] #+ m.slack[node,time]
        else:
            continue
    else:
        return Constraint.Skip
m.Active_power_link = Constraint(m.S, house, m.timestamp, nodes, battery, rule = linking_blocks_P)


def linking_blocks_Q(m,season,house,time,node):
    for n,v in house_bus.items():
        if n == node and v == house:
            return m.DES_res[season].Q_gen[house,time]/S_base == m.OPF_res[season].Q[node,time] #+ m.slack[node,time]
        else:
            continue
    else:
        return Constraint.Skip
m.Reactive_power_link = Constraint(m.S, house, m.timestamp, nodes, rule = linking_blocks_Q)


# solver = SolverFactory("octeract-engine")
# results = solver.solve(m, tee = True, keepfiles=False)
results = solver.solve(m, tee=True, solver = 'sbb', add_options=["GAMS_MODEL.nodlim = 5000;",'option optcr=0.1;'])
#results = solver.solve(m, tee=True, solver = 'dicopt')
# sys.stdout = open('output.txt','w')
# print(results)
# sys.exit()
#results = solver.solve(m, tee=True, solver = 'sbb', add_options=["GAMS_MODEL.nodlim = 5000;",'option optcr=0.01;'])


stop = perf_counter()
ex_time = stop - start 

print("****Total time*****: ", ex_time )
print('')
print('')

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
exp_inc = sum(m.export_income.value for m in m.DES_res[:])
print("exp_income = " + str(exp_inc))
gen_inc = sum(m.gen_income.value for m in m.DES_res[:])
print("gen_income = " + str(gen_inc))
print('*'*24)
print('')
print('')


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
for s in m.S:
    season = str(s)
    grid_cost[s] = m.DES_res[s].annual_cost_grid.value
    print(season + "grid cost = "  + str(grid_cost[s]))
    carb[s] = m.DES_res[s].carbon_cost.value
    print(season + "carb cost = "  + str(carb[s]))
    PV_inv[s] = m.DES_res[s].annual_inv_PV.value
    print(season + "PV_inv = "  + str(PV_inv[s]))
    PV_op[s] = m.DES_res[s].annual_oc_PV.value
    print(season + "PV_op = "  + str(PV_op[s]))
    B_I[s] = m.DES_res[s].annual_inv_B.value
    print(season + "Boiler_inv = "  + str(B_I[s]))
    B_O[s] = m.DES_res[s].annual_oc_b.value
    print(season + "Boiler_op = "  + str(B_O[s]))
    batt_inv[s] = m.DES_res[s].annual_inv_S.value
    print(season + "Batt_inv = "  + str(batt_inv[s]))
    batt_op[s] = m.DES_res[s].annual_oc_S.value
    print(season + "Batt_op = "  + str(batt_op[s]))
    einc[s] = m.DES_res[s].export_income.value
    print(season + "Exp Income = "  + str(einc[s]))
    ginc[s] = m.DES_res[s].gen_income.value
    print(season + "Gen Income = "  + str(ginc[s]))
    print('')
    print('')
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
    
    E_charge_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_charge.items()}
    rdf_result6 = pd.DataFrame.from_dict(E_charge_res_data, orient="index", columns=["variable value"])
    
    E_discharge_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_discharge.items()}
    rdf_result7 = pd.DataFrame.from_dict(E_discharge_res_data, orient="index", columns=["variable value"])
    
    Storage_cap_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].storage_cap.items()}
    rdf_result8 = pd.DataFrame.from_dict(Storage_cap_res_data, orient="index", columns=["variable value"])
    
    Q1_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].Q1.items()}
    rdf_result9 = pd.DataFrame.from_dict(Q1_res_data, orient="index", columns=["variable value"])
    
    E_PV_charge_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_PV_charge.items()}
    rdf_result10 = pd.DataFrame.from_dict(E_PV_charge_res_data, orient="index", columns=["variable value"])
    
    E_grid_charge_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_grid_charge.items()}
    rdf_result11 = pd.DataFrame.from_dict(E_grid_charge_res_data, orient="index", columns=["variable value"])
    
    E_stored_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_stored.items()}
    rdf_result12 = pd.DataFrame.from_dict(E_stored_res_data, orient="index", columns=["variable value"])
      
    Q2_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].Q2.items()}
    rdf_result13 = pd.DataFrame.from_dict(Q2_res_data, orient="index", columns=["variable value"])
    
    Storage_volume_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].volume.items()}
    rdf_result14 = pd.DataFrame.from_dict(Storage_volume_res, orient="index", columns=["variable value"])
    
    Inv_cap_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].inv_cap.items()}
    rdf_result15 = pd.DataFrame.from_dict(Inv_cap_res, orient="index", columns=["variable value"])
    
    X_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].X.items()}
    rdf_result22 = pd.DataFrame.from_dict(X_res_data, orient="index", columns=["variable value"])
    
    PInv_res_data = {(i, t, v.name): value(v) for (i, t), v in m.DES_res[s].P_inv.items()}
    rdf_result23 = pd.DataFrame.from_dict(PInv_res_data, orient="index", columns=["variable value"])
    
    slack_res = {(n,t, v.name): value(v) for (n,t), v in m.slack.items()}
    rdf_result24 = pd.DataFrame.from_dict(slack_res, orient="index", columns=["variable value"])
    
    Q_gen_res = {(i,t, v.name): value(v) for (i,t), v in m.DES_res[s].Q_gen.items()}
    rdf_result16 = pd.DataFrame.from_dict(Q_gen_res, orient="index", columns=["variable value"])
    
    V_res = {(n,t, v.name): value(v) for (n,t), v in m.OPF_res[s].V.items()}
    rdf_result17 = pd.DataFrame.from_dict(V_res, orient="index", columns=["variable value"])
    
    Theta_res = {(n,t, v.name): value(v) for (n,t), v in m.OPF_res[s].theta.items()}
    rdf_result18 = pd.DataFrame.from_dict(Theta_res, orient="index", columns=["variable value"])
    
    Q_res = {(n,t, v.name): value(v) for (n,t), v in m.OPF_res[s].Q.items()}
    rdf_result19 = pd.DataFrame.from_dict(Q_res, orient="index", columns=["variable value"])
    
    P_res = {(n,t, v.name): value(v) for (n,t), v in m.OPF_res[s].P.items()}
    rdf_result20 = pd.DataFrame.from_dict(P_res, orient="index", columns=["variable value"])
# =============================================================================
#     P_net_res = {(i,t, v.name): value(v) for (i,t), v in m.DES_res[s].Net_P.items()}
#     rdf_result21 = pd.DataFrame.from_dict(P_net_res, orient="index", columns=["variable value"])
# =============================================================================

    
    Results_file_name1 = results_file_name+str(s)+results_file_suffix
    writer = ExcelWriter(Results_file_name1)
    rdf_result23.to_excel(writer, 'P_inv')
    rdf_result16.to_excel(writer, 'Q_gen')
    rdf_result17.to_excel(writer, 'V_OPF')
    rdf_result18.to_excel(writer, 'Angle_OPF')
    rdf_result19.to_excel(writer, 'Q_node')
    rdf_result24.to_excel(writer, 'slack')
    rdf_result20.to_excel(writer, 'P_node')
    
    rdf_result1.to_excel(writer,'Res_E_grid')
    rdf_result22.to_excel(writer, 'X')
    rdf_result2.to_excel(writer, 'Res_Panels_PV')
    rdf_result3.to_excel(writer, 'Res_E_PV_sold')
    rdf_result4.to_excel(writer, 'Res_E_PV_used')
    rdf_result10.to_excel(writer, 'Res_E_PV_charge')
    rdf_result5.to_excel(writer, 'Res_max_H_b')
    rdf_result6.to_excel(writer, 'Res_E_charge')
    rdf_result7.to_excel(writer, 'Res_E_discharge')
    rdf_result8.to_excel(writer, 'Res_storage_cap')
    rdf_result9.to_excel(writer, 'Res_Q1')
    
    rdf_result11.to_excel(writer, 'Res_E_grid_charge')
    rdf_result12.to_excel(writer, 'Res_E_stored')
    rdf_result13.to_excel(writer, 'Res_Q2')
    rdf_result14.to_excel(writer, 'Res_stor_vol')
    rdf_result15.to_excel(writer, 'Inv_Cap')
    
    #rdf_result21.to_excel(writer, 'P_net')

    
    writer.save()


