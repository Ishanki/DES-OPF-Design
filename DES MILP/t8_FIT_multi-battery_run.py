from pyomo.environ import *
import pandas as pd
import numpy as np
import xlrd
import sys
from pandas import ExcelWriter
from time import perf_counter
from t8_FIT_multibattery import Residential_Seasonal_DES
#from trial8_residences_carbon_noCRF import Residential_Seasonal_DES
#from Trial7_com_batteries import Commercial_Seasonal_DES

'''
This is the run file for the model that runs all 4 seasons for residential users with batteries.
Important Updates (07/09/2020):
    > Trial_8 residences updated
    > New objective to minimise cost of buying electricity added
    > Capital costs divided by 4 to adjust for the seasonal blocks
    > New link to Trial_8_residences_carbon that includes carbon costs (added to obj)
    > Irradiance dataset changed
'''
m = ConcreteModel()

# =============================================================================
#                              '''Inputs here'''
# =============================================================================
house = ['h1', 'h2', 'h3','h4','h5']
#['h1', 'h2', 'h3','h4','h5']
loads_file_name = "Loads_houses.xls" #"summer.xls" #"Loads_houses.xls"
building = ['b1', 'b2']
#commercial_loads_file_name = "Loads_buildings.xls"
parameters_file_name = "Parameters_cmbnd.xlsx" #NOTE: slightly higher PV efficiency 
battery = ['LI','SS'] # LI - lithium ion, SS - Sodium-Sulphur
interval = 0.5 #duration of time interval
m.timestamp = RangeSet(48) #state the total amount of timestamps here
m.S = RangeSet(4) #number of seasons, adjust the number accordingly. 
ft = 48 #state the final time step e.g. 24 or 48
d = {1:90, 2:92, 3:92, 4:91} #days in each season, 1-winter, 2-spring, 3-summer, 4-autumn
results_file_name = 'MILP_season_' #looks incomplete but it will attach the string of season later
results_file_suffix = '.xlsx'

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
            m.elec_house[h, t] = round(float((m.df[n][t]))/interval,3) #no need to specify float
            m.heat_house[h, t] = round(float((m.dfh[n][t]))/interval,3) #as division returns float
    
    #print(m.elec_house['h1', 34])
    #print(m.heat_house['h2',34]) 
    
    m.dfi = pd.read_excel(parameters_file_name, sheet_name = "Irradiance")
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
    
    m.Int1 = Residential_Seasonal_DES(house=house, df=m.df, days=m.days, \
                                      interval=interval, ft=ft, irrad=m.irrad,\
                                          df_scalars=df_scalars,\
                                              df_roof=df_roof, \
                                           elec_house=m.elec_house,
                                              heat_house=m.heat_house,\
                                                  df_batteries=df_batteries,\
                                                      df_volume=df_volume, \
                                                          battery=battery)
            
    m = m.Int1.DES_MILP()
    #model = b.model
    print(m)
    
    m.objective.deactivate()  #deactivating the individual objectives in each block
    
    #this is the free variable for total cost
    m.cost = Var(bounds = (None, None))
    
    #This is the objective function rule that combines the costs of all the seasons
    def rule_objective(m):
        expr = 0
        expr += (m.annual_cost_grid + m.carbon_cost + m.annual_inv_PV + m.annual_inv_B + m.annual_inv_S + m.annual_oc_PV + m.annual_oc_b + m.annual_oc_S - m.annual_income - m.FIT_gen)
        #expr += m.annual_cost_grid
        #expr += (sum(m.E_discharge[i,t,c] for i in m.i for t in m.t for c in m.c))
        return m.cost == expr
    m.obj_constr = Constraint(rule = rule_objective)
    
    #this return is for the whole block function.
    return m
    
m.DES_res = Block(m.S, rule=seasonal_data)


# =============================================================================
#                  '''Objective + Linking Constraints'''
# =============================================================================
    
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
solver=SolverFactory('gams')
results = solver.solve(m, tee=True, solver = 'cplex',  add_options=['option optcr=0.0001;'])
# =============================================================================
# sys.stdout = open('output_MILP2.txt','w')
# print(results)
# sys.exit()
# =============================================================================

stop = perf_counter()
ex_time = stop - start 
print("****Total time*****: ", ex_time )
print('')
print('')

m.compute_statistics(active=True)


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
annual_inc = sum(m.annual_income.value for m in m.DES_res[:])
print("annual_income = " + str(annual_inc))
annual_FIT = sum(m.FIT_gen.value for m in m.DES_res[:])
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
inc ={}
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
    inc[s] = m.DES_res[s].annual_income.value
    print(season + "exp_income = "  + str(inc[s]))
    ginc[s] = m.DES_res[s].FIT_gen.value
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
    
    # E_charge_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_charge.items()}
    # rdf_result6 = pd.DataFrame.from_dict(E_charge_res_data, orient="index", columns=["variable value"])
    
    # E_discharge_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_discharge.items()}
    # rdf_result7 = pd.DataFrame.from_dict(E_discharge_res_data, orient="index", columns=["variable value"])
    
    Storage_cap_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].storage_cap.items()}
    rdf_result8 = pd.DataFrame.from_dict(Storage_cap_res_data, orient="index", columns=["variable value"])
    
    Q1_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].Q1.items()}
    rdf_result9 = pd.DataFrame.from_dict(Q1_res_data, orient="index", columns=["variable value"])
    
    # E_PV_charge_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_PV_charge.items()}
    # rdf_result10 = pd.DataFrame.from_dict(E_PV_charge_res_data, orient="index", columns=["variable value"])
    
    # E_grid_charge_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_grid_charge.items()}
    # rdf_result11 = pd.DataFrame.from_dict(E_grid_charge_res_data, orient="index", columns=["variable value"])
    
    # E_stored_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_stored.items() if c == battery[0]}
    # rdf_result12 = pd.DataFrame.from_dict(E_stored_res_data, orient="index", columns=["variable value"])
    
    # E_stored2_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].E_stored.items() if c == battery[1]}
    # rdf_result15 = pd.DataFrame.from_dict(E_stored2_res_data, orient="index", columns=["variable value"])
      
    Q2_res_data = {(i, t, c, v.name): value(v) for (i,t,c), v in m.DES_res[s].Q2.items()}
    rdf_result13 = pd.DataFrame.from_dict(Q2_res_data, orient="index", columns=["variable value"])
    
    Storage_volume_res = {(i, v.name): value(v) for (i), v in m.DES_res[s].volume.items()}
    rdf_result14 = pd.DataFrame.from_dict(Storage_volume_res, orient="index", columns=["variable value"])
    
    X_res_data = {(i, t, v.name): value(v) for (i,t), v in m.DES_res[s].X.items()}
    rdf_result16 = pd.DataFrame.from_dict(X_res_data, orient="index", columns=["variable value"])
    
    type_res_data = {(i, c, v.name): value(v) for (i,c), v in m.DES_res[s].W.items()}
    rdf_result17 = pd.DataFrame.from_dict(type_res_data, orient="index", columns=["variable value"])
    
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
    
    for bat_num in range(0,len(battery)):
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
    # rdf_result10.to_excel(writer, 'Res_E_PV_charge')
    #rdf_result12.to_excel(writer, 'Res_E_stored')
    #rdf_result15.to_excel(writer, 'Res_E_stored2')
    # rdf_result6.to_excel(writer, 'Res_E_charge')
    # rdf_result11.to_excel(writer, 'Res_E_grid_charge')
    # rdf_result7.to_excel(writer, 'Res_E_discharge')
    
    for bat_num in range(0,len(battery)):
        rdf_EPVch[bat_num].to_excel(writer, f"Res_E_PV_ch_{battery[bat_num]}")
        rdf_stored[bat_num].to_excel(writer, f"Res_E_stored_{battery[bat_num]}")
        rdf_gridcharge[bat_num].to_excel(writer, f"Res_E_grd_ch_{battery[bat_num]}")
        rdf_chg[bat_num].to_excel(writer, f"Res_E_charge_{battery[bat_num]}")
        rdf_dsch[bat_num].to_excel(writer, f"Res_E_disch_{battery[bat_num]}")
        
    rdf_result9.to_excel(writer, 'Res_Q1')
    rdf_result13.to_excel(writer, 'Res_Q2')
    rdf_result16.to_excel(writer, 'Res_X')

    
    writer.save()
        



