import math
import random as rd
import numpy as np
import copy
import scipy.stats
import mpmath
import pandas as pd

tolerance = 0.000000000000001
SteadyState = 0


def GetMyopicy(CriticalFractile, mean, stddev, truncated_LB, truncated_UB,):
    y = scipy.stats.truncnorm.ppf(CriticalFractile,truncated_LB, truncated_UB, mean, stddev)
    return y

def GetMyopicyUniform(CriticalFractile, mean, stddev, truncated_LB, truncated_UB,):
    y = scipy.stats.uniform.ppf(CriticalFractile,truncated_LB, truncated_UB)
    return y

def Regeneration_Process(q_list, used_capacity, capacity):


    q_list_new = [used_capacity]
    for i in range(0, len(q_list) - 1):
        q_list_new.append(q_list[i])


    regenerated = copy.deepcopy(q_list[len(q_list)-1])
    if regenerated[1] > 0:
        capacity[0] = capacity[0] + copy.deepcopy(capacity[2]) + copy.deepcopy(regenerated[0])
        capacity[1] = copy.deepcopy(regenerated[1])
        capacity[2] = copy.deepcopy(regenerated[2])
    else: #capacity[1] > 0:
        capacity[2] = capacity[2] + regenerated[0]+regenerated[2]
    # print(q_list, used_capacity, capacity, q_list_new)
    return q_list_new, capacity

def Capacity_Order_Satisfaction_Process(capacity, order, k):

    initial_track = copy.deepcopy(capacity[1])
    used_capacity = [0.0, 0.0, 0.0]
    # print(capacity, order)
    for i in range(0, len(capacity)):
        capacity_allocated = min(copy.deepcopy(capacity[i]), order)
        order = order - capacity_allocated
        capacity[i] = capacity[i] - capacity_allocated
        used_capacity[i] = capacity_allocated

    if capacity[0] < 1 and capacity[1] < 1 and capacity[2] > 0:
        capacity[0] = copy.deepcopy(capacity[2])
        capacity[2] = 0.0



    if initial_track == 1 and used_capacity[1] == 1:
        utilized = 1
    else:
        utilized = 0

    if k > 0:

        return capacity, used_capacity, utilized
    else:
        # print("BEF", capacity, used_capacity, utilized)
        if capacity[1] > 0:
            capacity[2] = copy.deepcopy(capacity[2]) + copy.deepcopy(used_capacity[0]) + copy.deepcopy(used_capacity[2])
        else:
            capacity[0] = copy.deepcopy(capacity[0]) + copy.deepcopy(capacity[2]) + copy.deepcopy(used_capacity[0])
            capacity[1] = copy.deepcopy(used_capacity[1])
            capacity[2] = copy.deepcopy(used_capacity[2])
        # print("AFT", capacity, used_capacity, utilized)
        return capacity, utilized






def Simulation_Utilization(inventory, y_M, y_L,  capacity, regeneration_schedule, average_regeneration, average_inventory, average_lost_sales, mean, stddev, truncated_LB, truncated_UB, k, t,  average_order, average_capacity, total_utilized, c, h, p, total_cost):


    # period_demand = scipy.stats.truncnorm.rvs(truncated_LB, truncated_UB, loc=mean, scale=stddev, size=1, random_state=None)
    period_demand = rd.randint(truncated_LB, truncated_UB)


    period_order_up_to = y_L



    order = min(sum(capacity), max(period_order_up_to - inventory, 0.0))




    if k > 0:
        capacity, used_capacity, utilized = Capacity_Order_Satisfaction_Process(capacity, order, k)
    else:
        capacity, utilized = Capacity_Order_Satisfaction_Process(capacity, order, k)
    total_utilized = total_utilized + utilized

    lost_sales = max(period_demand - inventory - order, 0)


    inventory = inventory + order

    inventory = max(inventory - period_demand,0)

    total_cost = total_cost + float(p*lost_sales) + float(h*inventory) + float(c*order)


    if k > 0:
        regeneration_schedule, capacity = Regeneration_Process(regeneration_schedule, used_capacity, capacity)
    else:
        pass

    # print(regeneration_schedule, capacity)



    return inventory, y_M, y_L,  capacity, regeneration_schedule, average_regeneration, average_inventory, average_lost_sales, mean, stddev, truncated_LB, truncated_UB, t+1, average_order, average_capacity, total_utilized, c, h, p, total_cost



def Simulate(y_M, y_L, capacity, mean, stddev, truncated_LB, truncated_UB, k, T,c,h,p):

    t = 1
    inventory = 0.0
    average_inventory = 0.0
    average_lost_sales = 0.0
    average_order = 0.0
    average_capacity = 0.0
    average_regeneration = 0.0
    regeneration_schedule = []
    total_utilized = 0.0
    initial_capacity = copy.deepcopy(capacity)
    capacity = copy.deepcopy(capacity)
    total_cost = 0.0


    if k > 0:
        for i in range(1, k+1):
            regeneration_schedule.append([0.0, 0.0, 0.0])
    else:
        pass

    while t <= T:
        inventory, y_M, y_L,  capacity, regeneration_schedule, average_regeneration, average_inventory, average_lost_sales, mean, stddev, truncated_LB, truncated_UB, t, average_order, average_capacity, total_utilized,c, h, p, total_cost = Simulation_Utilization(inventory, y_M, y_L,  capacity, regeneration_schedule, average_regeneration, average_inventory, average_lost_sales, mean, stddev, truncated_LB, truncated_UB, k, t, average_order, average_capacity, total_utilized,c, h, p, total_cost)

    average_utilization = round(float(total_utilized/T), 3)
    average_cost = round(float(total_cost / T), 3)

    # average_regeneration = [round(float(x/(T-SteadyState)),rounding) for x in average_regeneration]
    # average_inventory = round(float(average_inventory/(T-SteadyState)),rounding)
    # average_lost_sales = round(float(average_lost_sales/(T-SteadyState)),rounding)
    # average_order = round(float(average_order/(T-SteadyState)), rounding)
    # average_available_capacity = round(float(average_capacity/(T-SteadyState)), rounding)




    #Display (uncomment if you want to see)
    print("Cap = ", initial_capacity,  "Ave Order = ", average_order, "y_L = ", y_L, "Ave Lost Sales = ", round(average_lost_sales,rounding), "k = ", k, "Utilization = ",  average_utilization, "Average Cost = ", average_cost)
    return total_utilized, total_cost


h = 0.5
c = 0.9
p = 1.0
alpha = 0.0
truncated_LB = 0
truncated_UB = 40
mean = 10
stddev = mean/5.0
k = 12
T = 100000
rounding = 0

CriticalFractile_y_M = min(1,(1+alpha)*(p-c)/(h+p-alpha*c))
CriticalFractile_y_L = (p-c)/(h+p-alpha*c)



y_M = round(GetMyopicyUniform(CriticalFractile_y_M, mean, stddev, truncated_LB, truncated_UB),rounding)
y_L = round(GetMyopicyUniform(CriticalFractile_y_L, mean, stddev, truncated_LB, truncated_UB),rounding)


test = scipy.stats.truncnorm.cdf(y_L-math.sqrt(2)*stddev, truncated_LB, truncated_UB, mean, stddev)




print(y_L, test, GetMyopicy(test, mean, stddev, truncated_LB, truncated_UB))
Headers = ["k", "Utilization", "Cost", "y_L",  "Capacity","c","h","p"]
Overall_Consolidated_output = []
capacity_list = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
# capacity_list = [5]
total = len(capacity_list)*k
counter = 0
for i in range(0, len(capacity_list)):


    capacity = [0.0, 1.0, round(y_L*capacity_list[i],rounding)-1]

    results = []

    for j in range(0, k):
        counter += 1
        print(total, float(counter / total) * 100)
        # if capacity >= (j + 1) * y_L:
        #     multiple = 1
        # else:
        #     multiple = 0
        total_utilization, total_cost = Simulate(y_M, y_L, capacity, mean, stddev, truncated_LB, truncated_UB, j, T, c, h, p)
        utilization = float(total_utilization/T)
        cost = float(total_cost/T)
        Consolidated_Output = copy.deepcopy([])
        Consolidated_Output = [j, utilization, cost, y_L,capacity, c, h, p]
        Overall_Consolidated_output.append(Consolidated_Output)


ResultsDataFrame = pd.DataFrame(Overall_Consolidated_output, columns=Headers)

ResultsDataFrame.to_csv("Tracked_Utilization_Python_Output_c=0.9.csv", encoding='utf-8', index=False)

