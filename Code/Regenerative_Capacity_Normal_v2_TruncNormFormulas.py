
import time
import copy
import pandas as pd
import scipy.stats
import scipy.integrate
import scipy.optimize
from functools import lru_cache
import scipy.misc
import mpmath
import math

from Regenerative_Capacity_Myopic_Normal import Myopic_Start



global c, h, p, u_vector, T, D_realizations, D_probabilities, y_L, Total_u, alpha, mean, stddev, a, b


y_dict = dict()

def GetOptimaly(CriticalFractile):
    y = scipy.stats.truncnorm.ppf(CriticalFractile,a, b, mean, stddev)  # truncated normal ppf(q, a, b, loc=0, scale=1) truncated on [a, b], loc = mean, scale = std dev
    return y


def standardnormalpdf(x):
    return (1.0/math.sqrt(2*math.pi))*math.exp(-0.5*(x**2))

def standardnormalcdf(x):
    return 0.5*(1+math.erf(x/math.sqrt(2)))

def TruncatedNormalPdf(x):
    first = standardnormalcdf((b-mean)/stddev)
    second = standardnormalcdf((a-mean)/stddev)

    if first - second == 0:
        pdf = 0.0
    else:
        pdf  = (1.0/stddev)*(standardnormalpdf((x-mean)/stddev)/(first - second))
    return pdf

@lru_cache(maxsize=None)
def GetCurrentPeriodCost_lru_cache(y):
    # Current Period Cost Computation
    y = float(y)
    f = lambda D: (y-D)*TruncatedNormalPdf(D)

    CurrentPeriodCost = mpmath.quad(f, [a, y])
    CurrentPeriodCost = (h + p - alpha*c) * CurrentPeriodCost + (c-p) * y + p * mean
    return CurrentPeriodCost


@lru_cache(maxsize=None)
def x_vector_StateTransition_lru_cache(x_vector_input, y, D):
    x_vector_transition = list(x_vector_input)
    x_vector_transition.pop()
    x_vector_transition.insert(0, y + Total_u)
    x_vector_transition = [x - int(min(D, y)) for x in x_vector_transition]
    return tuple(x_vector_transition)

#monotone and bounded sensitivity
def y_cache(q, test,s):
    if s != 0:
        return 10000
    else:
        checker = 1
        for i in range(0, len(q)):
            if abs(q[i] - test[0][i]) > 0 and checker == 1:
                checker = 0
            elif abs(q[i] - test[0][i]) > 0 and checker == 0:
                return 10000
        return 0

@lru_cache(maxsize = None)
def Future_Cost_Computation(x_vector, y, t, k):
    Transition = lambda D: x_vector_StateTransition_lru_cache(tuple(x_vector), y, D)
    f = lambda D: V_lru_cache(Transition(D), t + 1, k)*TruncatedNormalPdf(D)

    cost = GetCurrentPeriodCost_lru_cache(y)
    # Future_cost = IntegralFunction_lru_cache(tuple(copy.copy(x_vector)), y, t+1, k)

    if t == T:
        Future_cost = 0
    else:
        # print(t)
        Future_cost = mpmath.quad(f, [a, b])


    cost = cost + alpha * Future_cost
    return cost



@lru_cache(maxsize=None)
def V_lru_cache(x_vector, t, k):
    if t == T:
        Future_u = 0
    else:
        Future_u = sum(u_vector[t:len(u_vector) - 1])

    LB_Usage = int(x_vector[0]) - int(Total_u)
    UB_Usage = max(int(x_vector[0]) - int(Total_u), int(x_vector[k]) - int(Future_u))

    current_cost = 10000000000.0
    current_y = 1000000.0


    #obtain original states
    q_vector_first_step = [i - Total_u for i in x_vector]
    q_vector_and_x = []
    q_vector_and_x.append(q_vector_first_step[0])
    for i in range(1, len(q_vector_first_step)):
        q_vector_and_x.append(q_vector_first_step[i-1] - q_vector_first_step[i])


    try:
        y = y_dict[(tuple(q_vector_and_x), t)]
    except KeyError:
        reference_dict = {key: value for key, value in y_dict.items() if y_cache(q_vector_and_x, key, key[1] - t) == 0}

        try:
            keys_list = list(reference_dict.keys())
            min_difference_list = [abs(sum(q_vector_and_x) - sum(list(keys_list[i][0]))) for i in range(0, len(keys_list)-1)]
            min_difference_dict = dict(zip(keys_list, min_difference_list))
            try:
                min_key = min(min_difference_dict.items(), key=lambda x: x[1])
                min_key = list(min_key)
                min_key[1] = sum(q_vector_and_x) - sum(list(min_key[0][0]))
                min_key = tuple(min_key)
                # max_key = max(min_difference_dict.items(), key=lambda x: x[1])
                y_min = reference_dict[min_key[0]]
                # y_max = reference_dict[max_key[0]]
                if q_vector_and_x[0] != min_key[0][0][0]:
                    x_or_q_min = "x"
                else:
                    x_or_q_min = "q"

            except ValueError:
                pass

        except IndexError:
            pass

    try:
        cost = Future_Cost_Computation(x_vector, y, t, k)
        if t == 1:
            return cost, y
        else:
            return cost
    except UnboundLocalError:
        try:
            if x_or_q_min == "x":
               if min_key[1] > 0:
                    y_Bound = y_min + min_key[1]
                    if y_min <= LB_Usage:
                        pass
                    elif y_min > LB_Usage and y_min <= UB_Usage:
                        LB_Usage = y_min
                    else:
                        LB_Usage = UB_Usage

                    if y_Bound <= LB_Usage:
                        UB_Usage = LB_Usage
                    elif y_Bound > LB_Usage and y_Bound <= UB_Usage:
                        UB_Usage  = y_Bound
                    else:
                        pass
               else:
                    y_Bound = y_min - min_key[1]
                    if y_Bound <= LB_Usage:
                        pass
                    elif y_Bound > LB_Usage and y_Bound <= UB_Usage:
                        LB_Usage = y_Bound
                    else:
                        LB_Usage = UB_Usage

                    if y_min <= LB_Usage:
                        UB_Usage = LB_Usage
                    elif y_min > LB_Usage and y_min <= UB_Usage:
                        UB_Usage = y_min
                    else:
                        pass
            else:
                if min_key[1] > 0:
                    y_Bound = y_min - min_key[1]
                    if y_Bound <= LB_Usage:
                        pass
                    elif y_Bound > LB_Usage and y_Bound <= UB_Usage:
                        LB_Usage = y_Bound
                    else:
                        LB_Usage = UB_Usage

                    if y_min <= LB_Usage:
                        UB_Usage = LB_Usage
                    elif y_min > LB_Usage and y_min <= UB_Usage:
                        UB_Usage = y_min
                    else:
                        pass
                else:
                    y_Bound = y_min + min_key[1]
                    if y_min <= LB_Usage:
                        pass
                    elif y_min > LB_Usage and y_min <= UB_Usage:
                        LB_Usage = y_min
                    else:
                        LB_Usage = UB_Usage

                    if y_Bound <= LB_Usage:
                        UB_Usage = LB_Usage
                    elif y_Bound > LB_Usage and y_Bound <= UB_Usage:
                        UB_Usage = y_Bound
                    else:
                        pass
        except UnboundLocalError:
            pass


        if t != T:
            for y in range(int(LB_Usage), int(UB_Usage+1)):
                cost = Future_Cost_Computation(x_vector, y, t, k)
                new_cost = cost
                new_y = y
                if new_cost < current_cost:
                    current_cost = new_cost
                    current_y = new_y
                if y == UB_Usage or new_cost > current_cost:

                    y_dict.update({(tuple(q_vector_and_x), t): current_y})

                    if t == 1:
                        return current_cost, current_y
                    else:
                        return current_cost
        else:
            y = float(max(LB_Usage, min(UB_Usage, y_L)))
            # print("lol", y)
            cost = GetCurrentPeriodCost_lru_cache(y)
            y_dict.update({(tuple(q_vector_and_x), t): y})
            if T == 1:
                return cost, y
            else:
                return cost



start_time = time.time()
Rolling = start_time




Overall_Consolidated_Results_cost = []
Overall_Consolidated_Results_y = []
Headers_cost = []
Headers_y = []


tolerance = 0.1
round_number = 0
u_list = [0.8, 1.5, 2.0] #fractions of y_L | 0.7
x_list = [0.0]
# for i in range(0, 16):
#     x_list.append(i)
#     # u_list.append(i)
k_list = [0,1,2,3,4]
c_list = [3.0]
h_list = [1.0]
p_list = [10.0]
alpha_list = [1.0]
T_list = [5]
# u_list = [0.8]
u_list_mid = [0.0]
a = 0.0
b = 10.0
mean_list = [5]

TotalRuns  = len(k_list)*len(c_list)*len(h_list)*len(p_list)*len(alpha_list)*len(T_list)*len(u_list)*len(u_list_mid)*len(mean_list)*len(x_list)
print (TotalRuns)


Headers_cost.append("c")
Headers_cost.append("h")
Headers_cost.append("p")
Headers_cost.append("alpha")
Headers_cost.append("T")
Headers_cost.append("u Fraction")
Headers_cost.append("u")
Headers_cost.append("x start")
Headers_cost.append("x vector")
Headers_cost.append("mean demand")
Headers_cost.append("k")
Headers_cost.append("Cost")
# Headers_cost.append("Myopic Cost")
Headers_cost.append("y")
Headers_cost.append("y_L")
# Headers_cost.append("Percent Difference in Cost")
Headers_cost.append("Time To Calculate - OPT")
# Headers_cost.append("Time To Calculate - Myopic")


counter = 0
for T_i in T_list:
    T = T_i
    for c_i in c_list:
        c = c_i
        for h_i in h_list:
            h = h_i
            for p_i in p_list:
                p = p_i
                for alpha_i in alpha_list:
                    alpha = alpha_i
                    # for T_i in T_list:
                    #     T = T_i

                    for mean_i in mean_list:
                        mean = mean_i
                        stddev = int(mean/3.0)

                        # for u_i in u_list:
                        for u_i in range(0, 21):
                            y_L = round(GetOptimaly((p - c) / (h + p - alpha * c)),round_number)

                            # u = int(u_i * y_L)
                            u = u_i

                            for u_mid_i in u_list_mid:
                                u_mid = int(u_mid_i*y_L)

                                for k in k_list:





                                    for x in x_list:

                                        u_vector = []
                                        for i in range(1, T +1):
                                            u_vector.append(0.0)
                                        u_vector[0] = u
                                        # print(u_vector)

                                        u_vector[int(T/2)] = u_mid
                                        Total_u = sum(u_vector)


                                        # for j in range(0, u+1): #we put in this loop to test for the effect of changing the regeneration schedule
                                        x_vector_start = []

                                            # counter += 1
                                            # y_dict.clear()
                                            # Consolidated_Results_cost = []
                                            # Consolidated_Results_y = []
                                            # x_vector_StateTransition_lru_cache.cache_clear()
                                            # GetCurrentPeriodCost_lru_cache.cache_clear()
                                            # V_lru_cache.cache_clear()

                                        for i in range(1, k+2):
                                            x_vector_start.append(Total_u+x)


                                        myopic_x_vector_start = copy.deepcopy(x_vector_start)


                                        # x_vector_start = copy.copy(x_vector_copy)
                                        counter += 1
                                        y_dict.clear()
                                        Consolidated_Results_cost = []
                                        Consolidated_Results_y = []
                                        x_vector_StateTransition_lru_cache.cache_clear()
                                        GetCurrentPeriodCost_lru_cache.cache_clear()
                                        V_lru_cache.cache_clear()
                                        # GoldenSectionSearch.cache_clear()
                                        Future_Cost_Computation.cache_clear()
                                        #
                                        # y_1 = int(GetOptimaly(float(((1 - alpha) * (p - c)) / ((1 - alpha) * (h + p - alpha * c) + (alpha ** 2 * (p - c))))))
                                        # y_2 = int(GetOptimaly(float((1 - alpha) * (p - c) / ((1 - alpha) * (h + p - alpha * c) + alpha * p - alpha * c))))
                                        # y_3 = int(GetOptimaly(min(float((1+alpha)*(p-c)/(h+p-alpha*c)),1.0)))
                                        # y_L = int(GetOptimaly((p-c)/(h+p-alpha*c)))

                                        # print(y_1, y_2, y_3, y_L)


                                        cost, y = V_lru_cache(tuple(copy.copy(x_vector_start)), 1, k)
                                        print(cost, y)

                                        x_vector_StateTransition_lru_cache.cache_clear()
                                        GetCurrentPeriodCost_lru_cache.cache_clear()
                                        V_lru_cache.cache_clear()



                                        CalculationTime_Optimal = time.time()-Rolling
                                        Rolling = time.time()
                                        # cost_Myopic, y_Myopic = Myopic_Start(c, h, p, u_vector, T, y_L, Total_u, alpha,mean, stddev, a, b, myopic_x_vector_start,k, round_number)
                                        # CalculationTime_Myopic = time.time()-Rolling
                                        # Rolling = time.time()
                                        # print(cost_Myopic, y_Myopic)
                                        # print(counter, round(100 * counter / TotalRuns, 2), " c = " + str(c),"h = " + str(h), "p = " + str(p), "alpha = " + str(alpha),"T = " + str(T), "u Fraction = " + str(u_i), "u = " + str(u),"u_mid = " + str(u_mid), "x_start = " + str(x), "mean = " + str(mean),"k = " + str(k), "cost = " + str(cost), "y = ", y, "Myopic Cost = ",cost_Myopic, "y_L = ", y_L, "Percent Difference = ", 100*(cost_Myopic - cost)/cost, CalculationTime_Optimal, CalculationTime_Myopic,time.time() - start_time)
                                        print(counter, round(100 * counter / TotalRuns, 2), " c = " + str(c),
                                              "h = " + str(h), "p = " + str(p), "alpha = " + str(alpha),
                                              "T = " + str(T), "u Fraction = " + str(u_i), "u = " + str(u),
                                              "u_mid = " + str(u_mid), "x_start = " + str(x), "mean = " + str(mean),
                                              "k = " + str(k), "cost = " + str(cost), "y = ", y, "y_L = ", y_L, CalculationTime_Optimal,
                                               time.time() - start_time)

                                        Consolidated_Results_cost.append(c)
                                        Consolidated_Results_cost.append(h)
                                        Consolidated_Results_cost.append(p)
                                        Consolidated_Results_cost.append(alpha)
                                        Consolidated_Results_cost.append(T)
                                        Consolidated_Results_cost.append(u_i)
                                        Consolidated_Results_cost.append(u)
                                        # Consolidated_Results_cost.append(u_mid_i)
                                        # Consolidated_Results_cost.append(u_mid)
                                        Consolidated_Results_cost.append(x)
                                        Consolidated_Results_cost.append(x_vector_start)
                                        Consolidated_Results_cost.append(mean)
                                        Consolidated_Results_cost.append(k)
                                        Consolidated_Results_cost.append(cost)
                                        # Consolidated_Results_cost.append(cost_Myopic)
                                        Consolidated_Results_cost.append(y)
                                        Consolidated_Results_cost.append(y_L)
                                        # Consolidated_Results_cost.append(100*(cost_Myopic - cost)/cost)
                                        Consolidated_Results_cost.append(CalculationTime_Optimal)
                                        # Consolidated_Results_cost.append(CalculationTime_Myopic)


                                        Overall_Consolidated_Results_cost.append(Consolidated_Results_cost)
                                        Data_cost = pd.DataFrame(Overall_Consolidated_Results_cost, columns=Headers_cost)




                                        Data_cost.to_csv("00_Numerical_Truncated_Normal_Cost_Comparison_varying k and u.csv", encoding='utf-8', index=False)



print("Speed", time.time() - start_time)