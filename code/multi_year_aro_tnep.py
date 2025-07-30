from input_data_processing import weights, RD1, lines, buses, ESS, CG, RES, loads, years_data, sigma_yt_data, tau_yth_data, gamma_dyth_data, gamma_ryth_data, ES_syt0_data, tol
from gamspy import Alias, Container, Domain, Equation, Model, Options, Ord, Card, Parameter, Set, Smax, Sum, Variable
from gamspy.math import power
import sys

def reformat_df(dataframe):
    return dataframe.reset_index().melt(id_vars="index", var_name="Category", value_name="Value")

# Optimization problem definition
m = Container()

# SETS #
# General sets
n = Set(m, name="n", records=buses['Bus'], description="Set of buses indexed by n")
d = Set(m, name="d", records=loads['Load'], description="Set of loads indexed by d")
g = Set(m, name="g", records=CG['Generating unit'], description="Set of conventional units indexed by g")
h = Set(m, name="h", records=RD1['RTP'], description="Set of RTPs indexed by h")
l = Set(m, name="l", records=lines['Transmission line'], description="Set of transmission lines indexed by l")
r = Set(m, name="r", records=RES['Generating unit'], description="Set of renewable units indexed by r")
s = Set(m, name="s", records=ESS['Storage unit'], description="Set of storage units indexed by s")
t = Set(m, name="t", records=weights['RD'], description="Set of RDs indexed by t")
y = Set(m, name="y", records=years_data, description="Set of years indexed by y")
# Subsets of the general sets
le = Set(m, name="le", domain=[l], records=lines[lines['IL_l [$]'] == 0]['Transmission line'], description="Set of existing transmission lines")
lc = Set(m, name="lc", domain=[l], records=lines[lines['IL_l [$]'] > 0]['Transmission line'], description="Set of candidate transmission lines")
rs = Set(m, name="rs", domain=[r], records=RES[RES['Technology'] == 'Solar']['Generating unit'], description="Set of solar units indexed by r")
rw = Set(m, name="rw", domain=[r], records=RES[RES['Technology'] == 'Wind']['Generating unit'], description="Set of wind units indexed by r")

# Multidimensional sets used to make associations between buses, lines and generating units
d_n = Set(m, name="d_n", domain=[d, n], records=loads[['Load', 'Bus']], description="Set of loads connected to bus n")
g_n = Set(m, name="g_n", domain=[g, n], records=CG[['Generating unit', 'Bus']], description="Set of conventional units connected to bus n")
r_n = Set(m, name="r_n", domain=[r, n], records=RES[['Generating unit', 'Bus']], description="Set of renewable units connected to bus n")
s_n = Set(m, name="s_n", domain=[s, n], records=ESS[['Storage unit', 'Bus']], description="Set of storage units connected to bus n")
rel_n = Set(m, name="rel_n", domain=[l, n], records=lines[['Transmission line', 'To bus']], description="Receiving bus of transmission line l")
sel_n = Set(m, name="sel_n", domain=[l, n], records=lines[['Transmission line', 'From bus']], description="Sending bus of transmission line l")

# Sets of indices for the outer and inner loop problems
i = Set(m, name="i", records=[1], description="Iteration of the outer loop")
# j = Set(m, name="j", records=[1], description="Iteration of the outer loop")
v = Set(m, name="v", records=[1], description="Iteration of the inner loop")
# k = Set(m, name="k", records=[1], description="Iteration of the outer loop")

# ALIAS #
yp = Alias(m, name="yp", alias_with=y)

# PARAMETERS #
# Scalars
GammaD = Parameter(m, name="GammaD", records=0, description="Uncertainty budget for increased loads")
GammaGC = Parameter(m, name="GammaGC", records=0, description="Uncertainty budget for increased CG marginal cost")
GammaGP = Parameter(m, name="GammaGP", records=0, description="Uncertainty budget for decreased CG marginal cost")
GammaRS = Parameter(m, name="GammaRS", records=0, description="Uncertainty budget for decreased solar capacity")
GammaRW = Parameter(m, name="GammaRW", records=0, description="Uncertainty budget for decreased wind capacity")
kappa = Parameter(m, name="kappa", records=0.1, description="Discount rate")
IT = Parameter(m, name="IT", records=400000000, description="Investment budget")
nb_H = Parameter(m, name="nb_H", records=8, description="Number of RTPs of each RD")
FL = Parameter(m, name="FL", records=9999999, description="Large constant for disjunctive linearization")
FD = Parameter(m, name="FD", records=9999999, description="Large constant for exact linearization")
FD_up = Parameter(m, name="FD_up", records=9999999, description="Large constant for exact linearization")
FG_up = Parameter(m, name="FG_up", records=9999999, description="Large constant for exact linearization")
FR_up = Parameter(m, name="FR_up", records=9999999, description="Large constant for exact linearization")

gammaD_dyth = Parameter(m, name="gammaD_dth", domain=[d, y, t, h], records=gamma_dyth_data, description="Demand factor of load d")
gammaR_ryth = Parameter(m, name="gammaR_rth", domain=[r, y, t, h], records=gamma_ryth_data, description="Capacity factor of renewable unit r")
zetaD_d_fc = Parameter(m, name="zetaD_d_fc", domain=[d], records=loads[['Load', 'zetaD_d_fc']], description="Annual evolution rate of the forecast peak power consumption of load d")
zetaD_d_max = Parameter(m, name="zetaD_d_max", domain=[d], records=loads[['Load', 'zetaD_d_max']], description="Annual evolution rate of the maximum deviation from the forecast peak power consumption of load d")
zetaGC_g_fc = Parameter(m, name="zetaGC_g_fc", domain=[g], records=CG[['Generating unit', 'zetaGC_g_fc']], description="Annual evolution rate of the forecast marginal production cost of conventional generating unit g")
zetaGC_g_max = Parameter(m, name="zetaGC_g_max", domain=[g], records=CG[['Generating unit', 'zetaGC_g_max']], description="Annual evolution rate of the maximum deviation from the forecast marginal production cost of conventional generating unit g")
zetaGP_g_fc = Parameter(m, name="zetaGP_g_fc", domain=[g], records=CG[['Generating unit', 'zetaGP_g_fc']], description="Annual evolution rate of the forecast capacity of conventional generating unit g")
zetaGP_g_max = Parameter(m, name="zetaGP_g_max", domain=[g], records=CG[['Generating unit', 'zetaGP_g_max']], description="Annual evolution rate of the maximum deviation from the forecast capacity of conventional generating unit g")
zetaR_r_fc = Parameter(m, name="zetaR_r_fc", domain=[r], records=RES[['Generating unit', 'zetaR_r_fc']], description="Annual evolution rate of the forecast capacity of renewable generating unit r")
zetaR_r_max = Parameter(m, name="zetaR_r_max", domain=[r], records=RES[['Generating unit', 'zetaR_r_max']], description="Annual evolution rate of the maximum deviation from the forecast capacity of renewable generating unit r")
etaSC_s = Parameter(m, name="etaSC_s", domain=[s], records=ESS[['Storage unit', 'etaSC_s']], description="Charging efficiency of storage facility")
etaSD_s = Parameter(m, name="etaSD_s", domain=[s], records=ESS[['Storage unit', 'etaSD_s']], description="Discharging efficiency of storage facility")
sigma_yt = Parameter(m, name="sigma_yt", domain=[y, t], records=sigma_yt_data, description="Weight of RD t")
tau_yth = Parameter(m, name="tau_yth", domain=[y, t, h], records=tau_yth_data, description="Duration of RTP h of RD t")

CG_g_fc = Parameter(m, name="CG_g_fc", domain=[g], records=CG[['Generating unit', 'CG_g [$/MWh]']], description="Forecast marginal production cost of conventional unit g in year 1")
CG_g_max = Parameter(m, name="CG_g_max", domain=[g], records=CG[['Generating unit', 'CG_g_max']], description="Maximum deviation from the forecast marginal production cost of conventional generating unit g in year 1")
CLS_d = Parameter(m, name="CLS_d", domain=[d], records=loads[['Load', 'CLS_d [$/MWh]']], description="Load-shedding cost coefficient of load d")
CR_r = Parameter(m, name="CR_r", domain=[r], records=RES[['Generating unit', 'CR_r [$/MWh]']], description="Spillage cost coefficient of renewable unit r")
ES_syt0 = Parameter(m, name="ES_syt0", domain=[s, y, t], records=ES_syt0_data, description="Energy initially stored of storage facility s")
ES_s_min = Parameter(m, name="ES_s_min", domain=[s], records=ESS[['Storage unit', 'ES_smin [MWh]']], description="Minimum energy level of storage facility s")
ES_s_max = Parameter(m, name="ES_s_max", domain=[s], records=ESS[['Storage unit', 'ES_smax [MWh]']], description="Maximum energy level of storage facility s")
IL_l = Parameter(m, name="IL_l", domain=[l], records=lines[['Transmission line', 'IL_l [$]']], description="Investment cost coefficient of candidate transmission line l")
PD_d_fc = Parameter(m, name="PD_d_fc", domain=[d], records=loads[['Load', 'PD_dfc [MW]']], description="Forecast peak power consumption of load d in year 1")
PD_d_max = Parameter(m, name="PD_d_max", domain=[d], records=loads[['Load', 'PD_d_max']], description="Maximum deviation from the forecast peak power consumption of load d in year 1")
PG_g_min = Parameter(m, name="PG_g_min", domain=[g], records=CG[['Generating unit', 'PG_gmin [MW]']], description="Minimum production level of conventional unit g")
PG_g_fc = Parameter(m, name="PG_g_fc", domain=[g], records=CG[['Generating unit', 'PG_gfc [MW]']], description="Forecast capacity of existing conventional unit g in year 1")
PG_g_max = Parameter(m, name="PG_g_max", domain=[g], records=CG[['Generating unit', 'PG_g_max']], description="Maximum deviation from the forecast capacity of conventional generating unit g in year 1")
PL_l = Parameter(m, name="PL_l", domain=[l], records=lines[['Transmission line', 'PL_l']], description="Power flow capacity of transmission line l")
PR_r_fc = Parameter(m, name="PR_r_fc", domain=[r], records=RES[['Generating unit', 'PR_rfc [MW]']], description="Forecast capacity of existing renewable unit r in year 1")
PR_r_max = Parameter(m, name="PR_r_max", domain=[r], records=RES[['Generating unit', 'PR_r_max']], description="Maximum deviation from the forecast capacity of renewable generating unit r in year 1")
PSC_s = Parameter(m, name="PSC_s", domain=[s], records=ESS[['Storage unit', 'PSC_smax [MW]']], description="Charging power capacity of storage facility s")
PSD_s = Parameter(m, name="PSD_s", domain=[s], records=ESS[['Storage unit', 'PSD_smax [MW]']], description="Discharging power capacity of storage facility s")
RGD_g = Parameter(m, name="RGD_g", domain=[g], records=CG[['Generating unit', 'RGD_g [MW]']], description="Ramp-down limit of conventional unit g")
RGU_g = Parameter(m, name="RGU_g", domain=[g], records=CG[['Generating unit', 'RGU_g [MW]']], description="Ramp-up limit of conventional unit")
X_l = Parameter(m, name="X_l", domain=[l], records=lines[['Transmission line', 'X_l']], description="Reactance of transmission line l")

# Parameters used to represent given results for certain variables
cG_gyi = Parameter(m, name='cG_gyi', domain=[g, y, i], description="Worst-case realization of the marginal production cost of conventional generating unit g for uncertainty realization i")


# VARIABLES #
# Optimization variables
theta_nyth = Variable(m, name="theta_nyth", domain=[n, y, t, h], description="Voltage angle at bus n")
xi_y = Variable(m, name='xi_y', domain=[y], description="Auxiliary variable of the inner-loop master problem")
xiP_y = Variable(m, name='xiP_y', domain=[y], description="Auxiliary variable of the first problem solved at each iteration of the ADA when it is applied to the inner-loop master problem")
xiQ_y = Variable(m, name='xiQ_y', domain=[y], description="Auxiliary variable of the second problem solved at each iteration of the ADA when it is applied to the inner-loop master problem")
rho_y = Variable(m, name='rho_y', domain=[y], description="Auxiliary variable of the outer-loop master problem")
aD_dy = Variable(m, type='positive', name='aD_dy', domain=[d, y], description="Continuous variable associated with the deviation that the peak power consumption of load d can experience from its forecast value in year y")
aGC_gy = Variable(m, type='positive', name='aGC_gy', domain=[g, y], description="Continuous variable associated with the deviation that the marginal production cost of conventional generating unit g can experience from its forecast value in year y")
aGP_gy = Variable(m, type='positive', name='aGP_gy', domain=[g, y], description="Continuous variable associated with the deviation that the capacity of conventional generating unit g can experience from its forecast value in year y")
aR_ry = Variable(m, type='positive', name='aR_ry', domain=[r, y], description="Continuous variable associated with the deviation that the capacity of renewable generating unit r can experience from its forecast value in year y")
cG_gy = Variable(m, name='cG_gy', domain=[g, y], description="Worst-case realization of the marginal production cost of conventional generating unit g")
cO_y = Variable(m, name='cO_y', domain=[y], description="Operating costs")
cOWC_y = Variable(m, name='cOWC_y', domain=[y], description="Worst case operating costs")
eS_syth = Variable(m, name='eS_syth', domain=[s, y, t, h], description="Energy stored in storage facility s")
pD_dy = Variable(m, name='pD_dy', domain=[d, y], description="Worst-case realization of the peak power consumption of load d")
pG_gyth = Variable(m, name='pG_gyth', domain=[g, y, t, h], description="Power produced by conventional generating unit g")
pG_gy = Variable(m, name='pG_gy', domain=[g, y], description="Worst-case realization of the capacity of conventional generating unit g")
pL_lyth = Variable(m, name='pL_lyth', domain=[l, y, t, h], description="Power flow through transmission line l")
pLS_dyth = Variable(m, type='positive', name='pLS_dyth', domain=[d, y, t, h], description="Unserved demand of load d")
pR_ryth = Variable(m, type='positive', name='pR_ryth', domain=[r, y, t, h], description="Power produced by renewable generating unit r")
pR_ry = Variable(m, name='pR_ry', domain=[r, y], description="Worst-case realization of the capacity of renewable generating unit r")
pSC_syth = Variable(m, type='positive', name='pSC_syth', domain=[s, y, t, h], description="Charging power of storage facility s")
pSD_syth = Variable(m, type='positive', name='pSD_syth', domain=[s, y, t, h], description="Discharging power of storage facility s")
uG_gyth = Variable(m, name='uG_gyth', type='binary', domain=[g, y, t, h], description="Binary variable used to model the commitment status of conventional unit g")
uS_syth = Variable(m, name='uS_syth', type='binary', domain=[s, y, t, h], description="Binary variable used to used to avoid the simultaneous charging and discharging of storage facility s")
vL_ly = Variable(m, name='vL_ly', type='binary', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y, which is otherwise 0")
vL_ly_prev = Variable(m, name='vL_ly_prev', type='binary', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y or in previous years, which is otherwise 0")
zD_dy = Variable(m, name='zD_dy', type='binary', domain=[d, y], description="Binary variable that is equal to 1 if the worst-case realization of the peak power consumption of load 𝑑 is equal to its upper bound, which is otherwise 0")
zGC_gy = Variable(m, name='zGC_gy', type='binary', domain=[g, y], description="Binary variable that is equal to 1 if the worst-case realization of the marginal production cost of conventional generating unit g is equal to its upper bound, which is otherwise 0")
zGP_gy = Variable(m, name='zGP_gy', type='binary', domain=[g, y], description="Binary variable that is equal to 1 if the worst-case realization of the capacity of conventional generating unit g is equal to its upper bound, which is otherwise 0")
zR_ry = Variable(m, name='zR_ry', type='binary', domain=[r, y], description="Binary variable that is equal to 1 if the worst-case realization of he capacity of renewable generating unit r is equal to its upper bound, which is otherwise 0")

# Dual variables
lambdaN_nyth = Variable(m, name='lambdaN_nyth', domain=[n, y, t, h], description="Dual variable associated with the power balance equation at bus n")
muD_dyth_up = Variable(m, name='muD_dyth_up', type='positive', domain=[d, y, t, h], description="Dual variable associated with the constraint imposing the upper bound for the unserved demand of load d")
muG_gyth_lo = Variable(m, name='muG_gyth_lo', type='positive', domain=[g, y, t, h], description="Dual variable associated with the constraint imposing the lower bound for the power produced by conventional generating unit g")
muG_gyth_up = Variable(m, name='muG_gyth_up', type='positive', domain=[g, y, t, h], description="Dual variable associated with the constraint imposing the upper bound for the power produced by conventional generating unit g")
muGD_gyth = Variable(m, name='muGD_gyth', type='positive', domain=[g, y, t, h], description="Dual variable associated with the constraint imposing the ramp-down limit of conventional generating unit g, being ℎ greater than 1")
muGU_gyth = Variable(m, name='muGU_gyth', type='positive', domain=[g, y, t, h], description="Dual variable associated with the constraint imposing the ramp-up limit of conventional generating unit g, being ℎ greater than 1")
muL_lyth_exist = Variable(m, name='muL_lyth_exist', domain=[le, y, t, h], description="Dual variable associated with the power flow through existing transmission line l")
muL_lyth_can = Variable(m, name='muL_lyth_can', domain=[lc, y, t, h], description="Dual variable associated with the power flow through candidate transmission line l")
muL_lyth_lo = Variable(m, name='muL_lyth_lo', type='positive', domain=[l, y, t, h], description="Dual variable associated with the constraint imposing the lower bound for the power flow through transmission line l")
muL_lyth_up = Variable(m, name='muL_lyth_up', type='positive', domain=[l, y, t, h], description="Dual variable associated with the constraint imposing the upper bound for the power flow through transmission line l")
muR_ryth_up = Variable(m, name='muR_ryth_up', type='positive', domain=[r, y, t, h], description="Dual variable associated with the constraint imposing the upper bound for the power produced by renewable generating unit r")
muS_syth = Variable(m, name='muS_syth', domain=[s, y, t, h], description="Dual variable associated with the energy stored in storage facility s, being h greater than 1")
muS_syth_lo = Variable(m, name='muS_syth_lo', type='positive', domain=[s, y, t, h], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s")
muS_syth_up = Variable(m, name='muS_syth_up', type='positive', domain=[s, y, t, h], description="Dual variable associated with the constraint imposing the upper bound for the energy stored in storage facility s")
muSC_syth_up = Variable(m, name='muSC_syth_up', type='positive', domain=[s, y, t, h], description="Dual variable associated with the constraint imposing the upper bound for the charging power of storage facility s")
muSD_syth_up = Variable(m, name='muSD_syth_up', type='positive', domain=[s, y, t, h], description="Dual variable associated with the constraint imposing the upper bound for the discharging power of storage facility s")
PhiS_syt = Variable(m, name='PhiS_syt', domain=[s, y, t], description="Dual variable associated with the energy stored in storage facility s at the first RTP of RD")
PhiS_syt_lo = Variable(m, name='PhiS_syt_lo', type='positive', domain=[s, y, t], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s at the last RTP of RD")
phiN_nyth = Variable(m, name='phiN_nyth', domain=[n, y, t, h], description="Dual variable associated with the definition of the reference bus n")

# Linearization variables
alphaD_dyth = Variable(m, name="alphaD_dyth", domain=[d, y, t, h], description="Auxiliary variable for the linearization of zD_dy*lambdaN_nyth")
alphaD_dyth_up = Variable(m, name="alphaD_dyth_up", type='positive', domain=[d, y, t, h], description="Auxiliary variable for the linearization of zD_dy*muD_dyth_up")
alphaGP_gyth_up = Variable(m, name="alphaGP_gyth_up", type='positive', domain=[g, y, t, h], description="Auxiliary variable for the linearization of zGP_gy*muGP_gyth_up")
alphaR_ryth_up = Variable(m, name="alphaR_ryth_up", type='positive', domain=[r, y, t, h], description="Auxiliary variable for the linearization of zR_ry*muR_ryth_up")

min_inv_cost_wc = Variable(m, name="min_inv_cost_wc", description="Worst-case investment costs")
min_op_cost_y = Variable(m, name="max_op_cost_wc", description="Minimized operating costs for year y")

# EQUATIONS #
# Outer-loop master problem OF and constraints
# Initialize uncertain parameters to forecast values
cG_gy[g,y].fx = CG_g_fc[g]
pD_dy[d,y].fx = PD_d_fc[d]
pG_gy[g,y].fx = PG_g_fc[g]
pR_ry[r,y].fx = PR_r_fc[r]

OF_olmp = Equation(m, name="OF_olmp", type="regular")
OF_olmp[...] = min_inv_cost_wc == Sum(y, rho_y[y] / power(1.0 + kappa, y.val) + \
                                      (1.0 / power(1.0 + kappa, y.val - 1)) * Sum(lc, IL_l[lc] * vL_ly[lc,y]))
con_1c = Equation(m, name="con_1c")
con_1c[...] = Sum(lc, Sum(y, (1.0 / power(1.0 + kappa, y.val - 1)) * IL_l[lc] * vL_ly[lc,y])) <= IT
con_1d = Equation(m, name="con_1d", domain=[lc])
con_1d[lc] = Sum(y, vL_ly[lc,y]) <= 1
con_1e = Equation(m, name="con_1e", domain=[lc, y])
con_1e[lc,y] = vL_ly_prev[lc,y] == Sum(yp.where[yp.val <= y.val], vL_ly[lc,yp])

con_4c = Equation(m, name="con_4c", domain=[y])
con_4c[y] = rho_y[y] >= Sum(t, sigma_yt[y,t] * Sum(h, tau_yth[y,t,h] * (Sum(g, CG_g_fc[g] * pG_gyth[g,y,t,h]) + \
                        Sum(r, CR_r[r] * (gammaR_ryth[r,y,t,h] * PR_r_fc[r] - pR_ryth[r,y,t,h])) + Sum(d, CLS_d[d] * pLS_dyth[d,y,t,h]))))
con_4d = Equation(m, name="con_4d", domain=[n, y, t, h])
con_4d[n,y,t,h] = Sum(g.where[g_n[g,n]], pG_gyth[g,y,t,h]) + Sum(r.where[r_n[r,n]], pR_ryth[r,y,t,h]) + \
                  Sum(l.where[sel_n[l,n]], pL_lyth[l,y,t,h]) - Sum(l.where[rel_n[l,n]], pL_lyth[l,y,t,h]) + \
                  Sum(s.where[s_n[s,n]], pSD_syth[s,y,t,h] - pSC_syth[s,y,t,h]) == \
                  Sum(d.where[d_n[d,n]], gammaD_dyth[d,y,t,h] * PD_d_fc[d] - pLS_dyth[d,y,t,h])
con_4e = Equation(m, name="con_4e", domain=[le, y, t, h])
con_4e[le,y,t,h] = pL_lyth[le,y,t,h] == (1.0 / X_l[le]) * (Sum(n.where[sel_n[le,n]], theta_nyth[n,y,t,h]) - Sum(n.where[rel_n[le,n]], theta_nyth[n,y,t,h]))
con_4f_lin1 = Equation(m, name="con_4f_lin1", domain=[lc, y, t, h]) # Linearized
con_4f_lin1[lc,y,t,h] = pL_lyth[lc,y,t,h] - (1 / X_l[lc]) * (Sum(n.where[sel_n[lc,n]], theta_nyth[n,y,t,h]) - Sum(n.where[rel_n[lc,n]], theta_nyth[n,y,t,h])) <= (1 - vL_ly_prev[lc,y]) * FL
con_4f_lin2 = Equation(m, name="con_4f_lin2", domain=[lc, y, t, h]) # Linearized
con_4f_lin2[lc,y,t,h] = pL_lyth[lc,y,t,h] - (1 / X_l[lc]) * (Sum(n.where[sel_n[lc,n]], theta_nyth[n,y,t,h]) - Sum(n.where[rel_n[lc,n]], theta_nyth[n,y,t,h])) >= -(1 - vL_ly_prev[lc,y]) * FL
con_4g_exist_lin1 = Equation(m, name="con_4g_exist_lin1", domain=[le, y, t, h])
con_4g_exist_lin1[le,y,t,h] = pL_lyth[le,y,t,h] <= PL_l[le]
con_4g_exist_lin2 = Equation(m, name="con_4g_exist_lin2", domain=[le, y, t, h])
con_4g_exist_lin2[le,y,t,h] = pL_lyth[le,y,t,h] >= -PL_l[le]
con_4g_can_lin1 = Equation(m, name="con_4g_can_lin1", domain=[lc, y, t, h])
con_4g_can_lin1[lc,y,t,h] = pL_lyth[lc,y,t,h] <= vL_ly_prev[lc,y] * PL_l[lc]
con_4g_can_lin2 = Equation(m, name="con_4g_can_lin2", domain=[lc, y, t, h])
con_4g_can_lin2[lc,y,t,h] = pL_lyth[lc,y,t,h] >= -vL_ly_prev[lc,y] * PL_l[lc]
con_4h = Equation(m, name="con_4h", domain=[s, y, t]) # H == 1
con_4h[s,y,t] = eS_syth[s,y,t,1] == ES_syt0[s,y,t] + (pSC_syth[s,y,t,1] * etaSC_s[s] - (pSD_syth[s,y,t,1] / etaSD_s[s])) * tau_yth[y,t,1]
con_4i = Equation(m, name="con_4i", domain=[s, y, t, h]) # H =/= 1
con_4i[s,y,t,h].where[Ord(h) > 1] = eS_syth[s,y,t,h] == eS_syth[s,y,t,h.lag(1)] + (pSC_syth[s,y,t,h] * etaSC_s[s] - (pSD_syth[s,y,t,h] / etaSD_s[s])) * tau_yth[y,t,h]
con_4j = Equation(m, name="con_4j", domain=[s, y, t, h]) # H == Hmax
con_4j[s,y,t,h].where[Ord(h) == Card(h)] = ES_syt0[s,y,t] <= eS_syth[s,y,t,h]
con_4k1 = Equation(m, name="con_4k1", domain=[s, y, t, h])
con_4k1[s,y,t,h] = eS_syth[s,y,t,h] <= ES_s_max[s]
con_4k2 = Equation(m, name="con_4k2", domain=[s, y, t, h])
con_4k2[s,y,t,h] = eS_syth[s,y,t,h] >= ES_s_min[s]
# con_4l = Equation(m, name="con_4l", domain=[S,T,H,Y])
con_4m1 = Equation(m, name="con_4m1", domain=[s, y, t, h])
con_4m1[s,y,t,h] = pSC_syth[s,y,t,h] <= uS_syth[s,y,t,h] * PSC_s[s]
con_4m2 = Equation(m, name="con_4m2", domain=[s, y, t, h])
con_4m2[s,y,t,h] = pSC_syth[s,y,t,h] >= 0
con_4n1 = Equation(m, name="con_4n1", domain=[s, y, t, h])
con_4n1[s,y,t,h] = pSD_syth[s,y,t,h] <= (1 - uS_syth[s,y,t,h]) * PSD_s[s]
con_4n2 = Equation(m, name="con_4n2", domain=[s, y, t, h])
con_4n2[s,y,t,h] = pSD_syth[s,y,t,h] >= 0
con_4o1 = Equation(m, name="con_4o1", domain=[d, y, t, h])
con_4o1[d,y,t,h] = pLS_dyth[d,y,t,h] <= gammaD_dyth[d,y,t,h] * PD_d_fc[d]#pD_dy[D,Y]
con_4o2 = Equation(m, name="con_4o2", domain=[d, y, t, h])
con_4o2[d,y,t,h] = pLS_dyth[d,y,t,h] >= 0
# con_4p = Equation(m, name="con_4p", domain=[G,T,H,Y])
con_4q1 = Equation(m, name="con_4q1", domain=[g, y, t, h])
con_4q1[g,y,t,h] = pG_gyth[g,y,t,h] <= uG_gyth[g,y,t,h] * PG_g_fc[g]#pG_gy[G,Y]
con_4q2 = Equation(m, name="con_4q2", domain=[g, y, t, h])
con_4q2[g,y,t,h] = pG_gyth[g,y,t,h] >= uG_gyth[g,y,t,h] * PG_g_min[g]
con_4r1 = Equation(m, name="con_4r1", domain=[g, y, t, h]) # H =/= 1
con_4r1[g,y,t,h].where[Ord(h) > 1] = pG_gyth[g,y,t,h] - pG_gyth[g,y,t,h.lag(1)] <= RGU_g[g]
con_4r2 = Equation(m, name="con_4r2", domain=[g, y, t, h]) # H =/= 1
con_4r2[g,y,t,h].where[Ord(h) > 1] = pG_gyth[g,y,t,h] - pG_gyth[g,y,t,h.lag(1)] >= -RGD_g[g]
con_4s1 = Equation(m, name="con_4s1", domain=[r, y, t, h])
con_4s1[r,y,t,h] = pR_ryth[r,y,t,h] <= gammaR_ryth[r,y,t,h] * PR_r_fc[r]#pR_ry[R,Y]
con_4s2 = Equation(m, name="con_4s2", domain=[r, y, t, h])
con_4s2[r,y,t,h] = pR_ryth[r,y,t,h] >= 0
con_4t = Equation(m, name="con_4t", domain=[n, y, t, h]) # N == ref bus
con_4t[n,y,t,h].where[Ord(n)==1] = theta_nyth[n,y,t,h] == 0

olmp_eqns = [OF_olmp, con_1c, con_1d, con_1e, con_4c, con_4d, con_4e, con_4f_lin1, con_4f_lin2, con_4g_exist_lin1,
             con_4g_exist_lin2, con_4g_can_lin1, con_4g_can_lin2, con_4h, con_4i, con_4j, con_4k1, con_4k2, con_4m1,
             con_4m2, con_4n1, con_4n2, con_4o1, con_4o2, con_4q1, con_4q2, con_4r1, con_4r2, con_4s1, con_4s2, con_4t]

## Outer-loop subproblem
# Inner-loop master problem OF and constraints
yi = 1
# OF_ilmp = Equation(m, name="OF_ilmp", type="regular")
con_2b = Equation(m, name="con_2b", domain=[g, y])
con_2b[g,y] = cG_gy[g,y] == CG_g_fc[g] * power(1 + zetaGC_g_fc[g], y.val - 1) + CG_g_max[g] * power(1 + zetaGC_g_max[g], y.val - 1) * zGC_gy[g,y]
con_2c = Equation(m, name="con_2c", domain=[d, y])
con_2c[d,y] = pD_dy[d,y] == PD_d_fc[d] * power(1 + zetaD_d_fc[d], y.val - 1) + PD_d_max[d] * power(1 + zetaD_d_max[d], y.val - 1) * zD_dy[d,y]
con_2d = Equation(m, name="con_2d", domain=[g, y])
con_2d[g,y] = pG_gy[g,y] == PG_g_fc[g] * power(1 + zetaGP_g_fc[g], y.val - 1) - PG_g_max[g] * power(1 + zetaGP_g_max[g], y.val - 1) * zGP_gy[g,y]
con_2e = Equation(m, name="con_2e", domain=[r, y])
con_2e[r,y] = pR_ry[r,y] == PR_r_fc[r] * power(1 + zetaR_r_fc[r], y.val - 1) - PR_r_max[r] * power(1 + zetaR_r_max[r], y.val - 1) * zR_ry[r,y]
# con_2f = Equation(m, name="con_2f", domain=[G,Y])
# con_2g = Equation(m, name="con_2g", domain=[D,Y])
# con_2h = Equation(m, name="con_2h", domain=[G,Y])
# con_2i = Equation(m, name="con_2i", domain=[R,Y])
con_2j = Equation(m, name="con_2j", domain=[y])
con_2j[y] = Sum(g, zGC_gy[g,y]) <= GammaGC
con_2k = Equation(m, name="con_2k", domain=[y])
con_2k[y] = Sum(d, zD_dy[d,y]) <= GammaD
con_2l = Equation(m, name="con_2l", domain=[y])
con_2l[y] = Sum(g, zGP_gy[g,y]) <= GammaGP
con_2m = Equation(m, name="con_2m", domain=[y])
con_2m[y] = Sum(rs, zR_ry[rs,y]) <= GammaRS
con_2n = Equation(m, name="con_2n", domain=[y])
con_2n[y] = Sum(rw, zR_ry[rw,y]) <= GammaRW

# con_5c = Equation(m, name="con_5c")
# con_5c[...] = xi_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h] * pD_dy[d,yi] * Sum(n.where[d_n[d,n]], lambdaN_nyth[n,yi,t,h])) - \
#                           Sum(l, PL_l[l] * (muL_lyth_lo[l,yi,t,h] + muL_lyth_up[l,yi,t,h])) - \
#                           Sum(s, uS_syth[s,yi,t,h] * PSC_s[s] * muSC_syth_up[s,yi,t,h] + (1 - uS_syth[s,yi,t,h]) * PSD_s[s] * muSD_syth_up[s,yi,t,h] - ES_s_min[s] * muS_syth_lo[s,yi,t,h] + ES_s_max[s] * muS_syth_up[s,yi,t,h]) + \
#                           Sum(g, uG_gyth[g,yi,t,h] * (PG_g_min[g] * muG_gyth_lo[g,yi,t,h] - pG_gy[g,yi] * muG_gyth_up[g,yi,t,h])) - \
#                           Sum(r, gammaR_ryth[r,yi,t,h] * pR_ry[r,yi] * (muR_ryth_up[r,yi,t,h] - sigma_yt[yi,t] * tau_yth[yi,t,h] * CR_r[r])) - \
#                           Sum(d, gammaD_dyth[d,yi,t,h] * pD_dy[d,yi] * muD_dyth_up[d,yi,t,h])) + \
#                           Sum(s, ES_syt0[s,yi,t] * (PhiS_syt[s,yi,t] + PhiS_syt_lo[s,yi,t])) - \
#                           Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g] * muGD_gyth[g,yi,t,h] + RGU_g[g] * muGU_gyth[g,yi,t,h])))
con_5c_lin_a = Equation(m, name="con_5c_lin_a")
con_5c_lin_a[...] = xi_y[yi] <= Sum(t, Sum(h,\
                    Sum(d, gammaD_dyth[d,yi,t,h]*(PD_d_fc[d]*power(1+zetaD_d_fc[d], yi-1)*Sum(n.where[d_n[d,n]], lambdaN_nyth[n,yi,t,h]) + PD_d_max[d]*power(1+zetaD_d_max[d], yi-1)*alphaD_dyth[d,yi,t,h]))\
                    -Sum(l, PL_l[l]*(muL_lyth_lo[l,yi,t,h] + muL_lyth_up[l,yi,t,h]))\
                    -Sum(s, uS_syth[s,yi,t,h]*PSC_s[s]*muSC_syth_up[s,yi,t,h] + (1-uS_syth[s,yi,t,h])*PSD_s[s]*muSD_syth_up[s,yi,t,h] - ES_s_min[s]*muS_syth_lo[s,yi,t,h] + ES_s_max[s]*muS_syth_up[s,yi,t,h])\
                    +Sum(g, uG_gyth[g,yi,t,h]*(PG_g_min[g]*muG_gyth_lo[g,yi,t,h] - (PG_g_fc[g]*power(1-zetaGP_g_fc[g], yi-1)*muG_gyth_up[g,yi,t,h] - PG_g_max[g]*power(1+zetaGP_g_max[g], yi-1)*alphaGP_gyth_up[g,yi,t,h])))\
                    -Sum(r, gammaR_ryth[r,yi,t,h]*(PR_r_fc[r]*power(1+zetaR_r_fc, yi-1)*muR_ryth_up[r,yi,t,h] - PR_r_max[r]*power(1+zetaR_r_max, yi-1)*alphaR_ryth_up[r,yi,t,h]))\
                    +Sum(r, sigma_yt[yi,t]*tau_yth[yi,t,h]*CR_r[r]*gammaR_ryth[r,yi,t,h]*(PR_r_fc[r]*power(1+zetaR_r_fc, yi-1) - PR_r_max[r]*power(1+zetaR_r_max, yi-1)*zR_ry[r,yi]))\
                    -Sum(d, gammaD_dyth[d,yi,t,h]*(PD_d_fc[d]*power(1+zetaD_d_fc[d], yi-1)*muD_dyth_up[d,yi,t,h] + PD_d_max[d]*power(1+zetaD_d_max[d], yi-1)*alphaD_dyth_up[d,yi,t,h])))\
                    +Sum(s, ES_syt0[s,yi,t] * (PhiS_syt[s,yi,t] + PhiS_syt_lo[s,yi,t]))\
                    -Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g]*muGD_gyth[g,yi,t,h] + RGU_g[g]*muGU_gyth[g,yi,t,h])))
con_5c_lin_b1 = Equation(m, name="con_5c_lin_b1", domain=[d, t, h])
con_5c_lin_b1[d,t,h] = alphaD_dyth[d,yi,t,h] <= zD_dy[d,yi]*FD
con_5c_lin_b2 = Equation(m, name="con_5c_lin_b2", domain=[d, t, h])
con_5c_lin_b2[d,t,h] = alphaD_dyth[d,yi,t,h] >= -zD_dy[d,yi]*FD
con_5c_lin_c1 = Equation(m, name="con_5c_lin_c1", domain=[d, t, h])
con_5c_lin_c1[d,t,h] = Sum(n.where[d_n[d,n]], lambdaN_nyth[n,yi,t,h]) - alphaD_dyth[d,yi,t,h] <= (1-zD_dy[d,yi])*FD
con_5c_lin_c2 = Equation(m, name="con_5c_lin_c2", domain=[d, t, h])
con_5c_lin_c2[d,t,h] = Sum(n.where[d_n[d,n]], lambdaN_nyth[n,yi,t,h]) - alphaD_dyth[d,yi,t,h] >= -(1-zD_dy[d,yi])*FD
con_5c_lin_d = Equation(m, name="con_5c_lin_d", domain=[d, t, h])
con_5c_lin_d[d,t,h] = alphaD_dyth_up[d,yi,t,h] <= zD_dy[d,yi]*FD_up
con_5c_lin_e1 = Equation(m, name="con_5c_lin_e1", domain=[d, t, h])
con_5c_lin_e1[d,t,h] = muD_dyth_up[d,yi,t,h] - alphaD_dyth_up[d,yi,t,h] <= (1-zD_dy[d,yi])*FD_up
con_5c_lin_e2 = Equation(m, name="con_5c_lin_e2", domain=[d, t, h])
con_5c_lin_e2[d,t,h] = muD_dyth_up[d,yi,t,h] - alphaD_dyth_up[d,yi,t,h] >= 0
con_5c_lin_f = Equation(m, name="con_5c_lin_f", domain=[g, t, h])
con_5c_lin_f[g,t,h] = alphaGP_gyth_up[g,yi,t,h] <= zGP_gy[g,yi]*FG_up
con_5c_lin_g1 = Equation(m, name="con_5c_lin_g1", domain=[g, t, h])
con_5c_lin_g1[g,t,h] = muG_gyth_up[g,yi,t,h] - alphaGP_gyth_up[g,yi,t,h] <= (1-zGP_gy[g,yi])*FG_up
con_5c_lin_g2 = Equation(m, name="con_5c_lin_g2", domain=[g, t, h])
con_5c_lin_g2[g,t,h] = muG_gyth_up[g,yi,t,h] - alphaGP_gyth_up[g,yi,t,h] >= 0
con_5c_lin_h = Equation(m, name="con_5c_lin_h", domain=[r, t, h])
con_5c_lin_h[r,t,h] = alphaR_ryth_up[r,yi,t,h] <= zR_ry[r,yi]*FR_up
con_5c_lin_i1 = Equation(m, name="con_5c_lin_i1", domain=[r, t, h])
con_5c_lin_i1[r,t,h] = muR_ryth_up[r,yi,t,h] - alphaR_ryth_up[r,yi,t,h] <= (1-zR_ry[r,yi])*FR_up
con_5c_lin_i2 = Equation(m, name="con_5c_lin_i2", domain=[r, t, h])
con_5c_lin_i1[r,t,h] = muR_ryth_up[r,yi,t,h] - alphaR_ryth_up[r,yi,t,h] >= 0

con_5def = Equation(m, name="con_5d", domain=[g, t, h])
con_5def[g,t,h] = Sum(n.where[g_n[g,n]], lambdaN_nyth[n,yi,t,h]) + muG_gyth_lo[g,yi,t,h] - muG_gyth_up[g,yi,t,h] + muGD_gyth[g,yi,t,h] - \
                  muGD_gyth[g,yi,t,h.lead(1)] - muGU_gyth[g,yi,t,h] + muGU_gyth[g,yi,t,h.lead(1)] == sigma_yt[yi,t] * tau_yth[yi,t,h] * cG_gy[g,yi]
con_5def[g,t,h].where[Ord(h) == 1] = Sum(n.where[g_n[g,n]], lambdaN_nyth[n,yi,t,h]) + muG_gyth_lo[g,yi,t,h] - muG_gyth_up[g,yi,t,h] - muGD_gyth[g,yi,t,h.lead(1)] + \
                                     muGU_gyth[g,yi,t,h.lead(1)] == sigma_yt[yi,t] * tau_yth[yi,t,h] * cG_gy[g,yi]
con_5def[g,t,h].where[Ord(h) == Card(h)] = Sum(n.where[g_n[g,n]], lambdaN_nyth[n,yi,t,h]) + muG_gyth_lo[g,yi,t,h] - muG_gyth_up[g,yi,t,h] + muGD_gyth[g,yi,t,h] - \
                                           muGU_gyth[g,yi,t,h] == sigma_yt[yi,t] * tau_yth[yi,t,h] * cG_gy[g,yi]
con_5g = Equation(m, name="con_5g", domain=[d, t, h])
con_5g[d,t,h] = Sum(n.where[d_n[d,n]], lambdaN_nyth[n,yi,t,h]) - muD_dyth_up[d,yi,t,h] <= sigma_yt[yi,t] * tau_yth[yi,t,h] * CLS_d[d]
con_5h = Equation(m, name="con_5h", domain=[r, t, h])
con_5h[r,t,h] = Sum(n.where[r_n[r,n]], lambdaN_nyth[n,yi,t,h]) - muR_ryth_up[r,yi,t,h] <= sigma_yt[yi,t] * tau_yth[yi,t,h] * CR_r[r]
con_5i = Equation(m, name="con_5i", domain=[le, t, h])
con_5i[le,t,h] = Sum(n.where[rel_n[le,n]], lambdaN_nyth[n,yi,t,h]) - Sum(n.where[sel_n[le,n]], lambdaN_nyth[n,yi,t,h]) \
                 + muL_lyth_exist[le,yi,t,h] + muL_lyth_lo[le,yi,t,h] - muL_lyth_up[le,yi,t,h] == 0
con_5j = Equation(m, name="con_5j", domain=[lc, t, h])
con_5j[lc,t,h] = Sum(n.where[rel_n[lc,n]], lambdaN_nyth[n,yi,t,h]) - Sum(n.where[sel_n[lc,n]], lambdaN_nyth[n,yi,t,h]) \
                 + muL_lyth_can[lc,yi,t,h] + muL_lyth_lo[lc,yi,t,h] - muL_lyth_up[lc,yi,t,h] == 0
con_5k = Equation(m, name="con_5k", domain=[s, t, h])
con_5k[s,t,h].where[Ord(h)==1] = Sum(n.where[s_n[s,n]], lambdaN_nyth[n,yi,t,h])+(tau_yth[yi,t,h]/etaSD_s[s])*PhiS_syt[s,yi,t]-\
                                 muSD_syth_up[s,yi,t,h] <=  0
con_5l = Equation(m, name="con_5l", domain=[s, t, h])
con_5l[s,t,h].where[Ord(h)>1] = Sum(n.where[s_n[s,n]], lambdaN_nyth[n,yi,t,h])+(tau_yth[yi,t,h]/etaSD_s[s])*muS_syth[s,yi,t,h]-\
                                 muSD_syth_up[s,yi,t,h] <=  0
con_5m = Equation(m, name="con_5m", domain=[s, t, h])
con_5m[s,t,h].where[Ord(h)==1] = -Sum(n.where[s_n[s,n]], lambdaN_nyth[n,yi,t,h])-etaSC_s[s]*tau_yth[yi,t,h]*PhiS_syt[s,yi,t]-\
                                 muSC_syth_up[s,yi,t,h] <= 0
con_5n = Equation(m, name="con_5n", domain=[s, t, h])
con_5n[s,t,h].where[Ord(h)>1] = -Sum(n.where[s_n[s,n]], lambdaN_nyth[n,yi,t,h])-etaSC_s[s]*tau_yth[yi,t,h]*muS_syth[s,yi,t,h]-\
                                 muSC_syth_up[s,yi,t,h] <= 0
con_5o = Equation(m, name="con_5o", domain=[n,t,h]) # N =/= ref bus
con_5o[n,t,h].where[Ord(n)>1] = -Sum(le.where[sel_n[le,n]], muL_lyth_exist[le,yi,t,h]/X_l[le])+\
                                  Sum(le.where[rel_n[le,n]], muL_lyth_exist[le,yi,t,h]/X_l[le])-\
                                  Sum(lc.where[sel_n[lc,n]], (vL_ly_prev[lc,yi]/X_l[lc])*muL_lyth_can[lc,yi,t,h])+\
                                  Sum(lc.where[rel_n[lc,n]], (vL_ly_prev[lc,yi]/X_l[lc])*muL_lyth_can[lc,yi,t,h]) == 0
con_5p = Equation(m, name="con_5p", domain=[n,t,h]) # N == ref bus
con_5p[n,t,h].where[Ord(n)==1] = -Sum(le.where[sel_n[le,n]], muL_lyth_exist[le,yi,t,h]/X_l[le])+\
                                  Sum(le.where[rel_n[le,n]], muL_lyth_exist[le,yi,t,h]/X_l[le])-\
                                  Sum(lc.where[sel_n[lc,n]], (vL_ly_prev[lc,yi]/X_l[lc])*muL_lyth_can[lc,yi,t,h])+\
                                  Sum(lc.where[rel_n[lc,n]], (vL_ly_prev[lc,yi]/X_l[lc])*muL_lyth_can[lc,yi,t,h])+phiN_nyth[n,yi,t,h] == 0
con_5qrs = Equation(m, name="con_5q", domain=[s,t,h])
con_5qrs[s,t,h] = muS_syth_up[s,yi,t,h]-muS_syth_up[s,yi,t,h.lead(1)]+muS_syth_lo[s,yi,t,h]-muS_syth_up[s,yi,t,h] == 0
con_5qrs[s,t,h].where[Ord(h)==1] = PhiS_syt[s,yi,t]-muS_syth_up[s,yi,t,h.lead(1)]+muS_syth_lo[s,yi,t,h]-muS_syth_up[s,yi,t,h] == 0
con_5qrs[s,t,h].where[Ord(h)==Card(h)] = muS_syth[s,yi,t,h]+PhiS_syt_lo[s,yi,t]+muS_syth_lo[s,yi,t,h]-muS_syth_up[s,yi,t,h] == 0

ilmp_eqns = [con_2b, con_2c, con_2d, con_2e, con_2j, con_2k, con_2l, con_2m, con_2n, con_5c_lin_a, con_5c_lin_b1,
             con_5c_lin_b2, con_5c_lin_c1, con_5c_lin_c2, con_5c_lin_d, con_5c_lin_e1, con_5c_lin_e2, con_5c_lin_f,
             con_5c_lin_g1, con_5c_lin_g2, con_5c_lin_h, con_5c_lin_i1, con_5c_lin_i2, con_5def, con_5g, con_5h, con_5i,
             con_5j, con_5k, con_5l, con_5m, con_5n, con_5o, con_5p, con_5qrs]

# Inner-loop subproblem OF and constraints
OF_ilsp = Equation(m, name="OF_ilsp", type="regular") # Double-check
OF_ilsp[...] = min_op_cost_y == Sum(t, Sum(h, sigma_yt[yi,t]*tau_yth[yi,t,h]*(Sum(g, cG_gy[g,yi]*pG_gyth[g,yi,t,h]) +\
          Sum(r, CR_r[r]*(gammaR_ryth[r,yi,t,h]*pR_ry[r,yi]-pR_ryth[r,yi,t,h])) + Sum(d, CLS_d[d]*pLS_dyth[d,yi,t,h]))))
con_6b = Equation(m, name="con_6b", domain=[n, t, h])
con_6b[n,t,h] = Sum(g, pG_gyth[g,yi,t,h]) + Sum(r, pR_ryth[r,yi,t,h]) + Sum(l.where[rel_n[l,n]], pL_lyth[l,yi,t,h]) -\
                Sum(l.where[sel_n[l,n]], pL_lyth[l,yi,t,h]) + Sum(s, pSD_syth[s,yi,t,h]-pSC_syth[s,yi,t,h]) ==\
                Sum(d, gammaD_dyth[d,yi,t,h]*pD_dy[d,yi]-pLS_dyth[d,yi,t,h])
con_6c = Equation(m, name="con_6c", domain=[lc, t, h])
con_6c[lc,t,h] = pL_lyth[lc,yi,t,h] == (vL_ly_prev[lc,yi]/X_l[lc])*(Sum(n.where[sel_n[lc,n]], theta_nyth[n,yi,t,h])-\
                 Sum(n.where[rel_n[lc,n]], theta_nyth[n,yi,t,h]))
con_6d = Equation(m, name="con_6d", domain=[d, t, h])
con_6d[d,t,h] = pLS_dyth[d,yi,t,h] <= gammaD_dyth[d,yi,t,h]*pD_dy[d,yi]
con_6e1 = Equation(m, name="con_6e1", domain=[g, t, h])
con_6e1 = pG_gyth[g,yi,t,h] <= uG_gyth[g,yi,t,h]*pG_gy[g,yi]
con_6e2 = Equation(m, name="con_6e2", domain=[g, t, h])
con_6e2 = pG_gyth[g,yi,t,h] >= uG_gyth[g,yi,t,h]*PG_g_min[g]
con_6f = Equation(m, name="con_6f", domain=[r, t, h])
con_6f[r,t,h] = pR_ryth[r,yi,t,h] <= gammaR_ryth[r,yi,t,h]*pR_ry[r,yi]

con_3c = Equation(m, name="con_3c", domain=[lc, t, h])
con_3c[lc,t,h] = pL_lyth[lc,yi,t,h] == (1.0/X_l[lc])*(Sum(n.where[sel_n[lc,n]], theta_nyth[n,yi,t,h])-\
                 Sum(n.where[rel_n[lc,n]], theta_nyth[n,yi,t,h]))
con_3e1 = Equation(m, name="con_3e1", domain=[l, t, h])
con_3e1[l,t,h] = pL_lyth[l,yi,t,h] <= PL_l[l]
con_3e2 = Equation(m, name="con_3e2", domain=[l, t, h])
con_3e2[l,t,h] = pL_lyth[l,yi,t,h] <= -PL_l[l]
con_3fg = Equation(m, name="con_3fg", domain=[s, t, h])
con_3fg[s,t,h] = eS_syth[s,yi,t,h] == eS_syth[s,yi,t,h.lag(1)] + (pSC_syth[s,yi,t,h]*etaSC_s[s]-(pSD_syth[s,yi,t,h]/etaSD_s[s]))*tau_yth[yi,t,h]
con_3fg[s,t,h].where[Ord(h)==1] = eS_syth[s,yi,t,h] == ES_syt0[s,yi,t] + (pSC_syth[s,yi,t,h]*etaSC_s[s]-(pSD_syth[s,yi,t,h]/etaSD_s[s]))*tau_yth[yi,t,h]
con_3h = Equation(m, name="con_3h", domain=[s, t, h])
con_3h[s,t,h] = eS_syth[s,yi,t,h] >= ES_syt0[s,yi,t]
con_3i1 = Equation(m, name="con_3i1", domain=[s, t, h])
con_3i1[s,t,h] = eS_syth[s,yi,t,h] <= ES_s_max[s]
con_3i2 = Equation(m, name="con_3i2", domain=[s, t, h])
con_3i2[s,t,h] = eS_syth[s,yi,t,h] >= ES_s_min[s]
# con_3j = Equation(m, name="con_3j", domain=[n, t, h])
con_3k = Equation(m, name="con_3k", domain=[s, t, h])
con_3k[s,t,h] = pSC_syth[s,yi,t,h] <= uS_syth[s,yi,t,h]*PSC_s[s]
con_3l = Equation(m, name="con_3l", domain=[s, t, h])
con_3l[s,t,h] = pSD_syth[s,yi,t,h] <= (1-uS_syth[s,yi,t,h])*PSD_s[s]
# con_3n = Equation(m, name="con_3n", domain=[n, t, h])
con_3p1 = Equation(m, name="con_3p1", domain=[g, t, h])
con_3p1[g,t,h] = pG_gyth[g,yi,t,h] - pG_gyth[g,yi,t,h.lag(1)] <= RGU_g[g]
con_3p2 = Equation(m, name="con_3p2", domain=[g, t, h])
con_3p2[g,t,h] = pG_gyth[g,yi,t,h] - pG_gyth[g,yi,t,h.lag(1)] >= -RGD_g[g]
con_3r = Equation(m, name="con_3r", domain=[n, t, h]) # n == ref bus
con_3r[n,t,h].where[Ord(n)==1] = theta_nyth[n,yi,t,h] == 0

ilsp_eqns = [OF_ilsp, con_6b, con_6c, con_6d, con_6e1, con_6e2, con_6f, con_3c, con_3e1, con_3e2, con_3fg, con_3h,
             con_3i1, con_3i2, con_3k, con_3l, con_3p1, con_3p2, con_3r]

## ADA-based initialization of the inner loop
# First linear problem OF and constraints
con_7b = Equation(m, name="con_7b", domain=[g])
con_7b[g] = cG_gy[g,yi] == CG_g_fc[g]*power(1+zetaGC_g_fc, yi-1) + CG_g_max[g]*power(1+zetaGC_g_max, yi-1)*aGC_gy[g,yi]
con_7c = Equation(m, name="con_7c", domain=[g])
con_7c[g] = aGC_gy[g,yi] <= 1
con_7d = Equation(m, name="con_7d")
con_7d[...] = Sum(g, aGC_gy[g,yi]) <= GammaGC
con_7e = Equation(m, name="con_7e")
con_7e[...] = xiP_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h]*pD_dy[d,yi]*Sum(n.where[d_n[d,n]], lambdaN_nyth[n,yi,t,h]))-\
              Sum(l, PL_l[l]*(muL_lyth_lo[l,yi,t,h]+muL_lyth_up[l,yi,t,h])) - Sum(s, uS_syth[s,yi,t,h]*PSC_s[s]*muSC_syth_up[s,yi,t,h]+ \
              (1-uS_syth[s,yi,t,h])*PSD_s[s]*muSD_syth_up[s,yi,t,h]-ES_s_min[s]*muS_syth_lo[s,yi,t,h]+ES_s_max[s]*muS_syth_up[s,yi,t,h]) +\
              Sum(g, uG_gyth[g,yi,t,h]*(PG_g_min[g]*muG_gyth_lo[g,yi,t,h]-pG_gy[g,yi]*muG_gyth_up[g,yi,t,h])) -\
              Sum(r, gammaR_ryth[r,yi,t,h]*pR_ry[r,yi]*(muR_ryth_up[r,yi,t,h]-sigma_yt[yi,t]*tau_yth[yi,t,h]*CR_r[r])) -\
              Sum(d, gammaD_dyth[d,yi,t,h]*pD_dy[d,yi]*muD_dyth_up[d,yi,t,h])) + Sum(s, ES_syt0[s,yi,t]*(PhiS_syt[s,yi,t]+PhiS_syt_lo[s,yi,t])) -\
              Sum(h.where[Ord(h)>1], Sum(g, RGD_g[g]*muGD_gyth[g,yi,t,h]+RGU_g[g]*muGU_gyth[g,yi,t,h])))
# Constraints 5d-5z

lp1_eqns = [con_7b, con_7c, con_7d, con_7e, con_5def, con_5g, con_5h, con_5i, con_5j, con_5j, con_5k, con_5l, con_5m,
            con_5n, con_5o, con_5p, con_5qrs]

# Second linear problem OF and constraints
con_8b = Equation(m, name="con_8b", domain=[d])
con_8b[d] = pD_dy[d,yi] == PD_d_fc[d]*power(1+zetaD_d_fc[d], yi-1) + PD_d_max[d]*power(1+zetaD_d_max[d], yi-1)*aD_dy[d,yi]
con_8c = Equation(m, name="con_8c", domain=[g])
con_8c[g] = pG_gy[g,yi] == PG_g_fc[g]*power(1-zetaGP_g_fc[g], yi-1) - PG_g_max[g]*power(1+zetaGP_g_max[g], yi-1)*aGP_gy[g,yi]
con_8d = Equation(m, name="con_8d", domain=[r])
con_8d[r] = pR_ry[r,yi] == PR_r_fc[r]*power(1+zetaR_r_fc[r], yi-1) - PR_r_max[r]*power(1+zetaR_r_max[r], yi-1)*aR_ry[r,yi]
con_8e = Equation(m, name="con_8e", domain=[d])
con_8e[d] = aD_dy[d,yi] <= 1
con_8f = Equation(m, name="con_8f", domain=[g])
con_8f[g] = aGP_gy[g,yi] <= 1
con_8g = Equation(m, name="con_8g", domain=[r])
con_8g[r] = aR_ry[r,yi] <= 1
con_8h = Equation(m, name="con_8h")
con_8h[...] = Sum(d, aD_dy[d,yi]) <= GammaD
con_8i = Equation(m, name="con_8i")
con_8i[...] = Sum(g, aGP_gy[g,yi]) <= GammaGP
con_8j = Equation(m, name="con_8j")
con_8j[...] = Sum(rs, aR_ry[rs,yi]) <= GammaRS
con_8k = Equation(m, name="con_8k")
con_8k[...] = Sum(rw, aR_ry[rw,yi]) <= GammaRW
con_8l = Equation(m, name="con_8l")
con_8l[...] = xiQ_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h]*pD_dy[d,yi]*Sum(n.where[d_n[d,n]], lambdaN_nyth[n,yi,t,h]))-\
              Sum(l, PL_l[l]*(muL_lyth_lo[l,yi,t,h]+muL_lyth_up[l,yi,t,h])) - Sum(s, uS_syth[s,yi,t,h]*PSC_s[s]*muSC_syth_up[s,yi,t,h]+ \
              (1-uS_syth[s,yi,t,h])*PSD_s[s]*muSD_syth_up[s,yi,t,h]-ES_s_min[s]*muS_syth_lo[s,yi,t,h]+ES_s_max[s]*muS_syth_up[s,yi,t,h]) +\
              Sum(g, uG_gyth[g,yi,t,h]*(PG_g_min[g]*muG_gyth_lo[g,yi,t,h]-pG_gy[g,yi]*muG_gyth_up[g,yi,t,h])) -\
              Sum(r, gammaR_ryth[r,yi,t,h]*pR_ry[r,yi]*(muR_ryth_up[r,yi,t,h]-sigma_yt[yi,t]*tau_yth[yi,t,h]*CR_r[r])) -\
              Sum(d, gammaD_dyth[d,yi,t,h]*pD_dy[d,yi]*muD_dyth_up[d,yi,t,h])) + Sum(s, ES_syt0[s,yi,t]*(PhiS_syt[s,yi,t]+PhiS_syt_lo[s,yi,t])) -\
              Sum(h.where[Ord(h)>1], Sum(g, RGD_g[g]*muGD_gyth[g,yi,t,h]+RGU_g[g]*muGU_gyth[g,yi,t,h])))

lp2_eqns = [con_8b, con_8c, con_8d, con_8e, con_8f, con_8g, con_8h, con_8i, con_8j, con_8k, con_8l]

# MODELS #
OLMP_model = Model(
    m,
    name="OLMP",
    description="Outer-loop subproblem",
    equations=olmp_eqns,
    problem='MIP',
    sense='min',
    objective=min_inv_cost_wc,
)

# Test solve for the outer loop master problem
summary = OLMP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_debug.txt"), output=sys.stdout)
#redirect output to a file
print("Objective Function Value:  ", round(OLMP_model.objective_value, 3))
print(vL_ly.records)
m.write(r'C:\Users\Kevin\OneDrive - McGill University\Research\Sandbox\optimization\multi-year_AROTNEP\results\aro_tnep_results.gdx')


# SOLUTION PROCEDURE #
lbo = -999999999999
ubo = 999999999999
ol_error = (ubo - lbo) / lbo
j = 0
for ol_iter in range(0,5):
    if ol_error < tol:
        pass
    elif ol_error >= tol:
        j += 1