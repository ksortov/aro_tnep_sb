from pandas.core.dtypes.inference import is_re

from input_data_processing import weights, RD1, lines, buses, ESS, CG, RES, loads, years_data, sigma_yt_data, tau_yth_data, gamma_dyth_data, gamma_ryth_data, ES_syt0_data, tol
from gamspy import Alias, Container, Domain, Equation, Model, Options, Ord, Card, Parameter, Set, Smax, Sum, Variable
from gamspy.math import power
from utils import logger
import pandas as pd
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
hm = Set(m, name="hm", records=h.records[1:-1], description="Subset of RTPs indexed by h that excludes the first and last elements")

# Multidimensional sets used to make associations between buses, lines and generating units
d_n = Set(m, name="d_n", domain=[d, n], records=loads[['Load', 'Bus']], description="Set of loads connected to bus n")
g_n = Set(m, name="g_n", domain=[g, n], records=CG[['Generating unit', 'Bus']], description="Set of conventional units connected to bus n")
r_n = Set(m, name="r_n", domain=[r, n], records=RES[['Generating unit', 'Bus']], description="Set of renewable units connected to bus n")
s_n = Set(m, name="s_n", domain=[s, n], records=ESS[['Storage unit', 'Bus']], description="Set of storage units connected to bus n")
rel_n = Set(m, name="rel_n", domain=[l, n], records=lines[['Transmission line', 'To bus']], description="Receiving bus of transmission line l")
sel_n = Set(m, name="sel_n", domain=[l, n], records=lines[['Transmission line', 'From bus']], description="Sending bus of transmission line l")

# Sets of indices for the outer and inner loop problems
j = Set(m, name="j", description="Iteration of the outer loop")
i = Set(m, name="i", domain=[j], description="Subset of iteration of the outer loop for relaxation")
k = Set(m, name="k", description="Iteration of the outer loop")
v = Set(m, name="v", domain=[k], description="Subset of iteration of the inner loop for relaxation")

# ALIAS #
yp = Alias(m, name="yp", alias_with=y)

# PARAMETERS #
# Scalars
GammaD = Parameter(m, name="GammaD", records=14, description="Uncertainty budget for increased loads")
GammaGC = Parameter(m, name="GammaGC", records=8, description="Uncertainty budget for increased CG marginal cost")
GammaGP = Parameter(m, name="GammaGP", records=8, description="Uncertainty budget for decreased CG marginal cost")
GammaRS = Parameter(m, name="GammaRS", records=4, description="Uncertainty budget for decreased solar capacity")
GammaRW = Parameter(m, name="GammaRW", records=4, description="Uncertainty budget for decreased wind capacity")
kappa = Parameter(m, name="kappa", records=0.1, description="Discount rate")
IT = Parameter(m, name="IT", records=400000000, description="Investment budget")
nb_H = Parameter(m, name="nb_H", records=8, description="Number of RTPs of each RD")
FL = Parameter(m, name="FL", records=99, description="Large constant for disjunctive linearization")
FD = Parameter(m, name="FD", records=99, description="Large constant for exact linearization")
FD_up = Parameter(m, name="FD_up", records=99, description="Large constant for exact linearization")
FG_up = Parameter(m, name="FG_up", records=99, description="Large constant for exact linearization")
FR_up = Parameter(m, name="FR_up", records=99, description="Large constant for exact linearization")

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
CG_gyi = Parameter(m, name='CG_gyi', domain=[g, y, i], description="Worst-case realization of the marginal production cost of conventional generating unit g for relaxed outer loop iteration i")
PD_dyi = Parameter(m, name='PD_dyi', domain=[d, y, i], description="Worst-case realization of the peak power consumption of load d for relaxed outer loop iteration i")
PG_gyi = Parameter(m, name='PG_gyi', domain=[g, y, i], description="Worst-case realization of the capacity of conventional generating unit g for relaxed outer loop iteration i")
PR_ryi = Parameter(m, name='PR_ryi', domain=[r, y, i], description="Worst-case realization of the capacity of renewable generating unit r for relaxed outer loop iteration i")
CG_gyk = Parameter(m, name='CG_gyk', domain=[g, y], description="Worst-case realization of the marginal production cost of conventional generating unit g for inner loop iteration k")
PD_dyk = Parameter(m, name='PD_dyk', domain=[d, y], description="Worst-case realization of the peak power consumption of load d for inner loop iteration k")
PG_gyk = Parameter(m, name='PG_gyk', domain=[g, y], description="Worst-case realization of the capacity of conventional generating unit g for inner loop iteration k")
PR_ryk = Parameter(m, name='PR_ryk', domain=[r, y], description="Worst-case realization of the capacity of renewable generating unit r for inner loop iteration k")
CG_gyo = Parameter(m, name='CG_gyo', domain=[g, y], description="Worst-case realization of the marginal production cost of conventional generating unit g for ADA iteration o")
PD_dyo = Parameter(m, name='PD_dyo', domain=[d, y], description="Worst-case realization of the peak power consumption of load d for ADA iteration o")
PG_gyo = Parameter(m, name='PG_gyo', domain=[g, y], description="Worst-case realization of the capacity of conventional generating unit g for ADA iteration o")
PR_ryo = Parameter(m, name='PR_ryo', domain=[r, y], description="Worst-case realization of the capacity of renewable generating unit r for ADA iteration o")
LambdaN_nythvo = Parameter(m, name='LambdaN_nythvo', domain=[n, y, t, h, v], description="Dual variable associated with the power balance equation at bus n for ADA iteration o")
muD_dythvo_up = Parameter(m, name='muD_dythvo_up', domain=[d, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the unserved demand of load d for ADA iteration o")
muG_gythvo_lo = Parameter(m, name='muG_gythvo_lo', domain=[g, y, t, h, v], description="Dual variable associated with the constraint imposing the lower bound for the power produced by conventional generating unit g for ADA iteration o")
muG_gythvo_up = Parameter(m, name='muG_gythvo_up', domain=[g, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the power produced by conventional generating unit g for ADA iteration o")
muGD_gythvo = Parameter(m, name='muGD_gythvo', domain=[g, y, t, h, v], description="Dual variable associated with the constraint imposing the ramp-down limit of conventional generating unit g, being ℎ greater than 1 for ADA iteration o")
muGU_gythvo = Parameter(m, name='muGU_gythvo', domain=[g, y, t, h, v], description="Dual variable associated with the constraint imposing the ramp-up limit of conventional generating unit g, being ℎ greater than 1 for ADA iteration o")
muL_lythvo_lo = Parameter(m, name='muL_lythvo_lo', domain=[l, y, t, h, v], description="Dual variable associated with the constraint imposing the lower bound for the power flow through transmission line l for ADA iteration o")
muL_lythvo_up = Parameter(m, name='muL_lythvo_up', domain=[l, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the power flow through transmission line l for ADA iteration o")
muR_rythvo_up = Parameter(m, name='muR_rythvo_up', domain=[r, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the power produced by renewable generating unit r for ADA iteration o")
muS_sythvo_lo = Parameter(m, name='muS_sythvo_lo', domain=[s, y, t, h, v], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s for ADA iteration o")
muS_sythvo_up = Parameter(m, name='muS_sythvo_up', domain=[s, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the energy stored in storage facility s for ADA iteration o")
muSC_sythvo_up = Parameter(m, name='muSC_sythvo_up', domain=[s, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the charging power of storage facility s for ADA iteration o")
muSD_sythvo_up = Parameter(m, name='muSD_sythvo_up', domain=[s, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the discharging power of storage facility s for ADA iteration o")
PhiS_sytvo = Parameter(m, name='PhiS_sytvo', domain=[s, y, t, v], description="Dual variable associated with the energy stored in storage facility s at the first RTP of RD for ADA iteration o")
PhiS_sytvo_lo = Parameter(m, name='PhiS_sytvo_lo', domain=[s, y, t, v], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s at the last RTP of RD for ADA iteration o")
VL_lyj = Parameter(m, name='VL_lyj', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y, which is otherwise 0, for outer loop iteration j")
VL_lyj_prev = Parameter(m, name='VL_lyj_prev', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y or in previous years, which is otherwise 0, for outer loop iteration j")
UG_gythv = Parameter(m, name='UG_gythv', domain=[g, y, t, h, v], description="Binary variable used to model the commitment status of conventional unit g, for relaxed inner loop iteration v")
US_sythv = Parameter(m, name='US_sythv', domain=[s, y, t, h, v], description="Binary variable used to used to avoid the simultaneous charging and discharging of storage facility s, for relaxed inner loop iteration v")

# VARIABLES #
# Optimization variables
theta_nythi = Variable(m, name="theta_nythi", domain=[n, y, t, h, i], description="Voltage angle at bus n")
xi_y = Variable(m, name='xi_y', domain=[y], description="Auxiliary variable of the inner-loop master problem")
xi = Variable(m, name='xi', description="Auxiliary variable of the inner-loop master problem")
xiP_y = Variable(m, name='xiP_y', domain=[y], description="Auxiliary variable of the first problem solved at each iteration of the ADA when it is applied to the inner-loop master problem")
xiP = Variable(m, name='xiP', description="Auxiliary variable of the first problem solved at each iteration of the ADA when it is applied to the inner-loop master problem")
xiQ_y = Variable(m, name='xiQ_y', domain=[y], description="Auxiliary variable of the second problem solved at each iteration of the ADA when it is applied to the inner-loop master problem")
xiQ = Variable(m, name='xiQ', description="Auxiliary variable of the second problem solved at each iteration of the ADA when it is applied to the inner-loop master problem")
rho_y = Variable(m, name='rho_y', domain=[y], description="Auxiliary variable of the outer-loop master problem")
aD_dy = Variable(m, type='positive', name='aD_dy', domain=[d, y], description="Continuous variable associated with the deviation that the peak power consumption of load d can experience from its forecast value in year y")
aGC_gy = Variable(m, type='positive', name='aGC_gy', domain=[g, y], description="Continuous variable associated with the deviation that the marginal production cost of conventional generating unit g can experience from its forecast value in year y")
aGP_gy = Variable(m, type='positive', name='aGP_gy', domain=[g, y], description="Continuous variable associated with the deviation that the capacity of conventional generating unit g can experience from its forecast value in year y")
aR_ry = Variable(m, type='positive', name='aR_ry', domain=[r, y], description="Continuous variable associated with the deviation that the capacity of renewable generating unit r can experience from its forecast value in year y")
cG_gy = Variable(m, name='cG_gy', domain=[g, y], description="Worst-case realization of the marginal production cost of conventional generating unit g")
cO_y = Variable(m, name='cO_y', domain=[y], description="Operating costs")
cOWC_y = Variable(m, name='cOWC_y', domain=[y], description="Worst case operating costs")
eS_sythi = Variable(m, name='eS_sythi', domain=[s, y, t, h, i], description="Energy stored in storage facility s")
pD_dy = Variable(m, name='pD_dy', domain=[d, y], description="Worst-case realization of the peak power consumption of load d")
pG_gythi = Variable(m, name='pG_gythi', domain=[g, y, t, h, i], description="Power produced by conventional generating unit g")
pG_gy = Variable(m, name='pG_gy', domain=[g, y], description="Worst-case realization of the capacity of conventional generating unit g")
pL_lythi = Variable(m, name='pL_lythi', domain=[l, y, t, h, i], description="Power flow through transmission line l")
pLS_dythi = Variable(m, type='positive', name='pLS_dythi', domain=[d, y, t, h, i], description="Unserved demand of load d")
pR_rythi = Variable(m, type='positive', name='pR_rythi', domain=[r, y, t, h, i], description="Power produced by renewable generating unit r")
pR_ry = Variable(m, name='pR_ry', domain=[r, y], description="Worst-case realization of the capacity of renewable generating unit r")
pSC_sythi = Variable(m, type='positive', name='pSC_sythi', domain=[s, y, t, h, i], description="Charging power of storage facility s")
pSD_sythi = Variable(m, type='positive', name='pSD_sythi', domain=[s, y, t, h, i], description="Discharging power of storage facility s")
uG_gyth = Variable(m, name='uG_gyth', type='binary', domain=[g, y, t, h], description="Binary variable used to model the commitment status of conventional unit g")
uS_syth = Variable(m, name='uS_syth', type='binary', domain=[s, y, t, h], description="Binary variable used to used to avoid the simultaneous charging and discharging of storage facility s")
vL_ly = Variable(m, name='vL_ly', type='binary', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y, which is otherwise 0")
vL_ly_prev = Variable(m, name='vL_ly_prev', type='binary', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y or in previous years, which is otherwise 0")
zD_dy = Variable(m, name='zD_dy', type='binary', domain=[d, y], description="Binary variable that is equal to 1 if the worst-case realization of the peak power consumption of load 𝑑 is equal to its upper bound, which is otherwise 0")
zGC_gy = Variable(m, name='zGC_gy', type='binary', domain=[g, y], description="Binary variable that is equal to 1 if the worst-case realization of the marginal production cost of conventional generating unit g is equal to its upper bound, which is otherwise 0")
zGP_gy = Variable(m, name='zGP_gy', type='binary', domain=[g, y], description="Binary variable that is equal to 1 if the worst-case realization of the capacity of conventional generating unit g is equal to its upper bound, which is otherwise 0")
zR_ry = Variable(m, name='zR_ry', type='binary', domain=[r, y], description="Binary variable that is equal to 1 if the worst-case realization of he capacity of renewable generating unit r is equal to its upper bound, which is otherwise 0")

# Dual variables
lambdaN_nythv = Variable(m, name='lambdaN_nythv', domain=[n, y, t, h, v], description="Dual variable associated with the power balance equation at bus n")
muD_dythv_up = Variable(m, name='muD_dythv_up', type='positive', domain=[d, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the unserved demand of load d")
muG_gythv_lo = Variable(m, name='muG_gythv_lo', type='positive', domain=[g, y, t, h, v], description="Dual variable associated with the constraint imposing the lower bound for the power produced by conventional generating unit g")
muG_gythv_up = Variable(m, name='muG_gythv_up', type='positive', domain=[g, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the power produced by conventional generating unit g")
muGD_gythv = Variable(m, name='muGD_gythv', type='positive', domain=[g, y, t, h, v], description="Dual variable associated with the constraint imposing the ramp-down limit of conventional generating unit g, being ℎ greater than 1")
muGU_gythv = Variable(m, name='muGU_gythv', type='positive', domain=[g, y, t, h, v], description="Dual variable associated with the constraint imposing the ramp-up limit of conventional generating unit g, being ℎ greater than 1")
muL_lythv_exist = Variable(m, name='muL_lythv_exist', domain=[le, y, t, h, v], description="Dual variable associated with the power flow through existing transmission line l")
muL_lythv_can = Variable(m, name='muL_lythv_can', domain=[lc, y, t, h, v], description="Dual variable associated with the power flow through candidate transmission line l")
muL_lythv_lo = Variable(m, name='muL_lythv_lo', type='positive', domain=[l, y, t, h, v], description="Dual variable associated with the constraint imposing the lower bound for the power flow through transmission line l")
muL_lythv_up = Variable(m, name='muL_lythv_up', type='positive', domain=[l, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the power flow through transmission line l")
muR_rythv_up = Variable(m, name='muR_rythv_up', type='positive', domain=[r, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the power produced by renewable generating unit r")
muS_sythv = Variable(m, name='muS_sythv', domain=[s, y, t, h, v], description="Dual variable associated with the energy stored in storage facility s, being h greater than 1")
muS_sythv_lo = Variable(m, name='muS_sythv_lo', type='positive', domain=[s, y, t, h, v], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s")
muS_sythv_up = Variable(m, name='muS_sythv_up', type='positive', domain=[s, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the energy stored in storage facility s")
muSC_sythv_up = Variable(m, name='muSC_sythv_up', type='positive', domain=[s, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the charging power of storage facility s")
muSD_sythv_up = Variable(m, name='muSD_sythv_up', type='positive', domain=[s, y, t, h, v], description="Dual variable associated with the constraint imposing the upper bound for the discharging power of storage facility s")
PhiS_sytv = Variable(m, name='PhiS_sytv', type='positive', domain=[s, y, t, v], description="Dual variable associated with the energy stored in storage facility s at the first RTP of RD") # Check if it should really be positive
PhiS_sytv_lo = Variable(m, name='PhiS_sytv_lo', type='positive', domain=[s, y, t, v], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s at the last RTP of RD")
phiN_nythv = Variable(m, name='phiN_nythv', domain=[n, y, t, h, v], description="Dual variable associated with the definition of the reference bus n")

# Linearization variables
alphaD_dyth = Variable(m, name="alphaD_dyth", domain=[d, y, t, h], description="Auxiliary variable for the linearization of zD_dy*lambdaN_nyth")
alphaD_dyth_up = Variable(m, name="alphaD_dyth_up", type='positive', domain=[d, y, t, h], description="Auxiliary variable for the linearization of zD_dy*muD_dyth_up")
alphaGP_gyth_up = Variable(m, name="alphaGP_gyth_up", type='positive', domain=[g, y, t, h], description="Auxiliary variable for the linearization of zGP_gy*muGP_gyth_up")
alphaR_ryth_up = Variable(m, name="alphaR_ryth_up", type='positive', domain=[r, y, t, h], description="Auxiliary variable for the linearization of zR_ry*muR_ryth_up")

min_inv_cost_wc = Variable(m, name="min_inv_cost_wc", description="Worst-case investment costs")
min_op_cost_y = Variable(m, name="max_op_cost_wc", description="Minimized operating costs for year y")

# EQUATIONS #
# Outer-loop master problem OF and constraints
OF_olmp = Equation(m, name="OF_olmp", type="regular")
con_1c = Equation(m, name="con_1c")
con_1d = Equation(m, name="con_1d", domain=[lc])
con_1e = Equation(m, name="con_1e", domain=[lc, y])

con_4c = Equation(m, name="con_4c", domain=[y, i])
con_4d = Equation(m, name="con_4d", domain=[n, y, t, h, i])
con_4e = Equation(m, name="con_4e", domain=[le, y, t, h, i])
con_4f_lin1 = Equation(m, name="con_4f_lin1", domain=[lc, y, t, h, i]) # Linearized
con_4f_lin2 = Equation(m, name="con_4f_lin2", domain=[lc, y, t, h, i]) # Linearized
con_4g_exist_lin1 = Equation(m, name="con_4g_exist_lin1", domain=[le, y, t, h, i])
con_4g_exist_lin2 = Equation(m, name="con_4g_exist_lin2", domain=[le, y, t, h, i])
con_4g_can_lin1 = Equation(m, name="con_4g_can_lin1", domain=[lc, y, t, h, i])
con_4g_can_lin2 = Equation(m, name="con_4g_can_lin2", domain=[lc, y, t, h, i])
con_4h = Equation(m, name="con_4h", domain=[s, y, t, i]) # H == 1
con_4i = Equation(m, name="con_4i", domain=[s, y, t, h, i]) # H =/= 1
con_4j = Equation(m, name="con_4j", domain=[s, y, t, h, i]) # H == Hmax
con_4k1 = Equation(m, name="con_4k1", domain=[s, y, t, h, i])
con_4k2 = Equation(m, name="con_4k2", domain=[s, y, t, h, i])
# con_4l = Equation(m, name="con_4l", domain=[S,T,H,Y])
con_4m1 = Equation(m, name="con_4m1", domain=[s, y, t, h, i])
con_4m2 = Equation(m, name="con_4m2", domain=[s, y, t, h, i])
con_4n1 = Equation(m, name="con_4n1", domain=[s, y, t, h, i])
con_4n2 = Equation(m, name="con_4n2", domain=[s, y, t, h, i])
con_4o1 = Equation(m, name="con_4o1", domain=[d, y, t, h, i])
con_4o2 = Equation(m, name="con_4o2", domain=[d, y, t, h, i])
# con_4p = Equation(m, name="con_4p", domain=[G,T,H,Y])
con_4q1 = Equation(m, name="con_4q1", domain=[g, y, t, h, i])
con_4q2 = Equation(m, name="con_4q2", domain=[g, y, t, h,i])
con_4r1 = Equation(m, name="con_4r1", domain=[g, y, t, h, i]) # H =/= 1
con_4r2 = Equation(m, name="con_4r2", domain=[g, y, t, h, i]) # H =/= 1
con_4s1 = Equation(m, name="con_4s1", domain=[r, y, t, h, i])
con_4s2 = Equation(m, name="con_4s2", domain=[r, y, t, h, i])
con_4t = Equation(m, name="con_4t", domain=[n, y, t, h, i]) # N == ref bus


def build_olmp_eqns():
    OF_olmp[...] = min_inv_cost_wc == Sum(y, rho_y[y] / power(1.0 + kappa, y.val) + \
                                          (1.0 / power(1.0 + kappa, y.val - 1)) * Sum(lc, IL_l[lc] * vL_ly[lc, y]))
    con_1c[...] = Sum(lc, Sum(y, (1.0 / power(1.0 + kappa, y.val - 1)) * IL_l[lc] * vL_ly[lc, y])) <= IT
    con_1d[lc] = Sum(y, vL_ly[lc, y]) <= 1
    con_1e[lc, y] = vL_ly_prev[lc, y] == Sum(yp.where[yp.val <= y.val], vL_ly[lc, yp])

    con_4c[y, i] = rho_y[y] >= Sum(t, sigma_yt[y, t] * Sum(h, tau_yth[y, t, h] * (Sum(g, CG_gyi[g,y,i] * pG_gythi[g, y, t, h, i]) \
    + Sum(r, CR_r[r] * (gammaR_ryth[r, y, t, h] * PR_ryi[r,y,i] - pR_rythi[r, y, t, h, i])) \
    + Sum(d, CLS_d[d] * pLS_dythi[d, y, t, h, i]))))
    con_4d[n, y, t, h, i] = Sum(g.where[g_n[g, n]], pG_gythi[g, y, t, h, i]) + Sum(r.where[r_n[r, n]], pR_rythi[r, y, t, h, i]) \
    + Sum(l.where[sel_n[l, n]], pL_lythi[l, y, t, h, i]) - Sum(l.where[rel_n[l, n]], pL_lythi[l, y, t, h, i]) \
    + Sum(s.where[s_n[s, n]], pSD_sythi[s, y, t, h, i] - pSC_sythi[s, y, t, h, i]) \
    == Sum(d.where[d_n[d, n]], gammaD_dyth[d, y, t, h] * PD_dyi[d,y,i] - pLS_dythi[d, y, t, h, i])
    con_4e[le, y, t, h, i] = pL_lythi[le, y, t, h, i] == (1.0 / X_l[le]) * (Sum(n.where[sel_n[le, n]], theta_nythi[n, y, t, h, i]) \
    - Sum(n.where[rel_n[le, n]], theta_nythi[n, y, t, h, i]))
    con_4f_lin1[lc, y, t, h, i] = pL_lythi[lc, y, t, h, i] - (1 / X_l[lc]) * (Sum(n.where[sel_n[lc, n]], theta_nythi[n, y, t, h, i]) \
    - Sum(n.where[rel_n[lc, n]], theta_nythi[n, y, t, h, i])) <= (1 - vL_ly_prev[lc, y]) * FL
    con_4f_lin2[lc, y, t, h, i] = pL_lythi[lc, y, t, h, i] - (1 / X_l[lc]) * (Sum(n.where[sel_n[lc, n]], theta_nythi[n, y, t, h, i]) \
    - Sum(n.where[rel_n[lc, n]], theta_nythi[n, y, t, h, i])) >= -(1 - vL_ly_prev[lc, y]) * FL
    con_4g_exist_lin1[le, y, t, h, i] = pL_lythi[le, y, t, h, i] <= PL_l[le]
    con_4g_exist_lin2[le, y, t, h, i] = pL_lythi[le, y, t, h, i] >= -PL_l[le]
    con_4g_can_lin1[lc, y, t, h, i] = pL_lythi[lc, y, t, h, i] <= vL_ly_prev[lc, y] * PL_l[lc]
    con_4g_can_lin2[lc, y, t, h, i] = pL_lythi[lc, y, t, h, i] >= -vL_ly_prev[lc, y] * PL_l[lc]
    con_4h[s, y, t, i] = eS_sythi[s, y, t, 1, i] == ES_syt0[s, y, t] + (pSC_sythi[s, y, t, 1, i] * etaSC_s[s] \
    - (pSD_sythi[s, y, t, 1, i] / etaSD_s[s])) * tau_yth[y, t, 1]
    con_4i[s, y, t, h, i].where[Ord(h) > 1] = eS_sythi[s, y, t, h, i] == eS_sythi[s, y, t, h.lag(1), i] \
    + (pSC_sythi[s, y, t, h, i] * etaSC_s[s] - (pSD_sythi[s, y, t, h, i] / etaSD_s[s])) * tau_yth[y, t, h]
    con_4j[s, y, t, h, i].where[Ord(h) == Card(h)] = ES_syt0[s, y, t] <= eS_sythi[s, y, t, h, i]
    con_4k1[s, y, t, h, i] = eS_sythi[s, y, t, h, i] <= ES_s_max[s]
    con_4k2[s, y, t, h, i] = eS_sythi[s, y, t, h, i] >= ES_s_min[s]
    con_4m1[s, y, t, h, i] = pSC_sythi[s, y, t, h, i] <= uS_syth[s, y, t, h] * PSC_s[s]
    con_4m2[s, y, t, h, i] = pSC_sythi[s, y, t, h, i] >= 0
    con_4n1[s, y, t, h, i] = pSD_sythi[s, y, t, h, i] <= (1 - uS_syth[s, y, t, h]) * PSD_s[s]
    con_4n2[s, y, t, h, i] = pSD_sythi[s, y, t, h, i] >= 0
    con_4o1[d, y, t, h, i] = pLS_dythi[d, y, t, h, i] <= gammaD_dyth[d, y, t, h] * PD_dyi[d,y,i]  # pD_dy[D,Y]
    con_4o2[d, y, t, h, i] = pLS_dythi[d, y, t, h, i] >= 0
    con_4q1[g, y, t, h, i] = pG_gythi[g, y, t, h, i] <= uG_gyth[g, y, t, h] * PG_gyi[g,y,i]  # pG_gy[G,Y]
    con_4q2[g, y, t, h, i] = pG_gythi[g, y, t, h, i] >= uG_gyth[g, y, t, h] * PG_g_min[g]
    con_4r1[g, y, t, h, i].where[Ord(h) > 1] = pG_gythi[g, y, t, h, i] - pG_gythi[g, y, t, h.lag(1), i] <= RGU_g[g]
    con_4r2[g, y, t, h, i].where[Ord(h) > 1] = pG_gythi[g, y, t, h, i] - pG_gythi[g, y, t, h.lag(1), i] >= -RGD_g[g]
    con_4s1[r, y, t, h, i] = pR_rythi[r, y, t, h, i] <= gammaR_ryth[r, y, t, h] * PR_ryi[r,y,i]  # pR_ry[R,Y]
    con_4s2[r, y, t, h, i] = pR_rythi[r, y, t, h, i] >= 0
    con_4t[n, y, t, h, i].where[Ord(n) == 1] = theta_nythi[n, y, t, h, i] == 0


olmp_eqns = [OF_olmp, con_1c, con_1d, con_1e, con_4c, con_4d, con_4e, con_4f_lin1, con_4f_lin2, con_4g_exist_lin1,
             con_4g_exist_lin2, con_4g_can_lin1, con_4g_can_lin2, con_4h, con_4i, con_4j, con_4k1, con_4k2, con_4m1,
             con_4m2, con_4n1, con_4n2, con_4o1, con_4o2, con_4q1, con_4q2, con_4r1, con_4r2, con_4s1, con_4s2, con_4t]

## Outer-loop subproblem
# Inner-loop master problem OF and constraints
# OF_ilmp = Equation(m, name="OF_ilmp", type="regular")
con_2b = Equation(m, name="con_2b", domain=[g])
con_2c = Equation(m, name="con_2c", domain=[d])
con_2d = Equation(m, name="con_2d", domain=[g])
con_2e = Equation(m, name="con_2e", domain=[r])
# con_2f, con_2g, con_2h, con_2i set z variables to binary type
con_2j = Equation(m, name="con_2j")
con_2k = Equation(m, name="con_2k")
con_2l = Equation(m, name="con_2l")
con_2m = Equation(m, name="con_2m")
con_2n = Equation(m, name="con_2n")

# con_5c = Equation(m, name="con_5c")
# con_5c[...] = xi_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h] * pD_dy[d,yi] * Sum(n.where[d_n[d,n]], lambdaN_nyth[n,yi,t,h])) - \
#                           Sum(l, PL_l[l] * (muL_lyth_lo[l,yi,t,h] + muL_lyth_up[l,yi,t,h])) - \
#                           Sum(s, uS_syth[s,yi,t,h] * PSC_s[s] * muSC_syth_up[s,yi,t,h] + (1 - uS_syth[s,yi,t,h]) * PSD_s[s] * muSD_syth_up[s,yi,t,h] - ES_s_min[s] * muS_syth_lo[s,yi,t,h] + ES_s_max[s] * muS_syth_up[s,yi,t,h]) + \
#                           Sum(g, uG_gyth[g,yi,t,h] * (PG_g_min[g] * muG_gyth_lo[g,yi,t,h] - pG_gy[g,yi] * muG_gyth_up[g,yi,t,h])) - \
#                           Sum(r, gammaR_ryth[r,yi,t,h] * pR_ry[r,yi] * (muR_ryth_up[r,yi,t,h] - sigma_yt[yi,t] * tau_yth[yi,t,h] * CR_r[r])) - \
#                           Sum(d, gammaD_dyth[d,yi,t,h] * pD_dy[d,yi] * muD_dyth_up[d,yi,t,h])) + \
#                           Sum(s, ES_syt0[s,yi,t] * (PhiS_syt[s,yi,t] + PhiS_syt_lo[s,yi,t])) - \
#                           Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g] * muGD_gyth[g,yi,t,h] + RGU_g[g] * muGU_gyth[g,yi,t,h])))
con_5c_lin_a = Equation(m, name="con_5c_lin_a", domain=[v])
con_5c_lin_b1 = Equation(m, name="con_5c_lin_b1", domain=[d, t, h])
con_5c_lin_b2 = Equation(m, name="con_5c_lin_b2", domain=[d, t, h])
con_5c_lin_c1 = Equation(m, name="con_5c_lin_c1", domain=[d, t, h, v])
con_5c_lin_c2 = Equation(m, name="con_5c_lin_c2", domain=[d, t, h, v])
con_5c_lin_d = Equation(m, name="con_5c_lin_d", domain=[d, t, h])
con_5c_lin_e1 = Equation(m, name="con_5c_lin_e1", domain=[d, t, h, v])
con_5c_lin_e2 = Equation(m, name="con_5c_lin_e2", domain=[d, t, h, v])
con_5c_lin_f = Equation(m, name="con_5c_lin_f", domain=[g, t, h])
con_5c_lin_g1 = Equation(m, name="con_5c_lin_g1", domain=[g, t, h, v])
con_5c_lin_g2 = Equation(m, name="con_5c_lin_g2", domain=[g, t, h, v])
con_5c_lin_h = Equation(m, name="con_5c_lin_h", domain=[r, t, h])
con_5c_lin_i1 = Equation(m, name="con_5c_lin_i1", domain=[r, t, h, v])
con_5c_lin_i2 = Equation(m, name="con_5c_lin_i2", domain=[r, t, h, v])

con_5d = Equation(m, name="con_5d", domain=[g, t, v])
con_5e = Equation(m, name="con_5e", domain=[g, t, h, v])
con_5f = Equation(m, name="con_5f", domain=[g, t, v])
con_5g = Equation(m, name="con_5g", domain=[d, t, h, v])
con_5h = Equation(m, name="con_5h", domain=[r, t, h, v])
con_5i = Equation(m, name="con_5i", domain=[le, t, h, v])
con_5j = Equation(m, name="con_5j", domain=[lc, t, h, v])
con_5k = Equation(m, name="con_5k", domain=[s, t, v])
con_5l = Equation(m, name="con_5l", domain=[s, t, h, v])
con_5m = Equation(m, name="con_5m", domain=[s, t, v])
con_5n = Equation(m, name="con_5n", domain=[s, t, h, v])
con_5o = Equation(m, name="con_5o", domain=[n,t,h,v]) # N =/= ref bus
con_5p = Equation(m, name="con_5p", domain=[n,t,h,v]) # N == ref bus
con_5q = Equation(m, name="con_5q", domain=[s,t,v])
con_5r = Equation(m, name="con_5r", domain=[s,t,h,v])
con_5s = Equation(m, name="con_5s", domain=[s,t,v])
ilmp_obj_var = Equation(m, name="ilmp_obj_var")

def build_ilmp_eqns(yi):
    con_2b[g] = cG_gy[g, yi] == CG_g_fc[g] * power(1 + zetaGC_g_fc[g], yi - 1) + CG_g_max[g] * power(1 + zetaGC_g_max[g], yi - 1) * zGC_gy[g, yi]
    con_2c[d] = pD_dy[d, yi] == PD_d_fc[d] * power(1 + zetaD_d_fc[d], yi - 1) + PD_d_max[d] * power(1 + zetaD_d_max[d], yi - 1) * zD_dy[d, yi]
    con_2d[g] = pG_gy[g, yi] == PG_g_fc[g] * power(1 + zetaGP_g_fc[g], yi - 1) - PG_g_max[g] * power(1 + zetaGP_g_max[g], yi - 1) * zGP_gy[g, yi]
    con_2e[r] = pR_ry[r, yi] == PR_r_fc[r] * power(1 + zetaR_r_fc[r], yi - 1) - PR_r_max[r] * power(1 + zetaR_r_max[r], yi - 1) * zR_ry[r, yi]
    con_2j[...] = Sum(g, zGC_gy[g, yi]) <= GammaGC
    con_2k[...] = Sum(d, zD_dy[d, yi]) <= GammaD
    con_2l[...] = Sum(g, zGP_gy[g, yi]) <= GammaGP
    con_2m[...] = Sum(rs, zR_ry[rs, yi]) <= GammaRS
    con_2n[...] = Sum(rw, zR_ry[rw, yi]) <= GammaRW

    ilmp_obj_var[...] = xi_y[yi] == xi
    con_5c_lin_a[v] = xi <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h] * (PD_d_fc[d]*power(1+zetaD_d_fc[d], yi-1)\
    * Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,v]) + PD_d_max[d]*power(1+zetaD_d_max[d], yi-1) * alphaD_dyth[d,yi,t,h]))\
    - Sum(l, PL_l[l]*(muL_lythv_lo[l,yi,t,h,v] + muL_lythv_up[l,yi,t,h,v])) \
    - Sum(s, US_sythv[s,yi,t,h,v]*PSC_s[s]*muSC_sythv_up[s,yi,t,h,v] + (1-US_sythv[s,yi,t,h,v])*PSD_s[s]*muSD_sythv_up[s,yi,t,h,v] - ES_s_min[s]*muS_sythv_lo[s,yi,t,h,v] + ES_s_max[s]*muS_sythv_up[s,yi,t,h,v])\
    + Sum(g, UG_gythv[g,yi,t,h,v]*(PG_g_min[g]*muG_gythv_lo[g,yi,t,h,v] - (PG_g_fc[g]*power(1-zetaGP_g_fc[g], yi-1)*muG_gythv_up[g,yi,t,h,v] - PG_g_max[g]*power(1+zetaGP_g_max[g], yi-1)*alphaGP_gyth_up[g,yi,t,h]))) \
    - Sum(r, gammaR_ryth[r,yi,t,h]*(PR_r_fc[r]*power(1+zetaR_r_fc, yi-1)*muR_rythv_up[r,yi,t,h,v] - PR_r_max[r]*power(1+zetaR_r_max, yi-1)*alphaR_ryth_up[r,yi,t,h]))\
    + Sum(r, sigma_yt[yi,t]*tau_yth[yi,t,h]*CR_r[r]*gammaR_ryth[r,yi,t,h]*(PR_r_fc[r]*power(1+zetaR_r_fc, yi-1) - PR_r_max[r]*power(1+zetaR_r_max, yi-1) * zR_ry[r,yi]))\
    - Sum(d, gammaD_dyth[d,yi,t,h]*(PD_d_fc[d]*power(1+zetaD_d_fc[d], yi-1)*muD_dythv_up[d,yi,t,h,v] + PD_d_max[d]*power(1+zetaD_d_max[d], yi-1)*alphaD_dyth_up[d,yi,t,h]))) \
    + Sum(s, ES_syt0[s,yi,t]*(PhiS_sytv[s,yi,t,v] + PhiS_sytv_lo[s,yi,t,v])) \
    - Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g]*muGD_gythv[g,yi,t,h,v] + RGU_g[g]*muGU_gythv[g,yi,t,h,v])))
    con_5c_lin_b1[d, t, h] = alphaD_dyth[d, yi, t, h] <= zD_dy[d, yi] * FD
    con_5c_lin_b2[d, t, h] = alphaD_dyth[d, yi, t, h] >= -zD_dy[d, yi] * FD
    con_5c_lin_c1[d,t,h,v] = Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,v]) - alphaD_dyth[d,yi,t,h] <= (1-zD_dy[d,yi])*FD
    con_5c_lin_c2[d,t,h,v] = Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,v]) - alphaD_dyth[d,yi,t,h] >= -(1-zD_dy[d,yi])*FD
    con_5c_lin_d[d, t, h] = alphaD_dyth_up[d, yi, t, h] <= zD_dy[d, yi] * FD_up
    con_5c_lin_e1[d, t, h, v] = muD_dythv_up[d, yi, t, h, v] - alphaD_dyth_up[d, yi, t, h] <= (1 - zD_dy[d, yi]) * FD_up
    con_5c_lin_e2[d, t, h, v] = muD_dythv_up[d, yi, t, h, v] - alphaD_dyth_up[d, yi, t, h] >= 0
    con_5c_lin_f[g, t, h] = alphaGP_gyth_up[g, yi, t, h] <= zGP_gy[g, yi] * FG_up
    con_5c_lin_g1[g,t,h,v] = muG_gythv_up[g,yi,t,h,v] - alphaGP_gyth_up[g,yi,t,h] <= (1-zGP_gy[g,yi])*FG_up
    con_5c_lin_g2[g,t,h,v] = muG_gythv_up[g,yi,t,h,v] - alphaGP_gyth_up[g,yi,t,h] >= 0
    con_5c_lin_h[r, t, h] = alphaR_ryth_up[r, yi, t, h] <= zR_ry[r, yi] * FR_up
    con_5c_lin_i1[r, t, h, v] = muR_rythv_up[r, yi, t, h, v] - alphaR_ryth_up[r, yi, t, h] <= (1 - zR_ry[r, yi]) * FR_up
    con_5c_lin_i2[r, t, h, v] = muR_rythv_up[r, yi, t, h, v] - alphaR_ryth_up[r, yi, t, h] >= 0

    con_5d[g,t,v] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,1,v]) + muG_gythv_lo[g,yi,t,1,v]\
    - muG_gythv_up[g,yi,t,1,v] - muGD_gythv[g,yi,t,2,v] + muGU_gythv[g,yi,t,2,v]\
    == sigma_yt[yi,t] * tau_yth[yi, t, 1] * cG_gy[g, yi]
    con_5e[g,t,h,v].where[(Ord(h)!=1) & (Ord(h)!=Card(h))] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,h,v])\
    + muG_gythv_lo[g,yi,t,h,v]- muG_gythv_up[g,yi,t,h,v] + muGD_gythv[g,yi,t,h,v] - muGD_gythv[g,yi,t,h.lead(1),v]\
    - muGU_gythv[g,yi,t,h,v] + muGU_gythv[g,yi,t,h.lead(1),v] == sigma_yt[yi,t]*tau_yth[yi,t,h]*cG_gy[g, yi]
    con_5f[g,t,v] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,8,v]) +  muG_gythv_lo[g,yi,t,8,v]\
    - muG_gythv_up[g,yi,t,8,v] + muGD_gythv[g,yi,t,8,v] - muGU_gythv[g, yi, t, 8, v]\
    == sigma_yt[yi, t] * tau_yth[yi, t, 8] * cG_gy[g, yi]
    con_5g[d,t,h,v] = Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,v])-muD_dythv_up[d, yi, t, h, v]\
    <= sigma_yt[yi, t] * tau_yth[yi, t, h] * CLS_d[d]
    con_5h[r,t,h,v] = Sum(n.where[r_n[r,n]], lambdaN_nythv[n,yi,t,h,v]) - muR_rythv_up[r,yi,t,h,v]\
    <= sigma_yt[yi, t] * tau_yth[yi, t, h] * CR_r[r]
    con_5i[le,t,h,v] = Sum(n.where[rel_n[le,n]], lambdaN_nythv[n,yi,t,h,v]) - Sum(n.where[sel_n[le,n]], lambdaN_nythv[n,yi,t,h,v]) \
    + muL_lythv_exist[le, yi, t, h, v] + muL_lythv_lo[le, yi, t, h, v] - muL_lythv_up[le, yi, t, h, v] == 0
    con_5j[lc,t,h,v] = Sum(n.where[rel_n[lc,n]], lambdaN_nythv[n,yi,t,h,v]) - Sum(n.where[sel_n[lc,n]], lambdaN_nythv[n,yi,t,h,v]) \
    + muL_lythv_can[lc,yi,t,h,v] + muL_lythv_lo[lc,yi,t,h,v] - muL_lythv_up[lc,yi,t,h,v] == 0
    con_5k[s,t,v] = Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,1,v])\
    + (tau_yth[yi,t,1]/etaSD_s[s])*PhiS_sytv[s,yi,t,v] - muSD_sythv_up[s,yi,t,1,v] <= 0
    con_5l[s,t,h,v].where[Ord(h)>1] = Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,h,v])\
    + (tau_yth[yi,t,h]/etaSD_s[s])*muS_sythv[s,yi,t,h,v] - muSD_sythv_up[s,yi,t,h,v] <= 0
    con_5m[s,t,v] = -Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,1,v])\
    - etaSC_s[s]*tau_yth[yi,t,1]*PhiS_sytv[s,yi,t,v] - muSC_sythv_up[s,yi,t,1,v] <= 0
    con_5n[s,t,h,v].where[Ord(h)>1] = -Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,h,v])\
    - etaSC_s[s]*tau_yth[yi,t,h]*muS_sythv[s,yi,t,h,v] - muSC_sythv_up[s,yi,t,h,v] <= 0
    con_5o[n, t, h, v].where[Ord(n) > 1] = -Sum(le.where[sel_n[le, n]], muL_lythv_exist[le, yi, t, h, v] / X_l[le])\
    + Sum(le.where[rel_n[le, n]], muL_lythv_exist[le, yi, t, h, v] / X_l[le])\
    - Sum(lc.where[sel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, v])\
    + Sum(lc.where[rel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, v]) == 0
    con_5p[n, t, h, v].where[Ord(n) == 1] = (-Sum(le.where[sel_n[le, n]], muL_lythv_exist[le, yi, t, h, v] / X_l[le])
    + Sum(le.where[rel_n[le, n]], muL_lythv_exist[le, yi, t, h, v] / X_l[le])\
    - Sum(lc.where[sel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, v])\
    + Sum(lc.where[rel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, v])\
    + phiN_nythv[n, yi, t, h, v] == 0)
    con_5q[s,t,v] = PhiS_sytv[s,yi,t,v] - muS_sythv[s,yi,t,2,v] + muS_sythv_lo[s,yi,t,1,v] - muS_sythv_up[s,yi,t,1,v] == 0
    con_5r[s,t,h,v].where[(Ord(h)!=1) & (Ord(h)!=Card(h))] = muS_sythv[s,yi,t,h,v] - muS_sythv[s,yi,t,h.lead(1),v] + muS_sythv_lo[s,yi,t,h,v] - muS_sythv_up[s,yi,t,h,v] == 0
    con_5s[s,t,v] = muS_sythv[s,yi,t,8,v] + PhiS_sytv_lo[s,yi,t,v] + muS_sythv_lo[s,yi,t,8,v] - muS_sythv_up[s,yi,t,8,v] == 0

ilmp_eqns = [con_2b, con_2c, con_2d, con_2e, con_2j, con_2k, con_2l, con_2m, con_2n, con_5c_lin_a, con_5c_lin_b1,
             con_5c_lin_b2, con_5c_lin_c1, con_5c_lin_c2, con_5c_lin_d, con_5c_lin_e1, con_5c_lin_e2, con_5c_lin_f,
             con_5c_lin_g1, con_5c_lin_g2, con_5c_lin_h, con_5c_lin_i1, con_5c_lin_i2, con_5d, con_5e, con_5f, con_5g,
             con_5h, con_5i, con_5j, con_5k, con_5l, con_5m, con_5n, con_5o, con_5p, con_5q, con_5r, con_5s]

# Inner-loop subproblem OF and constraints
OF_ilsp = Equation(m, name="OF_ilsp", type="regular") # Double-check
con_6b = Equation(m, name="con_6b", domain=[n, t, h])
con_6c = Equation(m, name="con_6c", domain=[lc, t, h])
con_6d = Equation(m, name="con_6d", domain=[d, t, h])
con_6e1 = Equation(m, name="con_6e1", domain=[g, t, h])
con_6e2 = Equation(m, name="con_6e2", domain=[g, t, h])
con_6f = Equation(m, name="con_6f", domain=[r, t, h])
con_3c = Equation(m, name="con_3c", domain=[lc, y, t, h])
con_3e1 = Equation(m, name="con_3e1", domain=[l, y, t, h])
con_3e2 = Equation(m, name="con_3e2", domain=[l, y, t, h])
con_3fg = Equation(m, name="con_3fg", domain=[s, y, t, h])
con_3h = Equation(m, name="con_3h", domain=[s, y, t, h])
con_3i1 = Equation(m, name="con_3i1", domain=[s, y, t, h])
con_3i2 = Equation(m, name="con_3i2", domain=[s, y, t, h])
# con_3j = Equation(m, name="con_3j", domain=[n, t, h])
con_3k = Equation(m, name="con_3k", domain=[s, y, t, h])
con_3l = Equation(m, name="con_3l", domain=[s, y, t, h])
# con_3n = Equation(m, name="con_3n", domain=[n, t, h])
con_3p1 = Equation(m, name="con_3p1", domain=[g, y, t, h])
con_3p2 = Equation(m, name="con_3p2", domain=[g, y, t, h])
con_3r = Equation(m, name="con_3r", domain=[n, y, t, h]) # n == ref bus


def build_ilsp_eqns(yi, ji):
    OF_ilsp[...] = min_op_cost_y == Sum(t, sigma_yt[yi, t] * Sum(h, tau_yth[yi, t, h] * (Sum(g, CG_gyk[g, yi] * pG_gythi[g, yi, t, h, ji])\
    + Sum(r, CR_r[r] * (gammaR_ryth[r, yi, t, h] * PR_ryk[r, yi] - pR_rythi[r, yi, t, h, ji]))\
    + Sum(d, CLS_d[d] * pLS_dythi[d, yi, t, h, ji]))))
    con_6b[n, t, h] = Sum(g, pG_gythi[g, yi, t, h, ji]) + Sum(r, pR_rythi[r, yi, t, h, ji])\
    + Sum(l.where[rel_n[l, n]], pL_lythi[l, yi, t, h, ji])\
    - Sum(l.where[sel_n[l, n]], pL_lythi[l, yi, t, h, ji])\
    + Sum(s, pSD_sythi[s, yi, t, h, ji] - pSC_sythi[s, yi, t, h, ji])\
    ==Sum(d, gammaD_dyth[d, yi, t, h] * PD_dyk[d, yi] - pLS_dythi[d, yi, t, h, ji])
    con_6c[lc, t, h] = pL_lythi[lc, yi, t, h, ji] == (VL_lyj_prev[lc, yi] / X_l[lc]) * (Sum(n.where[sel_n[lc, n]], theta_nythi[n, yi, t, h, ji])\
    - Sum(n.where[rel_n[lc, n]], theta_nythi[n, yi, t, h, ji]))
    con_6d[d, t, h] = pLS_dythi[d, yi, t, h, ji] <= gammaD_dyth[d, yi, t, h] * PD_dyk[d, yi]
    con_6e1[g, t, h] = pG_gythi[g, yi, t, h, ji] <= uG_gyth[g, yi, t, h] * PG_gyk[g, yi]
    con_6e2[g, t, h] = pG_gythi[g, yi, t, h, ji] >= uG_gyth[g, yi, t, h] * PG_g_min[g]
    con_6f[r, t, h] = pR_rythi[r, yi, t, h, ji] <= gammaR_ryth[r, yi, t, h] * PR_ryk[r, yi]

    con_3c[lc, yi, t, h] = pL_lythi[lc, yi, t, h, ji] == (1.0 / X_l[lc]) * (Sum(n.where[sel_n[lc, n]], theta_nythi[n, yi, t, h, ji])\
    - Sum(n.where[rel_n[lc, n]], theta_nythi[n, yi, t, h, ji]))
    con_3e1[l, yi, t, h] = pL_lythi[l, yi, t, h, ji] <= PL_l[l]
    con_3e2[l, yi, t, h] = pL_lythi[l, yi, t, h, ji] >= -PL_l[l]
    con_3fg[s, yi, t, h] = eS_sythi[s, yi, t, h, ji] == eS_sythi[s, yi, t, h.lag(1), ji] + (pSC_sythi[s, yi, t, h, ji] * etaSC_s[s] - (pSD_sythi[s, yi, t, h, ji] / etaSD_s[s])) * tau_yth[yi, t, h]
    con_3fg[s, yi, t, h].where[Ord(h) == 1] = eS_sythi[s, yi, t, h, ji] == ES_syt0[s, yi, t] + (pSC_sythi[s, yi, t, h, ji] * etaSC_s[s] - (pSD_sythi[s, yi, t, h, ji] / etaSD_s[s])) * tau_yth[yi, t, h]
    con_3h[s, yi, t, h] = eS_sythi[s, yi, t, h, ji] >= ES_syt0[s, yi, t]
    con_3i1[s, yi, t, h] = eS_sythi[s, yi, t, h, ji] <= ES_s_max[s]
    con_3i2[s, yi, t, h] = eS_sythi[s, yi, t, h, ji] >= ES_s_min[s]
    con_3k[s, yi, t, h] = pSC_sythi[s, yi, t, h, ji] <= uS_syth[s, yi, t, h] * PSC_s[s]
    con_3l[s, yi, t, h] = pSD_sythi[s, yi, t, h, ji] <= (1 - uS_syth[s, yi, t, h]) * PSD_s[s]
    con_3p1[g, yi, t, h] = pG_gythi[g, yi, t, h, ji] - pG_gythi[g, yi, t, h.lag(1), ji] <= RGU_g[g]
    con_3p2[g, yi, t, h] = pG_gythi[g, yi, t, h, ji] - pG_gythi[g, yi, t, h.lag(1), ji] >= -RGD_g[g]
    con_3r[n, yi, t, h].where[Ord(n) == 1] = theta_nythi[n, yi, t, h, ji] == 0

ilsp_eqns = [OF_ilsp, con_6b, con_6c, con_6d, con_6e1, con_6e2, con_6f, con_3c, con_3e1, con_3e2, con_3fg, con_3h,
             con_3i1, con_3i2, con_3k, con_3l, con_3p1, con_3p2, con_3r]

## ADA-based initialization of the inner loop
# First linear problem OF and constraints
con_7b = Equation(m, name="con_7b", domain=[g])
con_7c = Equation(m, name="con_7c", domain=[g])
con_7d = Equation(m, name="con_7d")
con_7e = Equation(m, name="con_7e", domain =[v])
lp1_obj_var = Equation(m, name="lp1_obj_var")
# Constraints 5d-5z

def build_lp1_eqns(yi):
    lp1_obj_var[...] = xiP_y[yi] == xiP
    con_7b[g] = cG_gy[g, yi] == CG_g_fc[g] * power(1 + zetaGC_g_fc, yi - 1) + CG_g_max[g] * power(1 + zetaGC_g_max, yi - 1) * aGC_gy[g, yi]
    con_7c[g] = aGC_gy[g, yi] <= 1
    con_7d[...] = Sum(g, aGC_gy[g, yi]) <= GammaGC
    con_7e[v] = xiP <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h]*PD_dyo[d,yi]*Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,v])) \
    - Sum(l, PL_l[l]*(muL_lythv_lo[l,yi,t,h,v] + muL_lythv_up[l,yi,t,h,v])) \
    - Sum(s, US_sythv[s,yi,t,h,v]*PSC_s[s]*muSC_sythv_up[s,yi,t,h,v] + (1-US_sythv[s,yi,t,h,v])*PSD_s[s]*muSD_sythv_up[s,yi,t,h,v] - ES_s_min[s]*muS_sythv_lo[s,yi,t,h,v] + ES_s_max[s]*muS_sythv_up[s,yi,t,h,v])\
    + Sum(g, UG_gythv[g,yi,t,h,v]*(PG_g_min[g]*muG_gythv_lo[g,yi,t,h,v] - PG_gyo[g,yi] * muG_gythv_up[g,yi,t,h,v])) \
    - Sum(r, gammaR_ryth[r,yi,t,h]*PR_ryo[r,yi]*(muR_rythv_up[r,yi,t,h,v] - sigma_yt[yi,t]*tau_yth[yi,t,h] * CR_r[r])) \
    - Sum(d, gammaD_dyth[d,yi,t,h]*PD_dyo[d,yi]*muD_dythv_up[d,yi,t,h,v]))\
    + Sum(s, ES_syt0[s,yi,t]*(PhiS_sytv[s,yi,t,v] + PhiS_sytv_lo[s,yi,t,v])) \
    - Sum(h.where[Ord(h)>1], Sum(g, RGD_g[g]*muGD_gythv[g,yi,t,h,v] + RGU_g[g]*muGU_gythv[g,yi,t,h,v])))

    con_5d[g,t,v] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,1,v]) + muG_gythv_lo[g,yi,t,1,v]\
    - muG_gythv_up[g,yi,t,1,v] - muGD_gythv[g,yi,t,2,v] + muGU_gythv[g,yi,t,2,v]\
    == sigma_yt[yi,t] * tau_yth[yi, t, 1] * cG_gy[g, yi]
    con_5e[g,t,h,v].where[(Ord(h)!=1) & (Ord(h)!=Card(h))] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,h,v])\
    + muG_gythv_lo[g,yi,t,h,v]- muG_gythv_up[g,yi,t,h,v] + muGD_gythv[g,yi,t,h,v] - muGD_gythv[g,yi,t,h.lead(1),v]\
    - muGU_gythv[g,yi,t,h,v] + muGU_gythv[g,yi,t,h.lead(1),v] == sigma_yt[yi,t]*tau_yth[yi,t,h]*cG_gy[g, yi]
    con_5f[g,t,v] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,8,v]) +  muG_gythv_lo[g,yi,t,8,v]\
    - muG_gythv_up[g,yi,t,8,v] + muGD_gythv[g,yi,t,8,v] - muGU_gythv[g, yi, t, 8, v]\
    == sigma_yt[yi, t] * tau_yth[yi, t, 8] * cG_gy[g, yi]
    con_5g[d,t,h,v] = Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,v])-muD_dythv_up[d, yi, t, h, v]\
    <= sigma_yt[yi, t] * tau_yth[yi, t, h] * CLS_d[d]
    con_5h[r,t,h,v] = Sum(n.where[r_n[r,n]], lambdaN_nythv[n,yi,t,h,v]) - muR_rythv_up[r,yi,t,h,v]\
    <= sigma_yt[yi, t] * tau_yth[yi, t, h] * CR_r[r]
    con_5i[le,t,h,v] = Sum(n.where[rel_n[le,n]], lambdaN_nythv[n,yi,t,h,v]) - Sum(n.where[sel_n[le,n]], lambdaN_nythv[n,yi,t,h,v]) \
    + muL_lythv_exist[le, yi, t, h, v] + muL_lythv_lo[le, yi, t, h, v] - muL_lythv_up[le, yi, t, h, v] == 0
    con_5j[lc,t,h,v] = Sum(n.where[rel_n[lc,n]], lambdaN_nythv[n,yi,t,h,v]) - Sum(n.where[sel_n[lc,n]], lambdaN_nythv[n,yi,t,h,v]) \
    + muL_lythv_can[lc,yi,t,h,v] + muL_lythv_lo[lc,yi,t,h,v] - muL_lythv_up[lc,yi,t,h,v] == 0
    con_5k[s,t,v] = Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,1,v])\
    + (tau_yth[yi,t,1]/etaSD_s[s])*PhiS_sytv[s,yi,t,v] - muSD_sythv_up[s,yi,t,1,v] <= 0
    con_5l[s,t,h,v].where[Ord(h)>1] = Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,h,v])\
    + (tau_yth[yi,t,h]/etaSD_s[s])*muS_sythv[s,yi,t,h,v] - muSD_sythv_up[s,yi,t,h,v] <= 0
    con_5m[s,t,v] = -Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,1,v])\
    - etaSC_s[s]*tau_yth[yi,t,1]*PhiS_sytv[s,yi,t,v] - muSC_sythv_up[s,yi,t,1,v] <= 0
    con_5n[s,t,h,v].where[Ord(h)>1] = -Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,h,v])\
    - etaSC_s[s]*tau_yth[yi,t,h]*muS_sythv[s,yi,t,h,v] - muSC_sythv_up[s,yi,t,h,v] <= 0
    con_5o[n, t, h, v].where[Ord(n) > 1] = -Sum(le.where[sel_n[le, n]], muL_lythv_exist[le, yi, t, h, v] / X_l[le])\
    + Sum(le.where[rel_n[le, n]], muL_lythv_exist[le, yi, t, h, v] / X_l[le])\
    - Sum(lc.where[sel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, v])\
    + Sum(lc.where[rel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, v]) == 0
    con_5p[n, t, h, v].where[Ord(n) == 1] = (-Sum(le.where[sel_n[le, n]], muL_lythv_exist[le, yi, t, h, v] / X_l[le])
    + Sum(le.where[rel_n[le, n]], muL_lythv_exist[le, yi, t, h, v] / X_l[le])\
    - Sum(lc.where[sel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, v])\
    + Sum(lc.where[rel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, v])\
    + phiN_nythv[n, yi, t, h, v] == 0)
    con_5q[s,t,v] = PhiS_sytv[s,yi,t,v] - muS_sythv[s,yi,t,2,v] + muS_sythv_lo[s,yi,t,1,v] - muS_sythv_up[s,yi,t,1,v] == 0
    con_5r[s,t,h,v].where[(Ord(h)!=1) & (Ord(h)!=Card(h))] = muS_sythv[s,yi,t,h,v] - muS_sythv[s,yi,t,h.lead(1),v] + muS_sythv_lo[s,yi,t,h,v] - muS_sythv_up[s,yi,t,h,v] == 0
    con_5s[s,t,v] = muS_sythv[s,yi,t,8,v] + PhiS_sytv_lo[s,yi,t,v] + muS_sythv_lo[s,yi,t,8,v] - muS_sythv_up[s,yi,t,8,v] == 0

lp1_eqns = [lp1_obj_var, con_7b, con_7c, con_7d, con_7e, con_5d, con_5e, con_5f, con_5g, con_5h, con_5i, con_5j, con_5k, con_5l, con_5m,
            con_5n, con_5o, con_5p, con_5q, con_5r, con_5s]

# Second linear problem OF and constraints
con_8b = Equation(m, name="con_8b", domain=[d])
con_8c = Equation(m, name="con_8c", domain=[g])
con_8d = Equation(m, name="con_8d", domain=[r])
con_8e = Equation(m, name="con_8e", domain=[d])
con_8f = Equation(m, name="con_8f", domain=[g])
con_8g = Equation(m, name="con_8g", domain=[r])
con_8h = Equation(m, name="con_8h")
con_8i = Equation(m, name="con_8i")
con_8j = Equation(m, name="con_8j")
con_8k = Equation(m, name="con_8k")
con_8l = Equation(m, name="con_8l", domain=[v])
lp2_obj_var = Equation(m, name="lp2_obj_var")

lp2_eqns = [con_8b, con_8c, con_8d, con_8e, con_8f, con_8g, con_8h, con_8i, con_8j, con_8k, con_8l]

def build_lp2_eqns(yi):
    lp2_obj_var[...] = xiQ_y[yi] == xiQ
    con_8b[d] = pD_dy[d, yi] == PD_d_fc[d] * power(1 + zetaD_d_fc[d], yi - 1) + PD_d_max[d] * power(1 + zetaD_d_max[d], yi - 1) * aD_dy[d, yi]
    con_8c[g] = pG_gy[g, yi] == PG_g_fc[g] * power(1 - zetaGP_g_fc[g], yi - 1) - PG_g_max[g] * power(1 + zetaGP_g_max[g], yi - 1) * aGP_gy[g, yi]
    con_8d[r] = pR_ry[r, yi] == PR_r_fc[r] * power(1 + zetaR_r_fc[r], yi - 1) - PR_r_max[r] * power(1 + zetaR_r_max[r], yi - 1) * aR_ry[r, yi]
    con_8e[d] = aD_dy[d, yi] <= 1
    con_8f[g] = aGP_gy[g, yi] <= 1
    con_8g[r] = aR_ry[r, yi] <= 1
    con_8h[...] = Sum(d, aD_dy[d, yi]) <= GammaD
    con_8i[...] = Sum(g, aGP_gy[g, yi]) <= GammaGP
    con_8j[...] = Sum(rs, aR_ry[rs, yi]) <= GammaRS
    con_8k[...] = Sum(rw, aR_ry[rw, yi]) <= GammaRW
    con_8l[v] = xiQ <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d, yi, t, h] * pD_dy[d, yi] * Sum(n.where[d_n[d, n]], LambdaN_nythvo[n, yi, t, h, v]))\
    - Sum(l, PL_l[l] * (muL_lythvo_lo[l, yi, t, h, v] + muL_lythvo_up[l, yi, t, h, v])) - Sum(s, US_sythv[s, yi, t, h, v] * PSC_s[s] * muSC_sythvo_up[s, yi, t, h, v]\
    + (1 - US_sythv[s, yi, t, h, v]) * PSD_s[s] * muSD_sythvo_up[s, yi, t, h, v] - ES_s_min[s] * muS_sythvo_lo[s, yi, t, h, v]\
    + ES_s_max[s] * muS_sythvo_up[s, yi, t, h, v]) + Sum(g, UG_gythv[g, yi, t, h, v] * (PG_g_min[g] * muG_gythvo_lo[g, yi, t, h, v] - pG_gy[g, yi] * muG_gythvo_up[g, yi, t, h, v]))\
    - Sum(r, gammaR_ryth[r, yi, t, h] * pR_ry[r, yi] * (muR_rythvo_up[r, yi, t, h, v] - sigma_yt[yi, t] * tau_yth[yi, t, h] * CR_r[r]))\
    - Sum(d, gammaD_dyth[d, yi, t, h] * pD_dy[d, yi] * muD_dythvo_up[d, yi, t, h, v])) + Sum(s, ES_syt0[s, yi, t] * (PhiS_sytvo[s, yi, t, v] + PhiS_sytvo_lo[s, yi, t, v]))\
    - Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g] * muGD_gythvo[g, yi, t, h, v] + RGU_g[g] * muGU_gythvo[g, yi, t, h, v])))

# MODELS #
OLMP_model = Model(
    m,
    name="OLMP",
    description="Outer-loop master problem",
    equations=olmp_eqns,
    problem='MIP',
    sense='min',
    objective=min_inv_cost_wc,
)
ILSP_model = Model(
    m,
    name="ILSP",
    description="Inner-loop subproblem",
    equations=ilsp_eqns,
    problem='MIP',
    sense='min',
    objective=min_op_cost_y,
)
LP1_model = Model(
    m,
    name="lp1",
    description="Fist linear problem (ADA)",
    equations=lp1_eqns,
    problem='MIP',
    sense='max',
    objective=xiP,
)
LP2_model = Model(
    m,
    name="lp2",
    description="Second linear problem (ADA)",
    equations=lp2_eqns,
    problem='MIP',
    sense='max',
    objective=xiQ,
)
ILMP_model = Model(
    m,
    name="ILMP",
    description="Inner-loop mster problem",
    equations=ilmp_eqns,
    problem='MIP',
    sense='max',
    objective=xi,
)

# Test solve for the outer loop master problem
# summary = OLMP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_debug.txt"), output=sys.stdout)
# #redirect output to a file
# print("Objective Function Value:  ", round(OLMP_model.objective_value, 3))
# print(vL_ly.records)
# m.write(r'C:\Users\Kevin\OneDrive - McGill University\Research\Sandbox\optimization\multi-year_AROTNEP\results\aro_tnep_results.gdx')

# Set values of the uncertain parameters for the given outer loop iteration
def set_uncertain_params_olmp(j_iter):
    # At the first iteration, uncertain parameters equal their forecast values
    if j_iter == 1:
        CG_gyi[g,y,j_iter] = CG_g_fc[g]
        PD_dyi[d,y,j_iter] = PD_d_fc[d]
        PG_gyi[g,y,j_iter] = PG_g_fc[g]
        PR_ryi[r,y,j_iter] = PR_r_fc[r]
    else:
        df_cG = cG_gy.l.records
        df_pD = pD_dy.l.records
        df_pG = pG_gy.l.records
        df_pR = pR_ry.l.records
        df_cG['i'] = str(j_iter)
        df_pD['i'] = str(j_iter)
        df_pG['i'] = str(j_iter)
        df_pR['i'] = str(j_iter)
        df_cG_ord = df_cG[['g', 'y', 'i', 'level']]
        df_pD_ord = df_pD[['d', 'y', 'i', 'level']]
        df_pG_ord = df_pG[['g', 'y', 'i', 'level']]
        df_pR_ord = df_pR[['r', 'y', 'i', 'level']]
        CG_gyi.setRecords(df_cG_ord[df_cG_ord['i'] == str(j_iter)])
        PD_dyi.setRecords(df_pD_ord[df_pD_ord['i'] == str(j_iter)])
        PG_gyi.setRecords(df_pG_ord[df_pG_ord['i'] == str(j_iter)])
        PR_ryi.setRecords(df_pR_ord[df_pR_ord['i'] == str(j_iter)])

# Set values of the uncertain parameters for the given inner loop iteration
def set_uncertain_params_ilsp(k_iter, is_ada):
    # At the first iteration, uncertain parameters equal their forecast values
    if is_ada and k_iter == 1:
        CG_gyk[g,y] = CG_g_fc[g]
        PD_dyk[d,y] = PD_d_fc[d]
        PG_gyk[g,y] = PG_g_fc[g]
        PR_ryk[r,y] = PR_r_fc[r]
    else:
        CG_gyk.setRecords(cG_gy.l.records)
        PD_dyk.setRecords(pD_dy.l.records)
        PG_gyk.setRecords(pG_gy.l.records)
        PR_ryk.setRecords(pR_ry.l.records)

# Solve the relaxed outer-loop master problem
def solve_olmp_relaxed(j_iter, lb_o):
    ro = 1 # Initialize relaxed iteration counter
    # Solve at least once, until ro == j
    while ro <= j_iter:
        # Determine the subset i as a function of j and ro
        i_range = list(range(j_iter - ro + 1, j_iter + 1))
        i.setRecords(i_range)
        # Solve the outer-loop master problem
        build_olmp_eqns() # Rebuild the olmp equations to account for the change in set i
        OLMP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_olmp.txt"),output=sys.stdout)
        if OLMP_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible']:
            raise RuntimeError('OLMP is infeasible at j = {}'.format(j_iter))
        VL_lyj.setRecords(vL_ly.l.records)
        VL_lyj_prev.setRecords(vL_ly_prev.l.records)
        olmp_ov = OLMP_model.objective_value
        # Exit if ro == j or if optimal value exceeds lb_o, else increment ro and iterate again
        if ro == j_iter or olmp_ov > lb_o:
            logger.info("Relaxed OLMP iteration (ro = {}) equals outer-loop iteration (j = {}) or LBO has increased --> Exit OLMP".format(ro, j_iter))
            break
        else:
            ro += 1

    return olmp_ov

# Solve the inner-loop subproblem
def solve_ilsp(y_iter, j_iter, k_iter):
    v_range = list(range(1, k_iter + 1))
    v.setRecords(v_range)

    build_ilsp_eqns(y_iter, j_iter) # Rebuild the ilsp equations for the given year and outer loop iteration j
    ILSP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_ilsp.txt"),output=sys.stdout)
    if ILSP_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible']:
        raise RuntimeError('ILSP is infeasible at y = {}, j ={}, k = {}'.format(y_iter, j_iter, k_iter))
    df_uG = uG_gyth.l.records
    df_uS = uS_syth.l.records
    df_uG['v'] = str(k_iter)
    df_uS['v'] = str(k_iter)
    df_uG_ord = df_uG[['g', 'y', 't', 'h', 'v', 'level']]
    df_uS_ord = df_uS[['s', 'y', 't', 'h', 'v', 'level']]
    UG_gythv.setRecords(df_uG_ord[(df_uG_ord['y'] == str(y_iter)) & (df_uG_ord['v'] == str(k_iter))])
    US_sythv.setRecords(df_uS_ord[(df_uS_ord['y'] == str(y_iter)) & (df_uS_ord['v'] == str(k_iter))])
    ilsp_ov = ILSP_model.objective_value

    return ilsp_ov

# Solve the inner-loop master problem by using ADA
def solve_ilmp_ada(y_iter, j_iter, k_iter, tol):
    v_range = list(range(1, k_iter + 1))
    v.setRecords(v_range)

    build_lp1_eqns(y_iter)
    build_lp2_eqns(y_iter)
    # Set binary decision variables to the last solved value for the given inner loop iteration
    ada_ov = 0
    o_iter = 1
    for ada_iter in range(5):
        if o_iter == 1:
            PD_dyo[d,y] = PD_d_fc[d]
            PG_gyo[g,y] = PG_g_fc[g]
            PR_ryo[r,y] = PR_r_fc[r]
        LP1_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_lp1.txt"),output=sys.stdout)
        if LP1_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible']:
            raise RuntimeError('LP1 is infeasible at y = {}, j ={}, k = {}'.format(y_iter, j_iter, k_iter))
        LambdaN_nythvo.setRecords(lambdaN_nythv.l.records)
        muD_dythvo_up.setRecords(muD_dythv_up.l.records)
        muG_gythvo_lo.setRecords(muG_gythv_lo.l.records)
        muG_gythvo_up.setRecords(muG_gythv_up.l.records)
        muGD_gythvo.setRecords(muGD_gythv.l.records)
        muGU_gythvo.setRecords(muGU_gythv.l.records)
        muL_lythvo_lo.setRecords(muL_lythv_lo.l.records)
        muL_lythvo_up.setRecords(muL_lythv_up.l.records)
        muR_rythvo_up.setRecords(muR_rythv_up.l.records)
        muS_sythvo_lo.setRecords(muS_sythv_lo.l.records)
        muS_sythvo_up.setRecords(muS_sythv_up.l.records)
        muSC_sythvo_up.setRecords(muSC_sythv_up.l.records)
        muSD_sythvo_up.setRecords(muSD_sythv_up.l.records)
        PhiS_sytvo.setRecords(PhiS_sytv.l.records)
        PhiS_sytvo_lo.setRecords(PhiS_sytv_lo.l.records)
        lp1_ov = LP1_model.objective_value

        LP2_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_lp2.txt"),output=sys.stdout)
        if LP2_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible']:
            raise RuntimeError('LP2 is infeasible at y = {}, j ={}, k = {}'.format(y_iter, j_iter, k_iter))
        PD_dyo.setRecords(pD_dy.l.records)
        PG_gyo.setRecords(pG_gy.l.records)
        PR_ryo.setRecords(pR_ry.l.records)
        lp2_ov = LP2_model.objective_value

        if (abs(lp1_ov - lp2_ov) / min(lp1_ov, lp2_ov)) < tol:
            logger.info("ADA finished in {} iterations".format(o_iter))
            break
        else:
            logger.info("ADA iteration {} finished, incrementing iteration counter".format(o_iter))
            o_iter += 1
            if o_iter == max(range(5)):
                logger.info("ADA not solved in max number of iterations")
        ada_ov = min(lp1_ov, lp2_ov)

    return ada_ov

# Solve the relaxed inner-loop master problem
def solve_ilmp_relaxed(y_iter, j_iter, k_iter, ub_i):
    ri = 1 # Initialize relaxed iteration counter
    # Solve at least once, until ri == k
    while ri <= k_iter:
        # Determine the subset v as a function of k and ri
        v_range = list(range(k_iter - ri + 1, k_iter + 1))
        v.setRecords(v_range)
        # Solve the inner-loop master problem
        build_ilmp_eqns(y_iter) # Rebuild the ilmp equations to account for the change in set v
        ILMP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_ilmp.txt"),output=sys.stdout)
        if ILMP_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible']:
            raise RuntimeError('ILMP is infeasible at y = {}, j ={}, k = {}'.format(y_iter, j_iter, k_iter))
        ilmp_ov = ILMP_model.objective_value
        # Exit if ri == k or if optimal value is less than ub_i, else increment ri and iterate again
        if ri == k_iter or ilmp_ov < ub_i:
            logger.info("Relaxed ILMP iteration (ri = {}) equals inner-loop iteration (k = {}) or UBI has decreased --> Exit ILMP".format(ri, k_iter))
            break
        else:
            ri += 1

    return ilmp_ov

# SOLUTION PROCEDURE #
lb_o = -999999999999
ub_o = 999999999999
VL_lyjm1_rec = None
VL_lyjm1_prev_rec = None
j_iter = 1
# OUTER LOOP #
for ol_iter in range(5):
    j.setRecords(list(range(1, j_iter+1)))
    i.setRecords(j.records)
    set_uncertain_params_olmp(j_iter)
    lb_o = solve_olmp_relaxed(j_iter, lb_o)
    if j_iter > 1:
        if VL_lyj.records.equals(VL_lyjm1_rec) and VL_lyj_prev.records.equals(VL_lyjm1_prev_rec):
            logger.info("No change in investment decision variables --> End outer loop")
            break
    # INNER LOOP: ILSP + ADA ILMP #
    lb_i_ada = -999999999999
    ub_i_ada = 999999999999
    y_iter = 1
    k_iter_ada = 1
    for il_ada_iter in range(5):
        k.setRecords(list(range(1, k_iter_ada + 1)))
        # v.setRecords(k.records)
        set_uncertain_params_ilsp(k_iter_ada, is_ada=True)
        lb_i_ada = solve_ilsp(y_iter, j_iter, k_iter_ada)
        il_error = (ub_i_ada - lb_i_ada) / lb_i_ada
        if il_error < tol:
            logger.info("Inner loop (ADA) has converged after k = {} iterations --> End inner loop".format(k_iter_ada))
            break
        elif il_error >= tol:
            logger.info("Solve the inner-loop master problem by using ADA (AT1)")
            ub_i_ada = solve_ilmp_ada(y_iter, j_iter, k_iter_ada, tol)
            k_iter_ada += 1
    # INNER LOOP: ILSP + relaxed ILMP #
    lb_i_rel = -999999999999
    ub_i_rel = 999999999999
    k_iter_rel = 1
    for il_ada_iter in range(5):
        k.setRecords(list(range(1, k_iter_rel + 1)))
        # v.setRecords(k.records)
        set_uncertain_params_ilsp(k_iter_rel, is_ada=False)
        lb_i_rel = solve_ilsp(y_iter, j_iter, k_iter_rel)
        il_error = (ub_i_rel - lb_i_rel) / lb_i_rel
        if il_error < tol:
            logger.info("Inner loop (relaxed) has converged after k = {} iterations --> End inner loop".format(k_iter_rel))
            break
        elif il_error >= tol:
            logger.info("Solve the relaxed inner-loop master problem")
            ub_i_rel = solve_ilmp_relaxed(y_iter, j_iter, k_iter_rel, tol)
            k_iter_rel += 1

    vL_vals = vL_ly.l.records
    IL_vals = IL_l.records
    IL_vals.columns = ['lc', 'value']
    merge = pd.merge(vL_vals, IL_vals, on='lc')
    merge['product'] = merge['level'] * merge['value']
    inv_cost = merge['product'].sum()
    ub_o = ub_i_rel + inv_cost
    ol_error = (ub_o - lb_o) / lb_o
    if ol_error < tol:
        logger.info("Outer loop has converged after j = {} iterations --> End problem".format(j_iter))
        break
    else:
        j_iter += 1
print(vL_ly.records)


# Testing olmp stuff
# j_iter = 1
# ro = 1
# j.setRecords(list(range(1, j_iter+1)))
# i.setRecords(j.records)
# set_uncertain_params_olmp(j_iter)
#
# i_range = list(range(j_iter - ro + 1, j_iter + 1))
# i.setRecords(i_range)
# # # solve olmp
# build_olmp_eqns()
# OLMP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_olmp.txt"),output=sys.stdout)
# VL_lyj.setRecords(vL_ly.l.records)
# VL_lyj_prev.setRecords(vL_ly_prev.l.records)
#
# # Testing ilsp stuff
# k_iter_ada = 1
# set_uncertain_params_ilsp(k_iter_ada, is_ada=True)
# build_ilsp_eqns(1, 1)
# ILSP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_ilsp.txt"),output=sys.stdout)
#
# k.setRecords(list(range(1, k_iter_ada + 1)))
# v.setRecords(k.records)
# print(v.records)
# build_ilmp_eqns(1)
# build_lp1_eqns(1)
# build_lp2_eqns(1)
# solve_ilmp_ada(1, k_iter_ada, tol)
#
# set_uncertain_params_ilsp(k_iter_ada, is_ada=False)
# build_ilsp_eqns(1, 1)
# ILSP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_ilsp.txt"),output=sys.stdout)
# solve_ilmp_relaxed(1, k_iter_ada, tol)
#
# vL_vals = vL_ly.l.records
# IL_vals = IL_l.records
# IL_vals.columns = ['lc', 'value']
# merge = pd.merge(vL_vals, IL_vals, on='lc')
# merge['product'] = merge['level']*merge['value']
# print(kappa.records['value'])
# cost = merge['product'].sum()
# print(cost)

