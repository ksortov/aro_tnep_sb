from pandas.compat.numpy.function import validate_round
from pandas.core.dtypes.inference import is_re

from input_data_processing import (weights, RD1, lines, buses, ESS, CG, RES, loads, years_data, sigma_yt_data,
                                   tau_yth_data, gamma_dyth_data, gamma_ryth_data, ES_syt0_data, tol, static, ess_inv)
from gamspy import Alias, Container, Domain, Equation, Model, Options, Ord, Card, Parameter, Set, Smax, Sum, Variable
from gamspy.math import power, Max
from utils import logger
import pandas as pd
import sys

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
hm = Set(m, name="hm", domain=[h], records=h.records[1:-1], description="Subset of RTPs indexed by h that excludes the first and last elements")
hu = Set(m, name="hu", domain=[h], records=h.records[1:], description="Subset of RTPs indexed by h that excludes the first and last elements")

# Multidimensional sets used to make associations between buses, lines and generating units
d_n = Set(m, name="d_n", domain=[d, n], records=loads[['Load', 'Bus']], description="Set of loads connected to bus n")
g_n = Set(m, name="g_n", domain=[g, n], records=CG[['Generating unit', 'Bus']], description="Set of conventional units connected to bus n")
r_n = Set(m, name="r_n", domain=[r, n], records=RES[['Generating unit', 'Bus']], description="Set of renewable units connected to bus n")
s_n = Set(m, name="s_n", domain=[s, n], records=ESS[['Storage unit', 'Bus']], description="Set of storage units connected to bus n")
rel_n = Set(m, name="rel_n", domain=[l, n], records=lines[['Transmission line', 'To bus']], description="Receiving bus of transmission line l")
sel_n = Set(m, name="sel_n", domain=[l, n], records=lines[['Transmission line', 'From bus']], description="Sending bus of transmission line l")
# Sets of indices for the outer and inner loop problems
j = Set(m, name="j", description="Iteration of the outer loop")
ir = Set(m, name="ir", domain=[j], description="Iteration of the outer loop (relaxed)")
k = Set(m, name="k", description="Iteration of the inner loop")
va = Set(m, name="va", domain=[k], description="Iteration of the inner loop (ADA)")
vr = Set(m, name="vr", domain=[k], description="Iteration of the inner loop (Relaxed)")

# ALIAS #
yp = Alias(m, name="yp", alias_with=y)
# i = Alias(m, name="i", alias_with=j)
# vr = Alias(m, name="v", alias_with=k)

# PARAMETERS #
# Scalars
GammaD = Parameter(m, name="GammaD", records=8, description="Uncertainty budget for increased loads")
GammaGC = Parameter(m, name="GammaGC", records=5, description="Uncertainty budget for increased CG marginal cost")
GammaGP = Parameter(m, name="GammaGP", records=5, description="Uncertainty budget for decreased CG marginal cost")
GammaRS = Parameter(m, name="GammaRS", records=3, description="Uncertainty budget for decreased solar capacity")
GammaRW = Parameter(m, name="GammaRW", records=3, description="Uncertainty budget for decreased wind capacity")
kappa = Parameter(m, name="kappa", records=0.1, description="Discount rate")
IT = Parameter(m, name="IT", records=1500000000, description="Investment budget")
nb_H = Parameter(m, name="nb_H", records=8, description="Number of RTPs of each RD")
FL = Parameter(m, name="FL", records=150, description="Large constant for disjunctive linearization")
FD = Parameter(m, name="FD", records=100000, description="Large constant for exact linearization")
FD_up = Parameter(m, name="FD_up", records=100000, description="Large constant for exact linearization")
FG_up = Parameter(m, name="FG_up", records=100000, description="Large constant for exact linearization")
FR_up = Parameter(m, name="FR_up", records=100000, description="Large constant for exact linearization")

gammaD_dyth = Parameter(m, name="gammaD_dyth", domain=[d, y, t, h], records=gamma_dyth_data, description="Demand factor of load d")
gammaR_ryth = Parameter(m, name="gammaR_ryth", domain=[r, y, t, h], records=gamma_ryth_data, description="Capacity factor of renewable unit r")
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
CG_gyi = Parameter(m, name='CG_gyi', domain=[g, y, j], description="Worst-case realization of the marginal production cost of conventional generating unit g for relaxed outer loop iteration i")
PD_dyi = Parameter(m, name='PD_dyi', domain=[d, y, j], description="Worst-case realization of the peak power consumption of load d for relaxed outer loop iteration i")
PG_gyi = Parameter(m, name='PG_gyi', domain=[g, y, j], description="Worst-case realization of the capacity of conventional generating unit g for relaxed outer loop iteration i")
PR_ryi = Parameter(m, name='PR_ryi', domain=[r, y, j], description="Worst-case realization of the capacity of renewable generating unit r for relaxed outer loop iteration i")
CG_gyk = Parameter(m, name='CG_gyk', domain=[g, y], description="Worst-case realization of the marginal production cost of conventional generating unit g for inner loop iteration k")
PD_dyk = Parameter(m, name='PD_dyk', domain=[d, y], description="Worst-case realization of the peak power consumption of load d for inner loop iteration k")
PG_gyk = Parameter(m, name='PG_gyk', domain=[g, y], description="Worst-case realization of the capacity of conventional generating unit g for inner loop iteration k")
PR_ryk = Parameter(m, name='PR_ryk', domain=[r, y], description="Worst-case realization of the capacity of renewable generating unit r for inner loop iteration k")
CG_gyo = Parameter(m, name='CG_gyo', domain=[g, y], description="Worst-case realization of the marginal production cost of conventional generating unit g for ADA iteration o")
PD_dyo = Parameter(m, name='PD_dyo', domain=[d, y], description="Worst-case realization of the peak power consumption of load d for ADA iteration o")
PG_gyo = Parameter(m, name='PG_gyo', domain=[g, y], description="Worst-case realization of the capacity of conventional generating unit g for ADA iteration o")
PR_ryo = Parameter(m, name='PR_ryo', domain=[r, y], description="Worst-case realization of the capacity of renewable generating unit r for ADA iteration o")
LambdaN_nythvo = Parameter(m, name='LambdaN_nythvo', domain=[n, y, t, h, k], description="Dual variable associated with the power balance equation at bus n for ADA iteration o")
muD_dythvo_up = Parameter(m, name='muD_dythvo_up', domain=[d, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the unserved demand of load d for ADA iteration o")
muG_gythvo_lo = Parameter(m, name='muG_gythvo_lo', domain=[g, y, t, h, k], description="Dual variable associated with the constraint imposing the lower bound for the power produced by conventional generating unit g for ADA iteration o")
muG_gythvo_up = Parameter(m, name='muG_gythvo_up', domain=[g, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the power produced by conventional generating unit g for ADA iteration o")
muGD_gythvo = Parameter(m, name='muGD_gythvo', domain=[g, y, t, h, k], description="Dual variable associated with the constraint imposing the ramp-down limit of conventional generating unit g, being ℎ greater than 1 for ADA iteration o")
muGU_gythvo = Parameter(m, name='muGU_gythvo', domain=[g, y, t, h, k], description="Dual variable associated with the constraint imposing the ramp-up limit of conventional generating unit g, being ℎ greater than 1 for ADA iteration o")
muL_lythvo_lo = Parameter(m, name='muL_lythvo_lo', domain=[l, y, t, h, k], description="Dual variable associated with the constraint imposing the lower bound for the power flow through transmission line l for ADA iteration o")
muL_lythvo_up = Parameter(m, name='muL_lythvo_up', domain=[l, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the power flow through transmission line l for ADA iteration o")
muR_rythvo_up = Parameter(m, name='muR_rythvo_up', domain=[r, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the power produced by renewable generating unit r for ADA iteration o")
muS_sythvo_lo = Parameter(m, name='muS_sythvo_lo', domain=[s, y, t, h, k], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s for ADA iteration o")
muS_sythvo_up = Parameter(m, name='muS_sythvo_up', domain=[s, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the energy stored in storage facility s for ADA iteration o")
muSC_sythvo_up = Parameter(m, name='muSC_sythvo_up', domain=[s, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the charging power of storage facility s for ADA iteration o")
muSD_sythvo_up = Parameter(m, name='muSD_sythvo_up', domain=[s, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the discharging power of storage facility s for ADA iteration o")
PhiS_sytvo = Parameter(m, name='PhiS_sytvo', domain=[s, y, t, k], description="Dual variable associated with the energy stored in storage facility s at the first RTP of RD for ADA iteration o")
PhiS_sytvo_lo = Parameter(m, name='PhiS_sytvo_lo', domain=[s, y, t, k], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s at the last RTP of RD for ADA iteration o")
VL_lyj = Parameter(m, name='VL_lyj', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y, which is otherwise 0, for outer loop iteration j")
VL_lyj_prev = Parameter(m, name='VL_lyj_prev', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y or in previous years, which is otherwise 0, for outer loop iteration j")
UG_gythv = Parameter(m, name='UG_gythv', domain=[g, y, t, h, k], description="Binary variable used to model the commitment status of conventional unit g, for relaxed inner loop iteration v")
US_sythv = Parameter(m, name='US_sythv', domain=[s, y, t, h, k], description="Binary variable used to used to avoid the simultaneous charging and discharging of storage facility s, for relaxed inner loop iteration v")

# New parameters for storage investment decisions
se = Set(m, name="se", domain=[s], records=ESS[ESS['IS_s [$]']==0]['Storage unit'], description="Set of existing storage units indexed by s")
sc = Set(m, name="sc", domain=[s], records=ESS[ESS['IS_s [$]']>0]['Storage unit'], description="Set of candidate storage units indexed by s")
IS_s = Parameter(m, name="IS_s", domain=[s], records=ESS[['Storage unit', 'IS_s [$]']], description="Investment cost coefficient of candidate energy storage system s")
sigma_s = Parameter(m, name="sigma_s", domain=[s], records=ESS[['Storage unit', 'sigma_s']], description="Fixed operation ans maintenance cost of candidate energy storage system s as pe percentage of the investments costs")
VS_syj_prev = Parameter(m, name='VS_syj_prev', domain=[s, y], description="Binary variable that is equal to 1 if energy storage system s is built in year y or in previous years, which is otherwise 0, for outer loop iteration j")

# VARIABLES #
# Optimization variables
theta_nythi = Variable(m, name="theta_nythi", domain=[n, y, t, h, j], description="Voltage angle at bus n")
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
eS_sythi = Variable(m, name='eS_sythi', domain=[s, y, t, h, j], description="Energy stored in storage facility s")
pD_dy = Variable(m, name='pD_dy', domain=[d, y], description="Worst-case realization of the peak power consumption of load d")
pG_gythi = Variable(m, name='pG_gythi', domain=[g, y, t, h, j], description="Power produced by conventional generating unit g")
pG_gy = Variable(m, name='pG_gy', domain=[g, y], description="Worst-case realization of the capacity of conventional generating unit g")
pL_lythi = Variable(m, name='pL_lythi', domain=[l, y, t, h, j], description="Power flow through transmission line l")
pLS_dythi = Variable(m, type='positive', name='pLS_dythi', domain=[d, y, t, h, j], description="Unserved demand of load d")
pR_rythi = Variable(m, type='positive', name='pR_rythi', domain=[r, y, t, h, j], description="Power produced by renewable generating unit r")
pR_ry = Variable(m, name='pR_ry', domain=[r, y], description="Worst-case realization of the capacity of renewable generating unit r")
pSC_sythi = Variable(m, type='positive', name='pSC_sythi', domain=[s, y, t, h, j], description="Charging power of storage facility s")
pSD_sythi = Variable(m, type='positive', name='pSD_sythi', domain=[s, y, t, h, j], description="Discharging power of storage facility s")
uG_gythi = Variable(m, name='uG_gyth', type='binary', domain=[g, y, t, h, j], description="Binary variable used to model the commitment status of conventional unit g")
uS_sythi = Variable(m, name='uS_syth', type='binary', domain=[s, y, t, h, j], description="Binary variable used to used to avoid the simultaneous charging and discharging of storage facility s")
vL_ly = Variable(m, name='vL_ly', type='binary', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y, which is otherwise 0")
vL_ly_prev = Variable(m, name='vL_ly_prev', type='binary', domain=[lc, y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y or in previous years, which is otherwise 0")
zD_dy = Variable(m, name='zD_dy', type='binary', domain=[d, y], description="Binary variable that is equal to 1 if the worst-case realization of the peak power consumption of load 𝑑 is equal to its upper bound, which is otherwise 0")
zGC_gy = Variable(m, name='zGC_gy', type='binary', domain=[g, y], description="Binary variable that is equal to 1 if the worst-case realization of the marginal production cost of conventional generating unit g is equal to its upper bound, which is otherwise 0")
zGP_gy = Variable(m, name='zGP_gy', type='binary', domain=[g, y], description="Binary variable that is equal to 1 if the worst-case realization of the capacity of conventional generating unit g is equal to its upper bound, which is otherwise 0")
zR_ry = Variable(m, name='zR_ry', type='binary', domain=[r, y], description="Binary variable that is equal to 1 if the worst-case realization of he capacity of renewable generating unit r is equal to its upper bound, which is otherwise 0")

# New variables for ESS investments
vS_sy = Variable(m, name='vS_sy', type='binary', domain=[s, y], description="Binary variable that is equal to 1 if candidate energy storage system s is built in year y, which is otherwise 0")
vS_sy_prev = Variable(m, name='vS_sy_prev', type='binary', domain=[s, y], description="Binary variable that is equal to 1 if candidate energy storage system s is built in year y or in previous years, which is otherwise 0")
alphaS_sythi = Variable(m, name="alphaS_sythi", type='positive', domain=[s, y, t, h, j], description="Auxiliary variable for the linearization of zS_sy*uS_syth")
eS_syt0_ess = Variable(m, name="eS_syt0_ess", type='positive', domain=[s, y, t], description="Energy initially stored of storage facility s")
PhiS_syt0v = Variable(m, name='PhiS_syt0v_lo', type='positive', domain=[s, y, t, k], description="Dual variable associated with the constraint imposing the initial energy in storage facility s at the last RTP of RD")

# Dual variables
lambdaN_nythv = Variable(m, name='lambdaN_nythv', domain=[n, y, t, h, k], description="Dual variable associated with the power balance equation at bus n")
muD_dythv_up = Variable(m, name='muD_dythv_up', type='positive', domain=[d, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the unserved demand of load d")
muG_gythv_lo = Variable(m, name='muG_gythv_lo', type='positive', domain=[g, y, t, h, k], description="Dual variable associated with the constraint imposing the lower bound for the power produced by conventional generating unit g")
muG_gythv_up = Variable(m, name='muG_gythv_up', type='positive', domain=[g, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the power produced by conventional generating unit g")
muGD_gythv = Variable(m, name='muGD_gythv', type='positive', domain=[g, y, t, h, k], description="Dual variable associated with the constraint imposing the ramp-down limit of conventional generating unit g, being ℎ greater than 1")
muGU_gythv = Variable(m, name='muGU_gythv', type='positive', domain=[g, y, t, h, k], description="Dual variable associated with the constraint imposing the ramp-up limit of conventional generating unit g, being ℎ greater than 1")
muL_lythv_exist = Variable(m, name='muL_lythv_exist', domain=[le, y, t, h, k], description="Dual variable associated with the power flow through existing transmission line l")
muL_lythv_can = Variable(m, name='muL_lythv_can', domain=[lc, y, t, h, k], description="Dual variable associated with the power flow through candidate transmission line l")
muL_lythv_lo = Variable(m, name='muL_lythv_lo', type='positive', domain=[l, y, t, h, k], description="Dual variable associated with the constraint imposing the lower bound for the power flow through transmission line l")
muL_lythv_up = Variable(m, name='muL_lythv_up', type='positive', domain=[l, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the power flow through transmission line l")
muR_rythv_up = Variable(m, name='muR_rythv_up', type='positive', domain=[r, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the power produced by renewable generating unit r")
muS_sythv = Variable(m, name='muS_sythv', domain=[s, y, t, h, k], description="Dual variable associated with the energy stored in storage facility s, being h greater than 1")
muS_sythv_lo = Variable(m, name='muS_sythv_lo', type='positive', domain=[s, y, t, h, k], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s")
muS_sythv_up = Variable(m, name='muS_sythv_up', type='positive', domain=[s, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the energy stored in storage facility s")
muSC_sythv_up = Variable(m, name='muSC_sythv_up', type='positive', domain=[s, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the charging power of storage facility s")
muSD_sythv_up = Variable(m, name='muSD_sythv_up', type='positive', domain=[s, y, t, h, k], description="Dual variable associated with the constraint imposing the upper bound for the discharging power of storage facility s")
PhiS_sytv = Variable(m, name='PhiS_sytv', domain=[s, y, t, k], description="Dual variable associated with the energy stored in storage facility s at the first RTP of RD") # Check if it should really be positive
PhiS_sytv_lo = Variable(m, name='PhiS_sytv_lo', type='positive', domain=[s, y, t, k], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s at the last RTP of RD")
phiN_nythv = Variable(m, name='phiN_nythv', domain=[n, y, t, h, k], description="Dual variable associated with the definition of the reference bus n")

# Linearization variables
alphaD_dythv = Variable(m, name="alphaD_dythv", domain=[d, y, t, h, k], description="Auxiliary variable for the linearization of zD_dy*lambdaN_nyth")
alphaD_dythv_up = Variable(m, name="alphaD_dythv_up", type='positive', domain=[d, y, t, h, k], description="Auxiliary variable for the linearization of zD_dy*muD_dyth_up")
alphaGP_gythv_up = Variable(m, name="alphaGP_gythv_up", type='positive', domain=[g, y, t, h, k], description="Auxiliary variable for the linearization of zGP_gy*muGP_gyth_up")
alphaR_rythv_up = Variable(m, name="alphaR_rythv_up", type='positive', domain=[r, y, t, h, k], description="Auxiliary variable for the linearization of zR_ry*muR_ryth_up")

min_inv_cost_wc = Variable(m, name="min_inv_cost_wc", description="Worst-case investment costs")
min_op_cost_y = Variable(m, name="min_op_cost_y", description="Minimized operating costs for year y")

# EQUATIONS #
# Outer-loop master problem OF and constraints
OF_olmp = Equation(m, name="OF_olmp", type="regular")
OF_olmp_ess = Equation(m, name="OF_olmp_ess", type="regular")
con_1c = Equation(m, name="con_1c")
con_1c_ess = Equation(m, name="con_1c_ess")
con_1d = Equation(m, name="con_1d", domain=[lc])
con_1e = Equation(m, name="con_1e", domain=[lc, y])
con_4c = Equation(m, name="con_4c", domain=[y, ir])
con_4d = Equation(m, name="con_4d", domain=[n, y, t, h, ir])
con_4e = Equation(m, name="con_4e", domain=[le, y, t, h, ir])
con_4f_lin1 = Equation(m, name="con_4f_lin1", domain=[lc, y, t, h, ir]) # Linearized
con_4f_lin2 = Equation(m, name="con_4f_lin2", domain=[lc, y, t, h, ir]) # Linearized
con_4g_exist_lin1 = Equation(m, name="con_4g_exist_lin1", domain=[le, y, t, h, ir])
con_4g_exist_lin2 = Equation(m, name="con_4g_exist_lin2", domain=[le, y, t, h, ir])
con_4g_can_lin1 = Equation(m, name="con_4g_can_lin1", domain=[lc, y, t, h, ir])
con_4g_can_lin2 = Equation(m, name="con_4g_can_lin2", domain=[lc, y, t, h, ir])
con_4h = Equation(m, name="con_4h", domain=[s, y, t, ir]) # H == 1
con_4h_ess = Equation(m, name="con_4h_ess", domain=[s, y, t, ir]) # H == 1
con_4h_ess0 = Equation(m, name="con_4h_ess0", domain=[s,y,t])
con_4i = Equation(m, name="con_4i", domain=[s, y, t, h, ir]) # H =/= 1
con_4i_ess = Equation(m, name="con_4i_ess", domain=[s, y, t, h, ir]) # H =/= 1
con_4j = Equation(m, name="con_4j", domain=[s, y, t, ir]) # H == Hmax
con_4k1 = Equation(m, name="con_4k1", domain=[s, y, t, h, ir])
con_4k2 = Equation(m, name="con_4k2", domain=[s, y, t, h, ir])
con_4j_ess = Equation(m, name="con_4j_ess", domain=[s, y, t, ir]) # H == Hmax
con_4k1_ess = Equation(m, name="con_4k1_ess", domain=[s, y, t, h, ir])
con_4k2_ess = Equation(m, name="con_4k2_ess", domain=[s, y, t, h, ir])
# con_4l = Equation(m, name="con_4l", domain=[S,T,H,Y])
con_4m = Equation(m, name="con_4m", domain=[s, y, t, h, ir])
con_4n = Equation(m, name="con_4n", domain=[s, y, t, h, ir])
con_4v1 = Equation(m, name="con_4v1", domain=[sc])
con_4v2 = Equation(m, name="con_4v2", domain=[sc,y])
con_4ess_lin1 = Equation(m, name="con_4ess_lin1", domain=[s, y, t, h, ir]) # Linearized
con_4ess_lin2 = Equation(m, name="con_4ess_lin2", domain=[s, y, t, h, ir]) # Linearized
con_4ess_lin3 = Equation(m, name="con_4ess_lin3", domain=[s, y, t, h, ir]) # Linearized
con_4m_ess = Equation(m, name="con_4m_ess", domain=[s, y, t, h, ir])
con_4n_ess = Equation(m, name="con_4n_ess", domain=[s, y, t, h, ir])
con_4o = Equation(m, name="con_4o", domain=[d, y, t, h, ir])
# con_4p = Equation(m, name="con_4p", domain=[G,T,H,Y])
con_4q1 = Equation(m, name="con_4q1", domain=[g, y, t, h, ir])
con_4q2 = Equation(m, name="con_4q2", domain=[g, y, t, h, ir])
con_4r1 = Equation(m, name="con_4r1", domain=[g, y, t, h, ir]) # H =/= 1
con_4r2 = Equation(m, name="con_4r2", domain=[g, y, t, h, ir]) # H =/= 1
con_4s = Equation(m, name="con_4s", domain=[r, y, t, h, ir])
con_4t = Equation(m, name="con_4t", domain=[y, t, h, ir]) # N == ref bus

con_4q_ess = Equation(m, name="con_4q_ess", domain=[n])

def build_olmp_eqns(ess_inv, i_range):
    imin = min(i_range)
    imax = max(i_range)
    # ir = (j.val >= imin) & (j.val <= imax)
    hmax = int(nb_H.toValue())

    # Original objective function and limit on investment costs
    OF_olmp[...] = min_inv_cost_wc == Sum(y, (1.0 / power(1.0 + kappa, y.val - 1)) * ((rho_y[y] / (1.0 + kappa)) + \
                     Sum(lc, IL_l[lc] * vL_ly[lc, y])))
    con_1c[...] = Sum(lc, Sum(y, (1.0 / power(1.0 + kappa, y.val - 1)) * IL_l[lc] * vL_ly[lc, y])) <= IT

    # Modified objective function with ESS investment costs
    OF_olmp_ess[...] = min_inv_cost_wc == Sum(y, (1.0 / power(1.0 + kappa, y.val - 1)) * (((rho_y[y]+Sum(sc, vS_sy_prev[sc,y]*sigma_s[sc]*IS_s[sc])) / (1.0 + kappa)) + \
                    Sum(lc, IL_l[lc] * vL_ly[lc, y]) + Sum(sc, IS_s[sc] * vS_sy[sc, y])))
    con_1c_ess[...] = Sum(y, (1.0 / power(1.0 + kappa, y.val - 1)) * (Sum(lc, IL_l[lc] * vL_ly[lc, y]) + Sum(sc, IS_s[sc] * vS_sy[sc, y]))) <= IT

    con_1d[lc] = Sum(y, vL_ly[lc, y]) <= 1
    con_1e[lc, y] = vL_ly_prev[lc, y] == Sum(yp.where[yp.val <= y.val], vL_ly[lc, yp])

    con_4c[y, ir] = rho_y[y] >= Sum(t, sigma_yt[y, t] * Sum(h, tau_yth[y, t, h] * (Sum(g, CG_gyi[g,y,ir] * pG_gythi[g, y, t, h, ir]) \
    + Sum(r, CR_r[r] * (gammaR_ryth[r, y, t, h] * PR_ryi[r,y,ir] - pR_rythi[r, y, t, h, ir])) \
    + Sum(d, CLS_d[d] * pLS_dythi[d, y, t, h, ir]))))
    con_4d[n, y, t, h, ir] = Sum(g.where[g_n[g, n]], pG_gythi[g, y, t, h, ir]) + Sum(r.where[r_n[r, n]], pR_rythi[r, y, t, h, ir]) \
    + Sum(l.where[rel_n[l, n]], pL_lythi[l, y, t, h, ir]) - Sum(l.where[sel_n[l, n]], pL_lythi[l, y, t, h, ir]) \
    + Sum(s.where[s_n[s, n]], pSD_sythi[s, y, t, h, ir] - pSC_sythi[s, y, t, h, ir]) \
    == Sum(d.where[d_n[d, n]], gammaD_dyth[d, y, t, h] * PD_dyi[d,y,ir] - pLS_dythi[d, y, t, h, ir])
    con_4e[le, y, t, h, ir] = pL_lythi[le, y, t, h, ir] == (1.0 / X_l[le]) * (Sum(n.where[sel_n[le, n]], theta_nythi[n, y, t, h, ir]) \
    - Sum(n.where[rel_n[le, n]], theta_nythi[n, y, t, h, ir]))
    con_4f_lin1[lc, y, t, h, ir] = pL_lythi[lc, y, t, h, ir] - (1 / X_l[lc]) * (Sum(n.where[sel_n[lc, n]], theta_nythi[n, y, t, h, ir]) \
    - Sum(n.where[rel_n[lc, n]], theta_nythi[n, y, t, h, ir])) <= (1 - vL_ly_prev[lc, y]) * FL
    con_4f_lin2[lc, y, t, h, ir] = pL_lythi[lc, y, t, h, ir] - (1 / X_l[lc]) * (Sum(n.where[sel_n[lc, n]], theta_nythi[n, y, t, h, ir]) \
    - Sum(n.where[rel_n[lc, n]], theta_nythi[n, y, t, h, ir])) >= -(1 - vL_ly_prev[lc, y]) * FL
    con_4g_exist_lin1[le, y, t, h, ir] = pL_lythi[le, y, t, h, ir] <= PL_l[le]
    con_4g_exist_lin2[le, y, t, h, ir] = pL_lythi[le, y, t, h, ir] >= -PL_l[le]
    con_4g_can_lin1[lc, y, t, h, ir] = pL_lythi[lc, y, t, h, ir] <= vL_ly_prev[lc, y] * PL_l[lc]
    con_4g_can_lin2[lc, y, t, h, ir] = pL_lythi[lc, y, t, h, ir] >= -vL_ly_prev[lc, y] * PL_l[lc]
    con_4h[se, y, t, ir] = eS_sythi[se, y, t, 1, ir] == ES_syt0[se,y,t] + (pSC_sythi[se, y, t, 1, ir] * etaSC_s[se] \
    - (pSD_sythi[se, y, t, 1, ir] / etaSD_s[se])) * tau_yth[y, t, 1]
    con_4h_ess[sc,y,t,ir] = eS_sythi[sc, y, t, 1, ir] == eS_syt0_ess[sc,y,t] + (pSC_sythi[sc, y, t, 1, ir] * etaSC_s[sc] \
    - (pSD_sythi[sc, y, t, 1, ir] / etaSD_s[sc])) * tau_yth[y, t, 1]
    con_4h_ess0[sc,y,t] = eS_syt0_ess[sc,y,t] == vS_sy_prev[sc,y]*ES_syt0[sc,y,t]
    con_4i[se, y, t, h, ir].where[(Ord(h) > 1)] = eS_sythi[se, y, t, h, ir] == eS_sythi[se, y, t, h.lag(1), ir] \
    + (pSC_sythi[se, y, t, h, ir] * etaSC_s[se] - (pSD_sythi[se, y, t, h, ir] / etaSD_s[se])) * tau_yth[y, t, h]
    con_4i_ess[sc, y, t, h, ir].where[(Ord(h) > 1)] = eS_sythi[sc, y, t, h, ir] == eS_sythi[sc, y, t, h.lag(1), ir] \
    + (pSC_sythi[sc, y, t, h, ir] * etaSC_s[sc] - (pSD_sythi[sc, y, t, h, ir] / etaSD_s[sc])) * tau_yth[y, t, h]

    con_4j[se, y, t, ir] =  eS_sythi[se, y, t, hmax, ir] >= ES_syt0[se, y, t]
    con_4k1[se, y, t, h, ir] = eS_sythi[se, y, t, h, ir] <= ES_s_max[se]
    con_4k2[se, y, t, h, ir] = eS_sythi[se, y, t, h, ir] >= ES_s_min[se]
    con_4j_ess[sc, y, t, ir] = eS_sythi[sc, y, t, hmax, ir] >= vS_sy_prev[sc,y] * ES_syt0[sc, y, t]
    con_4k1_ess[sc, y, t, h, ir] = eS_sythi[sc, y, t, h, ir] <= vS_sy_prev[sc,y] * ES_s_max[sc]
    con_4k2_ess[sc, y, t, h, ir] = eS_sythi[sc, y, t, h, ir] >= vS_sy_prev[sc,y] * ES_s_min[sc]
    # Original constraints for avoiding simultaneous charging and discharging, max and min charging/discharging power
    con_4m[s, y, t, h, ir] = pSC_sythi[s, y, t, h, ir] <= uS_sythi[s, y, t, h, ir] * PSC_s[s]
    con_4n[s, y, t, h, ir] = pSD_sythi[s, y, t, h, ir] <= (1 - uS_sythi[s, y, t, h, ir]) * PSD_s[s]

    # Modified constraints for avoiding simultaneous charging and discharging, max and min charging/discharging power
    con_4v1[sc] = Sum(y, vS_sy[sc, y]) <= 1
    con_4v2[sc, y] = vS_sy_prev[sc, y] == Sum(yp.where[yp.val <= y.val], vS_sy[sc, yp])
    con_4ess_lin1[s, y, t, h, ir] = alphaS_sythi[s, y, t, h, ir] <= vS_sy_prev[s, y]
    con_4ess_lin2[s, y, t, h, ir] = alphaS_sythi[s, y, t, h, ir] <= uS_sythi[s, y, t, h, ir]
    con_4ess_lin3[s, y, t, h, ir] = alphaS_sythi[s, y, t, h, ir] >= vS_sy_prev[s, y] + uS_sythi[s, y, t, h, ir] - 1
    con_4m_ess[s, y, t, h, ir] = pSC_sythi[s, y, t, h, ir] <= alphaS_sythi[s,y,t,h,ir] * PSC_s[s]
    con_4n_ess[s, y, t, h, ir] = pSD_sythi[s, y, t, h, ir] <= (vS_sy_prev[s,y] - alphaS_sythi[s,y,t,h,ir]) * PSD_s[s]

    con_4o[d, y, t, h, ir] = pLS_dythi[d, y, t, h, ir] <= gammaD_dyth[d, y, t, h] * PD_dyi[d,y,ir]  # pD_dy[D,Y]
    con_4q1[g, y, t, h, ir] = pG_gythi[g, y, t, h, ir] <= uG_gythi[g, y, t, h, ir] * PG_gyi[g,y,ir]  # pG_gy[G,Y]
    con_4q2[g, y, t, h, ir] = pG_gythi[g, y, t, h, ir] >= uG_gythi[g, y, t, h, ir] * PG_g_min[g]
    con_4r1[g, y, t, h, ir].where[Ord(h) > 1] = pG_gythi[g, y, t, h, ir] - pG_gythi[g, y, t, h.lag(1), ir] <= RGU_g[g]
    con_4r2[g, y, t, h, ir].where[Ord(h) > 1] = pG_gythi[g, y, t, h, ir] - pG_gythi[g, y, t, h.lag(1), ir] >= -RGD_g[g]
    con_4s[r, y, t, h, ir] = pR_rythi[r, y, t, h, ir] <= gammaR_ryth[r, y, t, h] * PR_ryi[r,y,ir]  # pR_ry[R,Y]
    con_4t[y, t, h, ir] = theta_nythi[1, y, t, h, ir] == 0

    # Only one candidate ESS may be built per bus
    con_4q_ess[n] = Sum(y, Sum(sc.where[s_n[sc,n]], vS_sy[sc,y])) <= 1

    olmp_eqns = [OF_olmp, con_1c, con_1d, con_1e, con_4c, con_4d, con_4e, con_4f_lin1, con_4f_lin2, con_4g_exist_lin1,
             con_4g_exist_lin2, con_4g_can_lin1, con_4g_can_lin2, con_4h, con_4i, con_4j, con_4k1, con_4k2, con_4m,
             con_4n, con_4o, con_4q1, con_4q2, con_4r1, con_4r2, con_4s, con_4t]
    olmp_ess_eqns = [OF_olmp_ess, con_1c_ess, con_1d, con_1e, con_4c, con_4d, con_4e, con_4f_lin1, con_4f_lin2, con_4g_exist_lin1,
                 con_4g_exist_lin2, con_4g_can_lin1, con_4g_can_lin2, con_4h, con_4h_ess, con_4h_ess0, con_4i, con_4i_ess,
                 con_4j, con_4k1, con_4k2, con_4j_ess, con_4k1_ess, con_4k2_ess, con_4m, con_4n, con_4m_ess,
                 con_4n_ess, con_4o, con_4q1, con_4q2, con_4r1, con_4r2, con_4s, con_4t, con_4v1, con_4v2, con_4q_ess, con_4ess_lin1, con_4ess_lin2, con_4ess_lin3]
    eqns = olmp_ess_eqns if ess_inv else olmp_eqns
    OLMP_model = Model(
        m,
        name="OLMP",
        description="Outer-loop master problem",
        equations=eqns,
        problem='MIP',
        sense='min',
        objective=min_inv_cost_wc,
    )
    return OLMP_model

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
con_5c_lin_a = Equation(m, name="con_5c_lin_a", domain=[vr])
con_5c_lin_a_ess = Equation(m, name="con_5c_lin_a_ess", domain=[vr])
con_5c_lin_b1 = Equation(m, name="con_5c_lin_b1", domain=[d, t, h, vr])
con_5c_lin_b2 = Equation(m, name="con_5c_lin_b2", domain=[d, t, h, vr])
con_5c_lin_c1 = Equation(m, name="con_5c_lin_c1", domain=[d, t, h, vr])
con_5c_lin_c2 = Equation(m, name="con_5c_lin_c2", domain=[d, t, h, vr])
con_5c_lin_d = Equation(m, name="con_5c_lin_d", domain=[d, t, h, vr])
con_5c_lin_e1 = Equation(m, name="con_5c_lin_e1", domain=[d, t, h, vr])
con_5c_lin_e2 = Equation(m, name="con_5c_lin_e2", domain=[d, t, h, vr])
con_5c_lin_f = Equation(m, name="con_5c_lin_f", domain=[g, t, h, vr])
con_5c_lin_g1 = Equation(m, name="con_5c_lin_g1", domain=[g, t, h, vr])
con_5c_lin_g2 = Equation(m, name="con_5c_lin_g2", domain=[g, t, h, vr])
con_5c_lin_h = Equation(m, name="con_5c_lin_h", domain=[r, t, h, vr])
con_5c_lin_i1 = Equation(m, name="con_5c_lin_i1", domain=[r, t, h, vr])
con_5c_lin_i2 = Equation(m, name="con_5c_lin_i2", domain=[r, t, h, vr])

con_5d = Equation(m, name="con_5d", domain=[g, t, vr])
con_5e = Equation(m, name="con_5e", domain=[g, t, h, vr])
con_5f = Equation(m, name="con_5f", domain=[g, t, vr])
con_5g = Equation(m, name="con_5g", domain=[d, t, h, vr])
con_5h = Equation(m, name="con_5h", domain=[r, t, h, vr])
con_5i = Equation(m, name="con_5i", domain=[le, t, h, vr])
con_5j = Equation(m, name="con_5j", domain=[lc, t, h, vr])
con_5k = Equation(m, name="con_5k", domain=[s, t, vr])
con_5l = Equation(m, name="con_5l", domain=[s, t, h, vr])
con_5m = Equation(m, name="con_5m", domain=[s, t, vr])
con_5n = Equation(m, name="con_5n", domain=[s, t, h, vr])
con_5o = Equation(m, name="con_5o", domain=[n,t,h,vr]) # N =/= ref bus
con_5p = Equation(m, name="con_5p", domain=[n,t,h,vr]) # N == ref bus
con_5q = Equation(m, name="con_5q", domain=[s,t,vr])
con_5r = Equation(m, name="con_5r", domain=[s,t,h,vr])
con_5s = Equation(m, name="con_5s", domain=[s,t,vr])
con_5t = Equation(m, name="con_5t", domain=[s,t,vr])
ilmp_obj_var = Equation(m, name="ilmp_obj_var")

def build_ilmp_eqns(yi, v_range, ess_inv):
    vmin = min(v_range)
    vmax = max(v_range)
    # vr = (v.val >= vmin) & (v.val <= vmax)
    hmax = int(nb_H.toValue())

    con_2b[g] = cG_gy[g, yi] == CG_g_fc[g] * power(1 + zetaGC_g_fc[g], yi - 1) + CG_g_max[g] * power(1 + zetaGC_g_max[g], yi - 1) * zGC_gy[g, yi]
    con_2c[d] = pD_dy[d, yi] == PD_d_fc[d] * power(1 + zetaD_d_fc[d], yi - 1) + PD_d_max[d] * power(1 + zetaD_d_max[d], yi - 1) * zD_dy[d, yi]
    con_2d[g] = pG_gy[g, yi] == PG_g_fc[g] * power(1 + zetaGP_g_fc[g], yi - 1) - PG_g_max[g] * power(1 + zetaGP_g_max[g], yi - 1) * zGP_gy[g, yi]
    con_2e[r] = pR_ry[r, yi] == PR_r_fc[r] * power(1 + zetaR_r_fc[r], yi - 1) - PR_r_max[r] * power(1 + zetaR_r_max[r], yi - 1) * zR_ry[r, yi]
    con_2j[...] = Sum(g, zGC_gy[g, yi]) <= GammaGC
    con_2k[...] = Sum(d, zD_dy[d, yi]) <= GammaD
    con_2l[...] = Sum(g, zGP_gy[g, yi]) <= GammaGP
    con_2m[...] = Sum(rs, zR_ry[rs, yi]) <= GammaRS
    con_2n[...] = Sum(rw, zR_ry[rw, yi]) <= GammaRW

    ilmp_obj_var[...] = xi == xi_y[yi]
    con_5c_lin_a[vr] = xi_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h] * (PD_d_fc[d] * power(1+zetaD_d_fc[d], yi-1) \
    * Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,vr]) + PD_d_max[d] * power(1+zetaD_d_max[d], yi-1) * alphaD_dythv[d,yi,t,h,vr])) \
    - Sum(l, PL_l[l]*(muL_lythv_lo[l,yi,t,h,vr] + muL_lythv_up[l,yi,t,h,vr])) \
    - Sum(s, US_sythv[s,yi,t,h,vr]*PSC_s[s]*muSC_sythv_up[s,yi,t,h,vr] + (1-US_sythv[s,yi,t,h,vr])*PSD_s[s]*muSD_sythv_up[s,yi,t,h,vr] - ES_s_min[s]*muS_sythv_lo[s,yi,t,h,vr] + ES_s_max[s]*muS_sythv_up[s,yi,t,h,vr]) \
    + Sum(g, UG_gythv[g,yi,t,h,vr] * (PG_g_min[g] * muG_gythv_lo[g,yi,t,h,vr] - (PG_g_fc[g] * power(1-zetaGP_g_fc[g], yi-1) * muG_gythv_up[g,yi,t,h,vr] - PG_g_max[g] * power(1+zetaGP_g_max[g], yi-1) * alphaGP_gythv_up[g,yi,t,h,vr]))) \
    - Sum(r, gammaR_ryth[r,yi,t,h] * (PR_r_fc[r] * power(1+zetaR_r_fc, yi-1) * muR_rythv_up[r,yi,t,h,vr] - PR_r_max[r] * power(1+zetaR_r_max, yi-1) * alphaR_rythv_up[r,yi,t,h,vr])) \
    + Sum(r, sigma_yt[yi,t]*tau_yth[yi,t,h]*CR_r[r]*gammaR_ryth[r,yi,t,h]*(PR_r_fc[r]*power(1+zetaR_r_fc, yi-1) - PR_r_max[r]*power(1+zetaR_r_max, yi-1) * zR_ry[r,yi])) \
    - Sum(d, gammaD_dyth[d,yi,t,h] * (PD_d_fc[d] * power(1+zetaD_d_fc[d], yi-1) * muD_dythv_up[d,yi,t,h,vr] + PD_d_max[d] * power(1+zetaD_d_max[d], yi-1) * alphaD_dythv_up[d,yi,t,h,vr]))) \
    + Sum(s, ES_syt0[s,yi,t]*(PhiS_sytv[s,yi,t,vr] + PhiS_sytv_lo[s,yi,t,vr])) \
    - Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g]*muGD_gythv[g,yi,t,h,vr] + RGU_g[g]*muGU_gythv[g,yi,t,h,vr])))
    con_5c_lin_a_ess[vr] = xi_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h] * (PD_d_fc[d] * power(1+zetaD_d_fc[d], yi-1) \
    * Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,vr]) + PD_d_max[d] * power(1+zetaD_d_max[d], yi-1) * alphaD_dythv[d,yi,t,h,vr])) \
    - Sum(l, PL_l[l]*(muL_lythv_lo[l,yi,t,h,vr] + muL_lythv_up[l,yi,t,h,vr])) \
    - Sum(s, VS_syj_prev[s,yi]*(US_sythv[s,yi,t,h,vr]*PSC_s[s]*muSC_sythv_up[s,yi,t,h,vr] + (1-US_sythv[s,yi,t,h,vr])*PSD_s[s]*muSD_sythv_up[s,yi,t,h,vr] - ES_s_min[s]*muS_sythv_lo[s,yi,t,h,vr] + ES_s_max[s]*muS_sythv_up[s,yi,t,h,vr])) \
    + Sum(g, UG_gythv[g,yi,t,h,vr] * (PG_g_min[g] * muG_gythv_lo[g,yi,t,h,vr] - (PG_g_fc[g] * power(1-zetaGP_g_fc[g], yi-1) * muG_gythv_up[g,yi,t,h,vr] - PG_g_max[g] * power(1+zetaGP_g_max[g], yi-1) * alphaGP_gythv_up[g,yi,t,h,vr]))) \
    - Sum(r, gammaR_ryth[r,yi,t,h] * (PR_r_fc[r] * power(1+zetaR_r_fc, yi-1) * muR_rythv_up[r,yi,t,h,vr] - PR_r_max[r] * power(1+zetaR_r_max, yi-1) * alphaR_rythv_up[r,yi,t,h,vr])) \
    + Sum(r, sigma_yt[yi,t]*tau_yth[yi,t,h]*CR_r[r]*gammaR_ryth[r,yi,t,h]*(PR_r_fc[r]*power(1+zetaR_r_fc, yi-1) - PR_r_max[r]*power(1+zetaR_r_max, yi-1) * zR_ry[r,yi])) \
    - Sum(d, gammaD_dyth[d,yi,t,h] * (PD_d_fc[d] * power(1+zetaD_d_fc[d], yi-1) * muD_dythv_up[d,yi,t,h,vr] + PD_d_max[d] * power(1+zetaD_d_max[d], yi-1) * alphaD_dythv_up[d,yi,t,h,vr]))) \
    + Sum(s, VS_syj_prev[s,yi]*ES_syt0[s,yi,t]*(PhiS_syt0v[s,yi,t,vr] + PhiS_sytv_lo[s,yi,t,vr])) \
    - Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g]*muGD_gythv[g,yi,t,h,vr] + RGU_g[g]*muGU_gythv[g,yi,t,h,vr])))
    con_5c_lin_b1[d, t, h, vr] = alphaD_dythv[d, yi, t, h, vr] <= zD_dy[d, yi] * FD
    con_5c_lin_b2[d, t, h, vr] = alphaD_dythv[d, yi, t, h, vr] >= -zD_dy[d, yi] * FD
    con_5c_lin_c1[d,t,h,vr] = Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,vr]) - alphaD_dythv[d,yi,t,h,vr] <= (1 - zD_dy[d,yi]) * FD
    con_5c_lin_c2[d,t,h,vr] = Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,vr]) - alphaD_dythv[d,yi,t,h,vr] >= -(1 - zD_dy[d,yi]) * FD
    con_5c_lin_d[d, t, h, vr] = alphaD_dythv_up[d, yi, t, h, vr] <= zD_dy[d, yi] * FD_up
    con_5c_lin_e1[d, t, h, vr] = muD_dythv_up[d, yi, t, h, vr] - alphaD_dythv_up[d, yi, t, h, vr] <= (1 - zD_dy[d, yi]) * FD_up
    con_5c_lin_e2[d, t, h, vr] = muD_dythv_up[d, yi, t, h, vr] - alphaD_dythv_up[d, yi, t, h, vr] >= 0
    con_5c_lin_f[g, t, h, vr] = alphaGP_gythv_up[g, yi, t, h, vr] <= zGP_gy[g, yi] * FG_up
    con_5c_lin_g1[g,t,h,vr] = muG_gythv_up[g,yi,t,h,vr] - alphaGP_gythv_up[g,yi,t,h,vr] <= (1 - zGP_gy[g,yi]) * FG_up
    con_5c_lin_g2[g,t,h,vr] = muG_gythv_up[g,yi,t,h,vr] - alphaGP_gythv_up[g,yi,t,h,vr] >= 0
    con_5c_lin_h[r, t, h, vr] = alphaR_rythv_up[r, yi, t, h, vr] <= zR_ry[r, yi] * FR_up
    con_5c_lin_i1[r, t, h, vr] = muR_rythv_up[r, yi, t, h, vr] - alphaR_rythv_up[r, yi, t, h, vr] <= (1 - zR_ry[r, yi]) * FR_up
    con_5c_lin_i2[r, t, h, vr] = muR_rythv_up[r, yi, t, h, vr] - alphaR_rythv_up[r, yi, t, h, vr] >= 0

    con_5d[g,t,vr] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,1,vr]) + muG_gythv_lo[g,yi,t,1,vr]\
    - muG_gythv_up[g,yi,t,1,vr] - muGD_gythv[g,yi,t,2,vr] + muGU_gythv[g,yi,t,2,vr]\
    == sigma_yt[yi,t] * tau_yth[yi, t, 1] * cG_gy[g, yi]
    con_5e[g,t,h,vr].where[(Ord(h)!=1) & (Ord(h)!=Card(h))] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,h,vr])\
    + muG_gythv_lo[g,yi,t,h,vr] - muG_gythv_up[g,yi,t,h,vr] + muGD_gythv[g,yi,t,h,vr] - muGD_gythv[g,yi,t,h.lead(1),vr]\
    - muGU_gythv[g,yi,t,h,vr] + muGU_gythv[g,yi,t,h.lead(1),vr] == sigma_yt[yi,t]*tau_yth[yi,t,h]*cG_gy[g, yi]
    con_5f[g,t,vr] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,hmax,vr]) +  muG_gythv_lo[g,yi,t,hmax,vr]\
    - muG_gythv_up[g,yi,t,hmax,vr] + muGD_gythv[g,yi,t,hmax,vr] - muGU_gythv[g, yi, t, hmax, vr]\
    == sigma_yt[yi, t] * tau_yth[yi, t, hmax] * cG_gy[g, yi]
    con_5g[d,t,h,vr] = Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,vr])-muD_dythv_up[d, yi, t, h, vr]\
    <= sigma_yt[yi, t] * tau_yth[yi, t, h] * CLS_d[d]
    con_5h[r,t,h,vr] = Sum(n.where[r_n[r,n]], lambdaN_nythv[n,yi,t,h,vr]) - muR_rythv_up[r,yi,t,h,vr]\
    <= -sigma_yt[yi, t] * tau_yth[yi, t, h] * CR_r[r]
    con_5i[le,t,h,vr] = Sum(n.where[rel_n[le,n]], lambdaN_nythv[n,yi,t,h,vr]) - Sum(n.where[sel_n[le,n]], lambdaN_nythv[n,yi,t,h,vr]) \
    + muL_lythv_exist[le, yi, t, h, vr] + muL_lythv_lo[le, yi, t, h, vr] - muL_lythv_up[le, yi, t, h, vr] == 0
    con_5j[lc,t,h,vr] = Sum(n.where[rel_n[lc,n]], lambdaN_nythv[n,yi,t,h,vr]) - Sum(n.where[sel_n[lc,n]], lambdaN_nythv[n,yi,t,h,vr]) \
    + muL_lythv_can[lc,yi,t,h,vr] + muL_lythv_lo[lc,yi,t,h,vr] - muL_lythv_up[lc,yi,t,h,vr] == 0
    con_5k[s,t,vr] = Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,1,vr])\
    + (tau_yth[yi,t,1]/etaSD_s[s])*PhiS_sytv[s,yi,t,vr] - muSD_sythv_up[s,yi,t,1,vr] <= 0
    con_5l[s,t,h,vr].where[Ord(h)>1] = Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,h,vr])\
    + (tau_yth[yi,t,h]/etaSD_s[s])*muS_sythv[s,yi,t,h,vr] - muSD_sythv_up[s,yi,t,h,vr] <= 0
    con_5m[s,t,vr] = -Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,1,vr])\
    - etaSC_s[s]*tau_yth[yi,t,1]*PhiS_sytv[s,yi,t,vr] - muSC_sythv_up[s,yi,t,1,vr] <= 0
    con_5n[s,t,h,vr].where[Ord(h)>1] = -Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,h,vr])\
    - etaSC_s[s]*tau_yth[yi,t,h]*muS_sythv[s,yi,t,h,vr] - muSC_sythv_up[s,yi,t,h,vr] <= 0
    con_5o[n, t, h, vr].where[Ord(n) > 1] = -Sum(le.where[sel_n[le, n]], muL_lythv_exist[le, yi, t, h, vr] / X_l[le])\
    + Sum(le.where[rel_n[le, n]], muL_lythv_exist[le, yi, t, h, vr] / X_l[le])\
    - Sum(lc.where[sel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, vr])\
    + Sum(lc.where[rel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, vr]) == 0
    con_5p[n, t, h, vr].where[Ord(n) == 1] = -Sum(le.where[sel_n[le, n]], muL_lythv_exist[le, yi, t, h, vr] / X_l[le])\
    + Sum(le.where[rel_n[le, n]], muL_lythv_exist[le, yi, t, h, vr] / X_l[le])\
    - Sum(lc.where[sel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, vr])\
    + Sum(lc.where[rel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, vr])\
    + phiN_nythv[n, yi, t, h, vr] == 0
    con_5q[s,t,vr] = PhiS_sytv[s,yi,t,vr] - muS_sythv[s,yi,t,2,vr] + muS_sythv_lo[s,yi,t,1,vr] - muS_sythv_up[s,yi,t,1,vr] == 0
    con_5r[s,t,h,vr].where[(Ord(h)!=1) & (Ord(h)!=Card(h))] = muS_sythv[s,yi,t,h,vr] - muS_sythv[s,yi,t,h.lead(1),vr] + muS_sythv_lo[s,yi,t,h,vr] - muS_sythv_up[s,yi,t,h,vr] == 0
    con_5s[s,t,vr] = muS_sythv[s,yi,t,hmax,vr] + PhiS_sytv_lo[s,yi,t,vr] + muS_sythv_lo[s,yi,t,hmax,vr] - muS_sythv_up[s,yi,t,hmax,vr] == 0
    con_5t[s,t,vr] = PhiS_syt0v[s,yi,t,vr] - PhiS_sytv[s,yi,t,vr] + muS_sythv_lo[s,yi,t,1,vr] - muS_sythv_up[s,yi,t,1,vr] == 0

    ilmp_eqns = [con_2b, con_2c, con_2d, con_2e, con_2j, con_2k, con_2l, con_2m, con_2n, ilmp_obj_var, con_5c_lin_a, con_5c_lin_b1,
             con_5c_lin_b2, con_5c_lin_c1, con_5c_lin_c2, con_5c_lin_d, con_5c_lin_e1, con_5c_lin_e2, con_5c_lin_f,
             con_5c_lin_g1, con_5c_lin_g2, con_5c_lin_h, con_5c_lin_i1, con_5c_lin_i2, con_5d, con_5e, con_5f, con_5g,
             con_5h, con_5i, con_5j, con_5k, con_5l, con_5m, con_5n, con_5o, con_5p, con_5q, con_5r, con_5s]
    ilmp_eqns_ess = [con_2b, con_2c, con_2d, con_2e, con_2j, con_2k, con_2l, con_2m, con_2n, ilmp_obj_var, con_5c_lin_a_ess, con_5c_lin_b1,
             con_5c_lin_b2, con_5c_lin_c1, con_5c_lin_c2, con_5c_lin_d, con_5c_lin_e1, con_5c_lin_e2, con_5c_lin_f,
             con_5c_lin_g1, con_5c_lin_g2, con_5c_lin_h, con_5c_lin_i1, con_5c_lin_i2, con_5d, con_5e, con_5f, con_5g,
             con_5h, con_5i, con_5j, con_5k, con_5l, con_5m, con_5n, con_5o, con_5p, con_5q, con_5r, con_5s, con_5t]
    eqns = ilmp_eqns_ess if ess_inv else ilmp_eqns
    ILMP_model = Model(
        m,
        name="ILMP",
        description="Inner-loop mster problem",
        equations=eqns,
        problem='MIP',
        sense='max',
        objective=xi,
    )
    return ILMP_model

# Inner-loop subproblem OF and constraints
OF_ilsp = Equation(m, name="OF_ilsp", type="regular")
OF_ilsp_ess = Equation(m, name="OF_ilsp_ess", type="regular")
con_6b = Equation(m, name="con_6b", domain=[n, t, h])
con_6c = Equation(m, name="con_6c", domain=[lc, t, h])
con_6d = Equation(m, name="con_6d", domain=[d, t, h])
con_6e1 = Equation(m, name="con_6e1", domain=[g, t, h])
con_6e2 = Equation(m, name="con_6e2", domain=[g, t, h])
con_6f = Equation(m, name="con_6f", domain=[r, t, h])
con_3c = Equation(m, name="con_3c", domain=[le, y, t, h])
con_3e1 = Equation(m, name="con_3e1", domain=[l, y, t, h])
con_3e2 = Equation(m, name="con_3e2", domain=[l, y, t, h])
con_3f = Equation(m, name="con_3f", domain=[s, y, t])
con_3g = Equation(m, name="con_3g", domain=[s, y, t, h])
con_3f_ess = Equation(m, name="con_3f_ess", domain=[s, y, t])
con_3f_ess0 = Equation(m, name="con_3f_ess0", domain=[s, y, t])
con_3g_ess = Equation(m, name="con_3g_ess", domain=[s, y, t, h])
con_3h = Equation(m, name="con_3h", domain=[s, y, t])
con_3h_ess = Equation(m, name="con_3h_ess", domain=[s, y, t])
con_3i1 = Equation(m, name="con_3i1", domain=[s, y, t, h])
con_3i2 = Equation(m, name="con_3i2", domain=[s, y, t, h])
con_3i1_ess = Equation(m, name="con_3i1_ess", domain=[s, y, t, h])
con_3i2_ess = Equation(m, name="con_3i2_ess", domain=[s, y, t, h])
# con_3j = Equation(m, name="con_3j", domain=[n, t, h])
con_3k = Equation(m, name="con_3k", domain=[s, y, t, h])
con_3l = Equation(m, name="con_3l", domain=[s, y, t, h])
con_3k_ess = Equation(m, name="con_3k_ess", domain=[s, y, t, h])
con_3l_ess = Equation(m, name="con_3l_ess", domain=[s, y, t, h])
# con_3n = Equation(m, name="con_3n", domain=[n, t, h])
con_3p1 = Equation(m, name="con_3p1", domain=[g, y, t, h])
con_3p2 = Equation(m, name="con_3p2", domain=[g, y, t, h])
con_3r = Equation(m, name="con_3r", domain=[y, t, h]) # n == ref bus

def build_ilsp_eqns(ess_inv, yi, ji):
    hmax = int(nb_H.toValue())

    OF_ilsp[...] = min_op_cost_y == Sum(t, sigma_yt[yi, t] * Sum(h, tau_yth[yi, t, h] * (Sum(g, CG_gyk[g, yi] * pG_gythi[g, yi, t, h, ji])\
    + Sum(r, CR_r[r] * (gammaR_ryth[r, yi, t, h] * PR_ryk[r, yi] - pR_rythi[r, yi, t, h, ji]))\
    + Sum(d, CLS_d[d] * pLS_dythi[d, yi, t, h, ji]))))

    con_6b[n, t, h] = Sum(g.where[g_n[g,n]], pG_gythi[g, yi, t, h, ji]) + Sum(r.where[r_n[r,n]], pR_rythi[r, yi, t, h, ji])\
    + Sum(l.where[rel_n[l, n]], pL_lythi[l, yi, t, h, ji])\
    - Sum(l.where[sel_n[l, n]], pL_lythi[l, yi, t, h, ji])\
    + Sum(s.where[s_n[s,n]], pSD_sythi[s, yi, t, h, ji] - pSC_sythi[s, yi, t, h, ji])\
    ==Sum(d.where[d_n[d,n]], gammaD_dyth[d, yi, t, h] * PD_dyk[d, yi] - pLS_dythi[d, yi, t, h, ji])
    con_6c[lc, t, h] = pL_lythi[lc, yi, t, h, ji] == (VL_lyj_prev[lc, yi] / X_l[lc]) * (Sum(n.where[sel_n[lc, n]], theta_nythi[n, yi, t, h, ji])\
    - Sum(n.where[rel_n[lc, n]], theta_nythi[n, yi, t, h, ji]))
    con_6d[d, t, h] = pLS_dythi[d, yi, t, h, ji] <= gammaD_dyth[d, yi, t, h] * PD_dyk[d, yi]
    con_6e1[g, t, h] = pG_gythi[g, yi, t, h, ji] <= uG_gythi[g, yi, t, h, ji] * PG_gyk[g, yi]
    con_6e2[g, t, h] = pG_gythi[g, yi, t, h, ji] >= uG_gythi[g, yi, t, h, ji] * PG_g_min[g]
    con_6f[r, t, h] = pR_rythi[r, yi, t, h, ji] <= gammaR_ryth[r, yi, t, h] * PR_ryk[r, yi]

    con_3c[le, yi, t, h] = pL_lythi[le, yi, t, h, ji] == (1.0 / X_l[le]) * (Sum(n.where[sel_n[le, n]], theta_nythi[n, yi, t, h, ji])\
    - Sum(n.where[rel_n[le, n]], theta_nythi[n, yi, t, h, ji]))
    con_3e1[l, yi, t, h] = pL_lythi[l, yi, t, h, ji] <= PL_l[l]
    con_3e2[l, yi, t, h] = pL_lythi[l, yi, t, h, ji] >= -PL_l[l]
    con_3f[se, yi, t] = eS_sythi[se, yi, t, 1, ji] == ES_syt0[se, yi, t] + (pSC_sythi[se, yi, t, 1, ji] * etaSC_s[se] - (pSD_sythi[se, yi, t, 1, ji] / etaSD_s[se])) * tau_yth[yi, t, 1]
    con_3g[se, yi, t, h].where[Ord(h) > 1] = eS_sythi[se, yi, t, h, ji] == eS_sythi[se, yi, t, h.lag(1), ji] + (pSC_sythi[se, yi, t, h, ji] * etaSC_s[se] - (pSD_sythi[se, yi, t, h, ji] / etaSD_s[se])) * tau_yth[yi, t, h]
    con_3f_ess[sc, yi, t] = eS_sythi[sc, yi, t, 1, ji] == (eS_syt0_ess[sc, yi, t] + (pSC_sythi[sc, yi, t, 1, ji] * etaSC_s[sc] - (pSD_sythi[sc, yi, t, 1, ji] / etaSD_s[sc])) * tau_yth[yi, t, 1])
    con_3f_ess0[sc, yi, t] = eS_syt0_ess[sc, yi, t] == VS_syj_prev[sc, yi] * ES_syt0[sc, yi, t]
    con_3g_ess[sc, yi, t, h].where[Ord(h) > 1] = eS_sythi[sc, yi, t, h, ji] == (eS_sythi[sc, yi, t, h.lag(1), ji] + (pSC_sythi[sc, yi, t, h, ji] * etaSC_s[sc] - (pSD_sythi[sc, yi, t, h, ji] / etaSD_s[sc])) * tau_yth[yi, t, h])
    con_3h[se, yi, t] = eS_sythi[se, yi, t, hmax, ji] >= ES_syt0[se, yi, t]
    con_3h_ess[sc, yi, t] = eS_sythi[sc, yi, t, hmax, ji] >= VS_syj_prev[sc,yi] * ES_syt0[sc, yi, t]
    con_3i1[se, yi, t, h] = eS_sythi[se, yi, t, h, ji] <= ES_s_max[se]
    con_3i2[se, yi, t, h] = eS_sythi[se, yi, t, h, ji] >= ES_s_min[se]
    con_3i1_ess[sc, yi, t, h] = eS_sythi[sc, yi, t, h, ji] <= VS_syj_prev[sc,yi] * ES_s_max[sc]
    con_3i2_ess[sc, yi, t, h] = eS_sythi[sc, yi, t, h, ji] >= VS_syj_prev[sc,yi] * ES_s_min[sc]
    con_3k[se, yi, t, h] = pSC_sythi[se, yi, t, h, ji] <= uS_sythi[se, yi, t, h, ji] * PSC_s[se]
    con_3l[se, yi, t, h] = pSD_sythi[se, yi, t, h, ji] <= (1 - uS_sythi[se, yi, t, h, ji]) * PSD_s[se]
    con_3k_ess[sc, yi, t, h] = pSC_sythi[sc, yi, t, h, ji] <= uS_sythi[sc, yi, t, h, ji] * PSC_s[sc] * VS_syj_prev[sc,yi]
    con_3l_ess[sc, yi, t, h] = pSD_sythi[sc, yi, t, h, ji] <= (1 - uS_sythi[sc, yi, t, h, ji]) * PSD_s[sc] * VS_syj_prev[sc,yi]
    con_3p1[g, yi, t, h].where[Ord(h)>1] = pG_gythi[g, yi, t, h, ji] - pG_gythi[g, yi, t, h.lag(1), ji] <= RGU_g[g]
    con_3p2[g, yi, t, h].where[Ord(h)>1] = pG_gythi[g, yi, t, h, ji] - pG_gythi[g, yi, t, h.lag(1), ji] >= -RGD_g[g]
    con_3r[yi, t, h] = theta_nythi[1, yi, t, h, ji] == 0

    ilsp_eqns = [OF_ilsp, con_6b, con_6c, con_6d, con_6e1, con_6e2, con_6f, con_3c, con_3e1, con_3e2, con_3f, con_3g, con_3h,
             con_3i1, con_3i2, con_3k, con_3l, con_3p1, con_3p2, con_3r]
    ilsp_ess_eqns = [OF_ilsp, con_6b, con_6c, con_6d, con_6e1, con_6e2, con_6f, con_3c, con_3e1, con_3e2, con_3f, con_3g,
                 con_3f_ess, con_3f_ess0, con_3g_ess, con_3h, con_3h_ess, con_3i1, con_3i2, con_3i1_ess, con_3i2_ess, con_3k_ess, con_3k, con_3l, con_3l_ess, con_3p1, con_3p2, con_3r]
    eqns = ilsp_ess_eqns if ess_inv else ilsp_eqns
    ILSP_model = Model(
        m,
        name="ILSP",
        description="Inner-loop subproblem",
        equations=eqns,
        problem='MIP',
        sense='min',
        objective=min_op_cost_y,
    )
    return ILSP_model

## ADA-based initialization of the inner loop
# First linear problem OF and constraints
con_7b = Equation(m, name="con_7b", domain=[g])
con_7c = Equation(m, name="con_7c", domain=[g])
con_7d = Equation(m, name="con_7d")
con_7e = Equation(m, name="con_7e", domain =[k])
con_7e_ess = Equation(m, name="con_7e_ess", domain =[k])
lp1_obj_var = Equation(m, name="lp1_obj_var")
# Constraints 5d-5z
con_5di = Equation(m, name="con_5di", domain=[g, t, k])
con_5ei = Equation(m, name="con_5ei", domain=[g, t, h, k])
con_5fi = Equation(m, name="con_5fi", domain=[g, t, k])
con_5gi = Equation(m, name="con_5gi", domain=[d, t, h, k])
con_5hi = Equation(m, name="con_5hi", domain=[r, t, h, k])
con_5ii = Equation(m, name="con_5ii", domain=[le, t, h, k])
con_5ji = Equation(m, name="con_5ji", domain=[lc, t, h, k])
con_5ki = Equation(m, name="con_5ki", domain=[s, t, k])
con_5li = Equation(m, name="con_5li", domain=[s, t, h, k])
con_5mi = Equation(m, name="con_5mi", domain=[s, t, k])
con_5ni = Equation(m, name="con_5ni", domain=[s, t, h, k])
con_5oi = Equation(m, name="con_5oi", domain=[n,t,h,k]) # N =/= ref bus
con_5pi = Equation(m, name="con_5pi", domain=[n,t,h,k]) # N == ref bus
con_5qi = Equation(m, name="con_5qi", domain=[s,t,k])
con_5ri = Equation(m, name="con_5ri", domain=[s,t,h,k])
con_5si = Equation(m, name="con_5si", domain=[s,t,k])
con_5ti = Equation(m, name="con_5ti", domain=[s,t,k])

def build_lp1_eqns(yi, v_range, ess_inv):
    vmin = min(v_range)
    vmax = max(v_range)
    # vr = (vr.val >= vmin) & (vr.val <= vmax)
    kr = (k.val >= vmin) & (k.val <= vmax)
    hmax = int(nb_H.toValue())

    lp1_obj_var[...] = xiP == xiP_y[yi]
    con_7b[g] = cG_gy[g, yi] == CG_g_fc[g] * (1 + zetaGC_g_fc)**(yi - 1) + (CG_g_max[g] * (1 + zetaGC_g_max)**(yi - 1)) * aGC_gy[g, yi]
    con_7c[g] = aGC_gy[g, yi] <= 1
    con_7d[...] = Sum(g, aGC_gy[g, yi]) <= GammaGC

    con_7e[va] = xiP_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h]*PD_dyo[d,yi]*Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,va])) \
    - Sum(l, PL_l[l]*(muL_lythv_lo[l,yi,t,h,va] + muL_lythv_up[l,yi,t,h,va])) \
    - Sum(s, US_sythv[s,yi,t,h,va]*PSC_s[s]*muSC_sythv_up[s,yi,t,h,va] + (1-US_sythv[s,yi,t,h,va])*PSD_s[s]*muSD_sythv_up[s,yi,t,h,va] - ES_s_min[s]*muS_sythv_lo[s,yi,t,h,va] + ES_s_max[s]*muS_sythv_up[s,yi,t,h,va])\
    + Sum(g, UG_gythv[g,yi,t,h,va]*(PG_g_min[g]*muG_gythv_lo[g,yi,t,h,va] - PG_gyo[g,yi] * muG_gythv_up[g,yi,t,h,va])) \
    - Sum(r, gammaR_ryth[r,yi,t,h]*PR_ryo[r,yi]*(muR_rythv_up[r,yi,t,h,va] - sigma_yt[yi,t]*tau_yth[yi,t,h] * CR_r[r])) \
    - Sum(d, gammaD_dyth[d,yi,t,h]*PD_dyo[d,yi]*muD_dythv_up[d,yi,t,h,va]))\
    + Sum(s, ES_syt0[s,yi,t]*(PhiS_sytv[s,yi,t,va] + PhiS_sytv_lo[s,yi,t,va])) \
    - Sum(h.where[Ord(h)>1], Sum(g, RGD_g[g]*muGD_gythv[g,yi,t,h,va] + RGU_g[g]*muGU_gythv[g,yi,t,h,va])))
    con_7e_ess[va] = xiP_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d,yi,t,h]*PD_dyo[d,yi]*Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,va])) \
    - Sum(l, PL_l[l]*(muL_lythv_lo[l,yi,t,h,va] + muL_lythv_up[l,yi,t,h,va])) \
    - Sum(s, VS_syj_prev[s,yi]*(US_sythv[s,yi,t,h,va]*PSC_s[s]*muSC_sythv_up[s,yi,t,h,va] + (1-US_sythv[s,yi,t,h,va])*PSD_s[s]*muSD_sythv_up[s,yi,t,h,va] - ES_s_min[s]*muS_sythv_lo[s,yi,t,h,va] + ES_s_max[s]*muS_sythv_up[s,yi,t,h,va]))\
    + Sum(g, UG_gythv[g,yi,t,h,va]*(PG_g_min[g]*muG_gythv_lo[g,yi,t,h,va] - PG_gyo[g,yi] * muG_gythv_up[g,yi,t,h,va])) \
    - Sum(r, gammaR_ryth[r,yi,t,h]*PR_ryo[r,yi]*(muR_rythv_up[r,yi,t,h,va] - sigma_yt[yi,t]*tau_yth[yi,t,h] * CR_r[r])) \
    - Sum(d, gammaD_dyth[d,yi,t,h]*PD_dyo[d,yi]*muD_dythv_up[d,yi,t,h,va]))\
    + Sum(s, VS_syj_prev[s,yi]*ES_syt0[s,yi,t]*(PhiS_syt0v[s,yi,t,va] + PhiS_sytv_lo[s,yi,t,va])) \
    - Sum(h.where[Ord(h)>1], Sum(g, RGD_g[g]*muGD_gythv[g,yi,t,h,va] + RGU_g[g]*muGU_gythv[g,yi,t,h,va])))

    con_5di[g,t,va] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,1,va]) + muG_gythv_lo[g,yi,t,1,va]\
    - muG_gythv_up[g,yi,t,1,va] - muGD_gythv[g,yi,t,2,va] + muGU_gythv[g,yi,t,2,va]\
    == sigma_yt[yi,t] * tau_yth[yi, t, 1] * cG_gy[g, yi]
    con_5ei[g,t,h,va].where[(Ord(h)!=1) & (Ord(h)!=Card(h))] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,h,va])\
    + muG_gythv_lo[g,yi,t,h,va] - muG_gythv_up[g,yi,t,h,va] + muGD_gythv[g,yi,t,h,va] - muGD_gythv[g,yi,t,h.lead(1),va]\
    - muGU_gythv[g,yi,t,h,va] + muGU_gythv[g,yi,t,h.lead(1),va] == sigma_yt[yi,t]*tau_yth[yi,t,h]*cG_gy[g, yi]
    con_5fi[g,t,va] = Sum(n.where[g_n[g,n]], lambdaN_nythv[n,yi,t,hmax,va]) +  muG_gythv_lo[g,yi,t,hmax,va]\
    - muG_gythv_up[g,yi,t,hmax,va] + muGD_gythv[g,yi,t,hmax,va] - muGU_gythv[g, yi, t, hmax, va]\
    == sigma_yt[yi, t] * tau_yth[yi, t, hmax] * cG_gy[g, yi]
    con_5gi[d,t,h,va] = Sum(n.where[d_n[d,n]], lambdaN_nythv[n,yi,t,h,va])-muD_dythv_up[d, yi, t, h, va]\
    <= sigma_yt[yi, t] * tau_yth[yi, t, h] * CLS_d[d]
    con_5hi[r,t,h,va] = Sum(n.where[r_n[r,n]], lambdaN_nythv[n,yi,t,h,va]) - muR_rythv_up[r,yi,t,h,va]\
    <= -sigma_yt[yi, t] * tau_yth[yi, t, h] * CR_r[r]
    con_5ii[le,t,h,va] = Sum(n.where[rel_n[le,n]], lambdaN_nythv[n,yi,t,h,va]) - Sum(n.where[sel_n[le,n]], lambdaN_nythv[n,yi,t,h,va]) \
    + muL_lythv_exist[le, yi, t, h, va] + muL_lythv_lo[le, yi, t, h, va] - muL_lythv_up[le, yi, t, h, va] == 0
    con_5ji[lc,t,h,va] = Sum(n.where[rel_n[lc,n]], lambdaN_nythv[n,yi,t,h,va]) - Sum(n.where[sel_n[lc,n]], lambdaN_nythv[n,yi,t,h,va]) \
    + muL_lythv_can[lc,yi,t,h,va] + muL_lythv_lo[lc,yi,t,h,va] - muL_lythv_up[lc,yi,t,h,va] == 0
    con_5ki[s,t,va] = Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,1,va])\
    + (tau_yth[yi,t,1]/etaSD_s[s])*PhiS_sytv[s,yi,t,va] - muSD_sythv_up[s,yi,t,1,va] <= 0
    con_5li[s,t,h,va].where[Ord(h)>1] = Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,h,va])\
    + (tau_yth[yi,t,h]/etaSD_s[s])*muS_sythv[s,yi,t,h,va] - muSD_sythv_up[s,yi,t,h,va] <= 0
    con_5mi[s,t,va] = -Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,1,va])\
    - etaSC_s[s]*tau_yth[yi,t,1]*PhiS_sytv[s,yi,t,va] - muSC_sythv_up[s,yi,t,1,va] <= 0
    con_5ni[s,t,h,va].where[Ord(h)>1] = -Sum(n.where[s_n[s,n]], lambdaN_nythv[n,yi,t,h,va])\
    - etaSC_s[s]*tau_yth[yi,t,h]*muS_sythv[s,yi,t,h,va] - muSC_sythv_up[s,yi,t,h,va] <= 0
    con_5oi[n, t, h, va].where[Ord(n) > 1] = -Sum(le.where[sel_n[le, n]], muL_lythv_exist[le, yi, t, h, va] / X_l[le])\
    + Sum(le.where[rel_n[le, n]], muL_lythv_exist[le, yi, t, h, va] / X_l[le])\
    - Sum(lc.where[sel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, va])\
    + Sum(lc.where[rel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, va]) == 0
    con_5pi[n, t, h, va].where[Ord(n) == 1] = -Sum(le.where[sel_n[le, n]], muL_lythv_exist[le, yi, t, h, va] / X_l[le])\
    + Sum(le.where[rel_n[le, n]], muL_lythv_exist[le, yi, t, h, va] / X_l[le])\
    - Sum(lc.where[sel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, va])\
    + Sum(lc.where[rel_n[lc, n]], (VL_lyj_prev[lc, yi] / X_l[lc]) * muL_lythv_can[lc, yi, t, h, va])\
    + phiN_nythv[n, yi, t, h, va] == 0
    con_5qi[s,t,va] = PhiS_sytv[s,yi,t,va] - muS_sythv[s,yi,t,2,va] + muS_sythv_lo[s,yi,t,1,va] - muS_sythv_up[s,yi,t,1,va] == 0
    con_5ri[s,t,h,va].where[(Ord(h)!=1) & (Ord(h)!=Card(h))] = muS_sythv[s,yi,t,h,va] - muS_sythv[s,yi,t,h.lead(1),va] + muS_sythv_lo[s,yi,t,h,va] - muS_sythv_up[s,yi,t,h,va] == 0
    con_5si[s,t,va] = muS_sythv[s,yi,t,hmax,va] + PhiS_sytv_lo[s,yi,t,va] + muS_sythv_lo[s,yi,t,hmax,va] - muS_sythv_up[s,yi,t,hmax,va] == 0
    con_5ti[s,t,va] = PhiS_syt0v[s,yi,t,va] - PhiS_sytv[s,yi,t,va] + muS_sythv_lo[s,yi,t,1,va] - muS_sythv_up[s,yi,t,1,va] == 0
    # con_5y1i[g, t, h, va].where[(Ord(h) > 1)] = muGD_gythv[g, yi, t, h, va] >= 0
    # con_5y2i[g, t, h, va].where[(Ord(h) > 1)] = muGU_gythv[g, yi, t, h, va] >= 0

    lp1_eqns = [lp1_obj_var, con_7b, con_7c, con_7d, con_7e, con_5di, con_5ei, con_5fi, con_5gi, con_5hi, con_5ii, con_5ji, con_5ki, con_5li, con_5mi,
            con_5ni, con_5oi, con_5pi, con_5qi, con_5ri, con_5si]
    lp1_eqns_ess = [lp1_obj_var, con_7b, con_7c, con_7d, con_7e_ess, con_5di, con_5ei, con_5fi, con_5gi, con_5hi, con_5ii,
                con_5ji, con_5ki, con_5li, con_5mi, con_5ni, con_5oi, con_5pi, con_5qi, con_5ri, con_5si, con_5ti]
    eqns = lp1_eqns_ess if ess_inv else lp1_eqns
    LP1_model = Model(
        m,
        name="lp1",
        description="Fist linear problem (ADA)",
        equations=eqns,
        problem='LP',
        sense='max',
        objective=xiP,
    )
    return LP1_model

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
con_8l = Equation(m, name="con_8l", domain=[k])
con_8l_ess = Equation(m, name="con_8l_ess", domain=[k])
lp2_obj_var = Equation(m, name="lp2_obj_var")

def build_lp2_eqns(yi, v_range, ess_inv):
    vmin = min(v_range)
    vmax = max(v_range)
    kr = (k.val >= vmin) & (k.val <= vmax)

    lp2_obj_var[...] = xiQ == xiQ_y[yi]
    con_8b[d] = pD_dy[d, yi] == PD_d_fc[d] * (1 + zetaD_d_fc[d])**(yi - 1) + (PD_d_max[d] * (1 + zetaD_d_max[d])**(yi - 1)) * aD_dy[d, yi]
    con_8c[g] = pG_gy[g, yi] == PG_g_fc[g] * (1 - zetaGP_g_fc[g])**(yi - 1) - (PG_g_max[g] * (1 + zetaGP_g_max[g])**(yi - 1)) * aGP_gy[g, yi]
    con_8d[r] = pR_ry[r, yi] == PR_r_fc[r] * (1 + zetaR_r_fc[r])**(yi - 1) - (PR_r_max[r] * (1 + zetaR_r_max[r])**(yi - 1)) * aR_ry[r, yi]
    con_8e[d] = aD_dy[d, yi] <= 1
    con_8f[g] = aGP_gy[g, yi] <= 1
    con_8g[r] = aR_ry[r, yi] <= 1
    con_8h[...] = Sum(d, aD_dy[d, yi]) <= GammaD
    con_8i[...] = Sum(g, aGP_gy[g, yi]) <= GammaGP
    con_8j[...] = Sum(rs, aR_ry[rs, yi]) <= GammaRS
    con_8k[...] = Sum(rw, aR_ry[rw, yi]) <= GammaRW
    con_8l[va] = xiQ_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d, yi, t, h] * pD_dy[d, yi] * Sum(n.where[d_n[d, n]], LambdaN_nythvo[n, yi, t, h, va]))\
        - Sum(l, PL_l[l] * (muL_lythvo_lo[l, yi, t, h, va] + muL_lythvo_up[l, yi, t, h, va])) - Sum(s, US_sythv[s, yi, t, h, va] * PSC_s[s] * muSC_sythvo_up[s, yi, t, h, va]\
        + (1 - US_sythv[s, yi, t, h, va]) * PSD_s[s] * muSD_sythvo_up[s, yi, t, h, va] - ES_s_min[s] * muS_sythvo_lo[s, yi, t, h, va]\
        + ES_s_max[s] * muS_sythvo_up[s, yi, t, h, va]) + Sum(g, UG_gythv[g, yi, t, h, va] * (PG_g_min[g] * muG_gythvo_lo[g, yi, t, h, va] - pG_gy[g, yi] * muG_gythvo_up[g, yi, t, h, va]))\
        - Sum(r, gammaR_ryth[r, yi, t, h] * pR_ry[r, yi] * (muR_rythvo_up[r, yi, t, h, va] - sigma_yt[yi, t] * tau_yth[yi, t, h] * CR_r[r]))\
        - Sum(d, gammaD_dyth[d, yi, t, h] * pD_dy[d, yi] * muD_dythvo_up[d, yi, t, h, va])) + Sum(s, ES_syt0[s, yi, t] * (PhiS_sytvo[s, yi, t, va] + PhiS_sytvo_lo[s, yi, t, va]))\
        - Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g] * muGD_gythvo[g, yi, t, h, va] + RGU_g[g] * muGU_gythvo[g, yi, t, h, va])))
    con_8l_ess[va] = xiQ_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d, yi, t, h] * pD_dy[d, yi] * Sum(n.where[d_n[d, n]], LambdaN_nythvo[n, yi, t, h, va]))\
        - Sum(l, PL_l[l] * (muL_lythvo_lo[l, yi, t, h, va] + muL_lythvo_up[l, yi, t, h, va])) - Sum(s, VS_syj_prev[s,yi]*(US_sythv[s, yi, t, h, va] * PSC_s[s] * muSC_sythvo_up[s, yi, t, h, va]\
        + (1 - US_sythv[s, yi, t, h, va]) * PSD_s[s] * muSD_sythvo_up[s, yi, t, h, va] - ES_s_min[s] * muS_sythvo_lo[s, yi, t, h, va]\
        + ES_s_max[s] * muS_sythvo_up[s, yi, t, h, va])) + Sum(g, UG_gythv[g, yi, t, h, va] * (PG_g_min[g] * muG_gythvo_lo[g, yi, t, h, va] - pG_gy[g, yi] * muG_gythvo_up[g, yi, t, h, va]))\
        - Sum(r, gammaR_ryth[r, yi, t, h] * pR_ry[r, yi] * (muR_rythvo_up[r, yi, t, h, va] - sigma_yt[yi, t] * tau_yth[yi, t, h] * CR_r[r]))\
        - Sum(d, gammaD_dyth[d, yi, t, h] * pD_dy[d, yi] * muD_dythvo_up[d, yi, t, h, va])) + Sum(s, VS_syj_prev[s,yi]*ES_syt0[s, yi, t] * (PhiS_syt0v[s, yi, t, va] + PhiS_sytvo_lo[s, yi, t, va]))\
        - Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g] * muGD_gythvo[g, yi, t, h, va] + RGU_g[g] * muGU_gythvo[g, yi, t, h, va])))

    # con_8l[k].where[kr] = xiQ_y[yi] <= Sum(t, Sum(h, Sum(d, gammaD_dyth[d, yi, t, h] * pD_dy[d, yi] * Sum(n.where[d_n[d, n]], LambdaN_nythvo[n, yi, t, h, k]))\
    #     - Sum(l, PL_l[l] * (muL_lythvo_lo[l, yi, t, h, k] + muL_lythvo_up[l, yi, t, h, k])) - Sum(s, US_sythv[s, yi, t, h, k] * PSC_s[s] * muSC_sythvo_up[s, yi, t, h, k]\
    #     + (1 - US_sythv[s, yi, t, h, k]) * PSD_s[s] * muSD_sythvo_up[s, yi, t, h, k] - ES_s_min[s] * muS_sythvo_lo[s, yi, t, h, k]\
    #     + ES_s_max[s] * muS_sythvo_up[s, yi, t, h, k]) + Sum(g, UG_gythv[g, yi, t, h, k] * (PG_g_min[g] * muG_gythvo_lo[g, yi, t, h, k] - pG_gy[g, yi] * muG_gythvo_up[g, yi, t, h, k]))\
    #     - Sum(r, gammaR_ryth[r, yi, t, h] * pR_ry[r, yi] * (muR_rythvo_up[r, yi, t, h, k] - sigma_yt[yi, t] * tau_yth[yi, t, h] * CR_r[r]))\
    #     - Sum(d, gammaD_dyth[d, yi, t, h] * pD_dy[d, yi] * muD_dythvo_up[d, yi, t, h, k])) + Sum(s, ES_syt0[s, yi, t] * (PhiS_sytvo[s, yi, t, k] + PhiS_sytvo_lo[s, yi, t, k]))\
    #     - Sum(h.where[Ord(h) > 1], Sum(g, RGD_g[g] * muGD_gythvo[g, yi, t, h, k] + RGU_g[g] * muGU_gythvo[g, yi, t, h, k])))

    lp2_eqns = [lp2_obj_var, con_8b, con_8c, con_8d, con_8e, con_8f, con_8g, con_8h, con_8i, con_8j, con_8k, con_8l]
    lp2_eqns_ess = [lp2_obj_var, con_8b, con_8c, con_8d, con_8e, con_8f, con_8g, con_8h, con_8i, con_8j, con_8k, con_8l_ess]
    eqns = lp2_eqns_ess if ess_inv else lp2_eqns
    LP2_model = Model(
        m,
        name="lp2",
        description="Second linear problem (ADA)",
        equations=eqns,
        problem='LP',
        sense='max',
        objective=xiQ,
    )
    return LP2_model

# Set values of the uncertain parameters for the given outer loop iteration
def set_uncertain_params_olmp(j_iter):
    # At the first iteration, uncertain parameters equal their forecast values
    if j_iter == 1:
        CG_gyi[g,y,j_iter] = CG_g_fc[g]
        PD_dyi[d,y,j_iter] = PD_d_fc[d]
        PG_gyi[g,y,j_iter] = PG_g_fc[g]
        PR_ryi[r,y,j_iter] = PR_r_fc[r]
    else:
        CG_gyi[g, y, j_iter] = cG_gy.l[g,y]
        PD_dyi[d, y, j_iter] = pD_dy.l[d,y]
        PG_gyi[g, y, j_iter] = pG_gy.l[g,y]
        PR_ryi[r, y, j_iter] = pR_ry.l[r,y]

# Set values of the uncertain parameters for the given inner loop iteration
def set_uncertain_params_ilsp(k_iter, is_ada):
    # At the first iteration, uncertain parameters equal their forecast values
    if is_ada and k_iter == 1:
        CG_gyk[g,y] = CG_g_fc[g]
        PD_dyk[d,y] = PD_d_fc[d]
        PG_gyk[g,y] = PG_g_fc[g]
        PR_ryk[r,y] = PR_r_fc[r]
    else:
        CG_gyk[g,y] = cG_gy.l[g,y]
        PD_dyk[d,y] = pD_dy.l[d,y]
        PG_gyk[g,y] = pG_gy.l[g,y]
        PR_ryk[r,y] = pR_ry.l[r,y]

# Solve the relaxed outer-loop master problem
def solve_olmp_relaxed(j_iter, lb_o, ess_inv):
    ro = 1 # Initialize relaxed iteration counter
    # Solve at least once, until ro == j
    while ro <= j_iter:
        # Determine the subset i as a function of j and ro
        i_range = list(range(j_iter - ro + 1, j_iter + 1))
        ir.setRecords(i_range)
        # Solve the outer-loop master problem
        OLMP_model = build_olmp_eqns(ess_inv, i_range) # Rebuild the olmp equations to account for the change in set i
        OLMP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_olmp.txt"),output=sys.stdout)
        if OLMP_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible', 'InfeasibleNoSolution']:
            raise RuntimeError('OLMP is infeasible at j = {}'.format(j_iter))
        VL_lyj[lc,y] = vL_ly.l[lc,y]
        VL_lyj_prev[lc,y] = vL_ly_prev.l[lc,y]
        VS_syj_prev[sc,y] = vS_sy_prev.l[sc,y]
        olmp_ov = OLMP_model.objective_value
        # Exit if ro == j or if optimal value exceeds lb_o, else increment ro and iterate again
        if ro == j_iter or olmp_ov > lb_o:
            logger.info("Relaxed OLMP iteration (ro = {}) equals outer-loop iteration (j = {}) or LBO has increased --> Exit OLMP".format(ro, j_iter))
            break
        else:
            ro += 1

    return olmp_ov

# Solve the inner-loop subproblem
def solve_ilsp(ess_inv, y_iter, j_iter, k_iter):
    ILSP_model = build_ilsp_eqns(ess_inv, y_iter, j_iter) # Rebuild the ilsp equations for the given year and outer loop iteration j
    ILSP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_ilsp.txt"),output=sys.stdout)
    if ILSP_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible', 'InfeasibleNoSolution']:
        raise RuntimeError('ILSP is infeasible at y = {}, j = {}, k = {}'.format(y_iter, j_iter, k_iter))
    UG_gythv[g,y,t,h,k_iter] = uG_gythi.l[g,y,t,h,j_iter]
    US_sythv[s,y,t,h,k_iter] = uS_sythi.l[s,y,t,h,j_iter]
    ilsp_ov = ILSP_model.objective_value

    return ilsp_ov

# Solve the inner-loop master problem by using ADA
def solve_ilmp_ada(y_iter, j_iter, k_iter, tol):
    v_range = list(range(1, k_iter + 1))
    va.setRecords(v_range)
    # Set binary decision variables to the last solved value for the given inner loop iteration
    ada_ov = 0
    o_iter = 1
    for ada_iter in range(5):
        if o_iter == 1:
            PD_dyo[d,y] = PD_d_fc[d]
            PG_gyo[g,y] = PG_g_fc[g]
            PR_ryo[r,y] = PR_r_fc[r]
        LP1_model = build_lp1_eqns(y_iter, v_range, ess_inv)
        LP1_model.solve(options=Options(relative_optimality_gap=tol, lp="CPLEX", savepoint=1, log_file="log_lp1.txt"),output=sys.stdout)
        if LP1_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible', 'InfeasibleNoSolution']:
            raise RuntimeError('LP1 is infeasible at y = {}, j = {}, k = {}'.format(y_iter, j_iter, k_iter))
        LambdaN_nythvo[n,y,t,h,k] = lambdaN_nythv.l[n,y,t,h,k]
        muD_dythvo_up[d,y,t,h,k] = muD_dythv_up.l[d,y,t,h,k]
        muG_gythvo_lo[g,y,t,h,k] = muG_gythv_lo.l[g,y,t,h,k]
        muG_gythvo_up[g,y,t,h,k] = muG_gythv_up.l[g,y,t,h,k]
        muGD_gythvo[g,y,t,h,k] = muGD_gythv.l[g,y,t,h,k]
        muGU_gythvo[g,y,t,h,k] = muGU_gythv.l[g,y,t,h,k]
        muL_lythvo_lo[l,y,t,h,k] = muL_lythv_lo.l[l,y,t,h,k]
        muL_lythvo_up[l,y,t,h,k] = muL_lythv_up.l[l,y,t,h,k]
        muR_rythvo_up[r,y,t,h,k] = muR_rythv_up.l[r,y,t,h,k]
        muS_sythvo_lo[s,y,t,h,k] = muS_sythv_lo.l[s,y,t,h,k]
        muS_sythvo_up[s,y,t,h,k] = muS_sythv_up.l[s,y,t,h,k]
        muSC_sythvo_up[s,y,t,h,k] = muSC_sythv_up.l[s,y,t,h,k]
        muSD_sythvo_up[s,y,t,h,k] = muSD_sythv_up.l[s,y,t,h,k]
        PhiS_sytvo[s,y,t,k] = PhiS_sytv.l[s,y,t,k]
        PhiS_sytvo_lo[s,y,t,k] = PhiS_sytv_lo.l[s,y,t,k]
        lp1_ov = LP1_model.objective_value

        LP2_model = build_lp2_eqns(y_iter, v_range, ess_inv)
        LP2_model.solve(options=Options(relative_optimality_gap=tol, lp="CPLEX", savepoint=1, log_file="log_lp2.txt"),output=sys.stdout)
        if LP2_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible', 'InfeasibleNoSolution']:
            raise RuntimeError('LP2 is infeasible at y = {}, j ={}, k = {}'.format(y_iter, j_iter, k_iter))
        PD_dyo[d,y] = pD_dy.l[d,y]
        PG_gyo[g,y] = pG_gy.l[g,y]
        PR_ryo[r,y] = pR_ry.l[r,y]
        lp2_ov = LP2_model.objective_value

        # logger.info("LP1 = {} and LP2 = {} before computing ADA inner loop error.".format(lp1_ov, lp2_ov))
        if (abs(lp1_ov - lp2_ov) / min(lp1_ov, lp2_ov)) < tol:
            ada_ov = min(lp1_ov, lp2_ov)
            logger.info("ADA ILMP converged in o = {} iteration(s)".format(o_iter))
            break
        else:
            o_iter += 1
            if o_iter == max(range(5+1)):
                logger.info("ADA ILMP did not converge in max number of iterations")

    return ada_ov

# Solve the relaxed inner-loop master problem
def solve_ilmp_relaxed(y_iter, j_iter, k_iter, ub_i):
    ri = 1 # Initialize relaxed iteration counter
    # Solve at least once, until ri == k
    cG_solved = None
    pD_solved = None
    pG_solved = None
    pR_solved = None
    while ri <= k_iter:
        # Determine the subset v as a function of k and ri
        v_range = list(range(k_iter - ri + 1, k_iter + 1))
        vr.setRecords(v_range)
        logger.info('v_range = {}'.format(v_range))
        # Solve the inner-loop master problem
        ILMP_model = build_ilmp_eqns(y_iter, v_range, ess_inv) # Rebuild the ilmp equations to account for the change in set v
        ILMP_model.solve(options=Options(relative_optimality_gap=tol, mip="CPLEX", savepoint=1, log_file="log_ilmp.txt"),output=sys.stdout)
        if ILMP_model.status.name in ['InfeasibleGlobal', 'InfeasibleLocal', 'InfeasibleIntermed', 'IntegerInfeasible', 'InfeasibleNoSolution']:
            raise RuntimeError('ILMP is infeasible at y = {}, j = {}, k = {}'.format(y_iter, j_iter, k_iter))
        ilmp_ov = ILMP_model.objective_value
        # Exit if ri == k or if optimal value is less than ub_i, else increment ri and iterate again
        if ri == k_iter or ilmp_ov < ub_i:
            logger.info("Relaxed ILMP iteration (ri = {}) equals inner-loop iteration (k = {}) or UBI has decreased --> Exit ILMP".format(ri, k_iter))
            break
        # elif cG_gy.l.records.equals(cG_solved) and pD_dy.l.records.equals(pD_solved) and pG_gy.l.records.equals(pG_solved) and pR_ry.l.records.equals(pR_solved):
        #     logger.info("Relaxed ILMP iteration (ri = {}): no change in solution --> Exit ILMP".format(ri, k_iter))
        #     break
        else:
            ri += 1
            cG_solved = cG_gy.l.records
            pD_solved = pD_dy.l.records
            pG_solved = pG_gy.l.records
            pR_solved = pR_ry.l.records

    return ilmp_ov

def  compute_worst_case_total_cost(ess_inv):
    vL_vals = vL_ly.l.records
    xi_y_vals = xi_y.records
    IL_vals = IL_l.records
    IL_vals.columns = ['lc', 'value']
    if ess_inv:
        vS_vals = vS_sy.l.records
        IS_vals = IS_s.records
        IS_vals.columns = ['sc', 'value']
    cost = 0
    for year in years_data:
        # try:
        sum_line_cost = 0
        if vL_vals is not None:
            vL_vals_year = vL_vals[vL_vals['y'] == str(year)]
            vL_IL = pd.merge(vL_vals_year, IL_vals, on='lc')
            vL_IL['line_cost'] = vL_IL['level'] * vL_IL['value']
            sum_line_cost = vL_IL['line_cost'].sum()
        sum_ess_cost = 0
        if ess_inv and vS_vals is not None:
            vS_vals_year = vS_vals[vS_vals['y'] == str(year)]
            vS_IS = pd.merge(vS_vals_year, IS_vals, on='sc')
            vS_IS['ess_cost'] = vS_IS['level'] * vS_IS['value']
            sum_ess_cost = vS_IS['ess_cost'].sum()
        xi_y_year = xi_y_vals[xi_y_vals['y'] == str(year)]['level'].values[0]
        logger.info('sum_line_cost = {}'.format(sum_line_cost))
        logger.info('sum_ess_cost = {}'.format(sum_ess_cost))
        logger.info('xi_y_year = {}'.format(xi_y_year))
        cost += (1/((1+kappa.toValue())**(year-1))) * (((xi_y_year+0.04*sum_ess_cost)/(1+kappa.toValue())) + sum_line_cost + sum_ess_cost)
        # except:
        #     xi_y_year = xi_y_vals[xi_y_vals['y'] == str(year)]['level'].values[0]
        #     cost += (1/((1+kappa.toValue())**(year-1))) * ((xi_y_year/(1+kappa.toValue())))

    return cost

# SOLUTION PROCEDURE #
lb_o = -999999999999
ub_o = 999999999999
VL_lyjm1_rec = None
VS_syjm1_rec = None
VL_lyjm1_prev_rec = None
j_max = 5
j.setRecords(list(range(1, j_max+1)))
k_max = 5
k.setRecords(list(range(1, k_max+1)))
# OUTER LOOP #
j_iter = 1
for ol_iter in range(j_max):
    logger.info("Starting outer loop problem for j = {}".format(j_iter))
    set_uncertain_params_olmp(j_iter)
    lb_o = solve_olmp_relaxed(j_iter, lb_o, ess_inv)
    if j_iter > 1:
        if vL_ly.l.records is not None and vS_sy.l.records is not None:
            if vL_ly.l.records.equals(VL_lyjm1_rec) and vS_sy.l.records.equals(VS_syjm1_rec):
                logger.info("No change in investment decision variables --> End outer loop")
                break
        elif vL_ly.l.records is None and VL_lyjm1_rec is None and vS_sy.l.records is None and VS_syjm1_rec is None:
            logger.info("No change in investment decision variables (zero investment) --> End outer loop")
            break
    # YEAR LOOP
    for y_iter in years_data:
        logger.info("Starting inner loop problems for y = {}".format(y_iter))
        # INNER LOOP: ILSP + ADA ILMP #
        lb_i_ada = -999999999999
        ub_i_ada = 999999999999
        k_iter_ada = 1
        logger.info("Starting first inner loop (ADA) for y = {}".format(y_iter))
        for il_ada_iter in range(k_max):
            logger.info("Starting ADA inner loop iteration k = {}".format(k_iter_ada))
            set_uncertain_params_ilsp(k_iter_ada, is_ada=True)
            lb_i_ada = solve_ilsp(ess_inv, y_iter, j_iter, k_iter_ada)
            logger.info("LBI = {} and UBI = {} before computing ADA inner loop error.".format(lb_i_ada, ub_i_ada))
            il_error_ada = (ub_i_ada - lb_i_ada) / lb_i_ada
            if il_error_ada < tol:
                logger.info("First inner loop (ADA) has converged after k = {} iterations --> End ADA inner loop".format(k_iter_ada))
                break
            elif il_error_ada >= tol:
                logger.info("First inner loop (ADA) has not converged after k = {} iterations --> Solve ADA ILMP".format(k_iter_ada))
                ub_i_ada = solve_ilmp_ada(y_iter, j_iter, k_iter_ada, tol)
                k_iter_ada += 1
        # INNER LOOP: ILSP + relaxed ILMP #
        lb_i_rel = -999999999999
        ub_i_rel = 999999999999
        k_iter_rel = 1
        cG_solved = None
        pD_solved = None
        pG_solved = None
        pR_solved = None
        logger.info("Starting second inner loop (relaxed) for y = {}".format(y_iter))
        for il_rel_iter in range(k_max):
            logger.info("Starting relaxed inner loop iteration  k = {}".format(k_iter_rel))
            set_uncertain_params_ilsp(k_iter_rel, is_ada=False)
            lb_i_rel = solve_ilsp(ess_inv, y_iter, j_iter, k_iter_rel)
            logger.info("LBI = {} and UBI = {} before computing relaxed inner loop error.".format(lb_i_rel, ub_i_rel))
            il_error_rel = (ub_i_rel - lb_i_rel) / lb_i_rel
            if il_error_rel < tol:
                logger.info("Second inner loop (relaxed) has converged after k = {} iterations --> End relaxed inner loop".format(k_iter_rel))
                break
            elif il_error_rel >= tol:
                logger.info("Second inner loop (relaxed) has not converged after k = {} iterations --> Solve relaxed ILMP".format(k_iter_rel))
                ub_i_rel = solve_ilmp_relaxed(y_iter, j_iter, k_iter_rel, ub_i_rel)
                # if k_iter_rel > 1 and cG_gy.l.records.equals(cG_solved) and pD_dy.l.records.equals(pD_solved) and pG_gy.l.records.equals(pG_solved) and pR_ry.l.records.equals(pR_solved):
                #     logger.info("Second inner loop (relaxed): no change in solution after k = {} iterations-->  End relaxed inner loop".format(k_iter_rel))
                #     break
                k_iter_rel += 1
                cG_solved = cG_gy.l.records
                pD_solved = pD_dy.l.records
                pG_solved = pG_gy.l.records
                pR_solved = pR_ry.l.records
        if y_iter == max(years_data):
            logger.info("Reached end of last year (y = {}) in the planning horizon --> End year loop".format(y_iter))
            break
        else:
            y_iter += 1

    # Update ub_o
    wc_cost = compute_worst_case_total_cost(ess_inv)
    ub_o = wc_cost
    logger.info("LBO = {} and UBO = {} before computing outer loop error.".format(lb_o, ub_o))
    print("Total worst-case cost = {}".format(ub_o))
    ol_error = (ub_o - lb_o) / lb_o
    if ol_error < tol:
        logger.info("Outer loop has converged after j = {} iterations --> End problem".format(j_iter))
        break
    else:
        j_iter += 1
        VL_lyjm1_rec = vL_ly.l.records
        VS_syjm1_rec = vS_sy.l.records
print(min_inv_cost_wc.records)
print(wc_cost)
print(vL_ly.l.records)
if ess_inv:
    print(vS_sy.l.records)
m.write(r'C:\Users\Kevin\OneDrive - McGill University\Research\Sandbox\optimization\multi-year_AROTNEP\results\aro_tnep_results.gdx')
