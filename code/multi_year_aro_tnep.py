from input_data_processing import weights, RD1, lines, buses, ESS, CG, RES, loads, years_data, sigma_yt_data, tau_yth_data, gamma_dyth_data, gamma_ryth_data, ES_syt0_data
from gamspy import Alias, Container, Domain, Equation, Model, Options, Ord, Card, Parameter, Set, Smax, Sum, Variable
from gamspy.math import power
import sys

def reformat_df(dataframe):
    return dataframe.reset_index().melt(id_vars="index", var_name="Category", value_name="Value")

# Optimization problem definition
m = Container()

# SETS #
# General sets
N = Set(m, name="N", records=buses['Bus'], description="Set of buses indexed by n")
D = Set(m, name="D", records=loads['Load'], description="Set of loads indexed by d")
G = Set(m, name="G", records=CG['Generating unit'], description="Set of conventional units indexed by g")
H = Set(m, name="H", records=RD1['RTP'], description="Set of RTPs indexed by h")
L = Set(m, name="L", records=lines['Transmission line'], description="Set of transmission lines indexed by l")
R = Set(m, name="R", records=RES['Generating unit'], description="Set of renewable units indexed by r")
S = Set(m, name="S", records=ESS['Storage unit'], description="Set of storage units indexed by s")
T = Set(m, name="T", records=weights['RD'], description="Set of RDs indexed by t")
Y = Set(m, name="Y", records=years_data, description="Set of years indexed by y")
# Subsets of the general sets
L_exist = Set(m, name="L_exist", domain=[L], records=lines[lines['IL_l [$]'] == 0]['Transmission line'], description="Set of existing transmission lines")
L_can = Set(m, name="L_can", domain=[L], records=lines[lines['IL_l [$]'] > 0]['Transmission line'], description="Set of candidate transmission lines")
RS = Set(m, name="RS", domain=[R], records=RES[RES['Technology'] == 'Solar']['Generating unit'], description="Set of solar units indexed by r")
RW = Set(m, name="RW", domain=[R], records=RES[RES['Technology'] == 'Wind']['Generating unit'], description="Set of wind units indexed by r")
HM = Set(m, name="HM", domain=[H], records=RD1[RD1['middle'] == 1]['RTP'], description="Set of wind units indexed by r")

D_n = Set(m, name="D_n", domain=[D,N], records=loads[['Load', 'Bus']], description="Set of loads connected to bus n")
G_n = Set(m, name="G_n", domain=[G,N], records=CG[['Generating unit', 'Bus']], description="Set of conventional units connected to bus n")
R_n = Set(m, name="R_n", domain=[R,N], records=RES[['Generating unit', 'Bus']], description="Set of renewable units connected to bus n")
S_n = Set(m, name="S_n", domain=[S,N], records=ESS[['Storage unit', 'Bus']], description="Set of storage units connected to bus n")
RE_l = Set(m, name="RE_l", domain=[L,N], records=lines[['Transmission line', 'To bus']], description="Receiving bus of transmission line l")
SE_l = Set(m, name="SE_l", domain=[L,N], records=lines[['Transmission line', 'From bus']], description="Sending bus of transmission line l")
# ALIAS #
Yp = Alias(m, name="Yp", alias_with=Y)

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

gammaD_dyth = Parameter(m, name="gammaD_dth", domain=[D,Y,T,H], records=gamma_dyth_data, description="Demand factor of load d")
gammaR_ryth = Parameter(m, name="gammaR_rth", domain=[R,Y,T,H], records=gamma_ryth_data, description="Capacity factor of renewable unit r")
zetaD_d_fc = Parameter(m, name="zetaD_d_fc", domain=[D], records=loads[['Load', 'zetaD_d_fc']], description="Annual evolution rate of the forecast peak power consumption of load d")
zetaD_d_max = Parameter(m, name="zetaD_d_max", domain=[D], records=loads[['Load', 'zetaD_d_max']], description="Annual evolution rate of the maximum deviation from the forecast peak power consumption of load d")
zetaGC_g_fc = Parameter(m, name="zetaGC_g_fc", domain=[G], records=CG[['Generating unit', 'zetaGC_g_fc']], description="Annual evolution rate of the forecast marginal production cost of conventional generating unit g")
zetaGC_g_max = Parameter(m, name="zetaGC_g_max", domain=[G], records=CG[['Generating unit', 'zetaGC_g_max']], description="Annual evolution rate of the maximum deviation from the forecast marginal production cost of conventional generating unit g")
zetaGP_g_fc = Parameter(m, name="zetaGP_g_fc", domain=[G], records=CG[['Generating unit', 'zetaGP_g_fc']], description="Annual evolution rate of the forecast capacity of conventional generating unit g")
zetaGP_g_max = Parameter(m, name="zetaGP_g_max", domain=[G], records=CG[['Generating unit', 'zetaGP_g_max']], description="Annual evolution rate of the maximum deviation from the forecast capacity of conventional generating unit g")
zetaR_r_fc = Parameter(m, name="zetaR_r_fc", domain=[R], records=RES[['Generating unit', 'zetaR_r_fc']], description="Annual evolution rate of the forecast capacity of renewable generating unit r")
zetaR_r_max = Parameter(m, name="zetaR_r_max", domain=[R], records=RES[['Generating unit', 'zetaR_r_max']], description="Annual evolution rate of the maximum deviation from the forecast capacity of renewable generating unit r")
etaSC_s = Parameter(m, name="etaSC_s", domain=[S], records=ESS[['Storage unit', 'etaSC_s']], description="Charging efficiency of storage facility")
etaSD_s = Parameter(m, name="etaSD_s", domain=[S], records=ESS[['Storage unit', 'etaSD_s']], description="Discharging efficiency of storage facility")
sigma_yt = Parameter(m, name="sigma_yt", domain=[Y,T], records=sigma_yt_data, description="Weight of RD t")
tau_yth = Parameter(m, name="tau_yth", domain=[Y,T,H], records=tau_yth_data, description="Duration of RTP h of RD t")

CG_g_fc = Parameter(m, name="CG_g_fc", domain=[G], records=CG[['Generating unit', 'CG_g [$/MWh]']], description="Forecast marginal production cost of conventional unit g in year 1")
CG_g_max = Parameter(m, name="CG_g_max", domain=[G], records=CG[['Generating unit', 'CG_g_max']], description="Maximum deviation from the forecast marginal production cost of conventional generating unit g in year 1")
CLS_d = Parameter(m, name="CLS_d", domain=[D], records=loads[['Load', 'CLS_d [$/MWh]']], description="Load-shedding cost coefficient of load d")
CR_r = Parameter(m, name="CR_r", domain=[R], records=RES[['Generating unit', 'CR_r [$/MWh]']], description="Spillage cost coefficient of renewable unit r")
ES_syt0 = Parameter(m, name="ES_syt0", domain=[S,Y,T], records=ES_syt0_data, description="Energy initially stored of storage facility s")
ES_s_min = Parameter(m, name="ES_s_min", domain=[S], records=ESS[['Storage unit', 'ES_smin [MWh]']], description="Minimum energy level of storage facility s")
ES_s_max = Parameter(m, name="ES_s_max", domain=[S], records=ESS[['Storage unit', 'ES_smax [MWh]']], description="Maximum energy level of storage facility s")
IL_l = Parameter(m, name="IL_l", domain=[L], records=lines[['Transmission line', 'IL_l [$]']], description="Investment cost coefficient of candidate transmission line l")
PD_d_fc = Parameter(m, name="PD_d_fc", domain=[D], records=loads[['Load', 'PD_dfc [MW]']], description="Forecast peak power consumption of load d in year 1")
PD_d_max = Parameter(m, name="PD_d_max", domain=[D], records=loads[['Load', 'PD_d_max']], description="Maximum deviation from the forecast peak power consumption of load d in year 1")
PG_g_min = Parameter(m, name="PG_g_min", domain=[G], records=CG[['Generating unit', 'PG_gmin [MW]']], description="Minimum production level of conventional unit g")
PG_g_fc = Parameter(m, name="PG_g_fc", domain=[G], records=CG[['Generating unit', 'PG_gfc [MW]']], description="Forecast capacity of existing conventional unit g in year 1")
PG_g_max = Parameter(m, name="PG_g_max", domain=[G], records=CG[['Generating unit', 'PG_g_max']], description="Maximum deviation from the forecast capacity of conventional generating unit g in year 1")
PL_l = Parameter(m, name="PL_l", domain=[L], records=lines[['Transmission line', 'PL_l']], description="Power flow capacity of transmission line l")
PR_r_fc = Parameter(m, name="PR_r_fc", domain=[R], records=RES[['Generating unit', 'PR_rfc [MW]']], description="Forecast capacity of existing renewable unit r in year 1")
PR_r_max = Parameter(m, name="PR_r_max", domain=[R], records=RES[['Generating unit', 'PR_r_max']], description="Maximum deviation from the forecast capacity of renewable generating unit r in year 1")
PSC_s = Parameter(m, name="PSC_s", domain=[S], records=ESS[['Storage unit', 'PSC_smax [MW]']], description="Charging power capacity of storage facility s")
PSD_s = Parameter(m, name="PSD_s", domain=[S], records=ESS[['Storage unit', 'PSD_smax [MW]']], description="Discharging power capacity of storage facility s")
RGD_g = Parameter(m, name="RGD_g", domain=[G], records=CG[['Generating unit', 'RGD_g [MW]']], description="Ramp-down limit of conventional unit g")
RGU_g = Parameter(m, name="RGU_g", domain=[G], records=CG[['Generating unit', 'RGU_g [MW]']], description="Ramp-up limit of conventional unit")
X_l = Parameter(m, name="X_l", domain=[L], records=lines[['Transmission line', 'X_l']], description="Reactance of transmission line l")

# VARIABLES #
# Optimization variables
theta_nyth = Variable(m, name="theta_nyth", domain=[N,Y,T,H], description="Voltage angle at bus n")
xi_y = Variable(m, name='xi_y', domain=[Y], description="Auxiliary variable of the inner-loop master problem")
xiP_y = Variable(m, name='xiP_y', domain=[Y], description="Auxiliary variable of the first problem solved at each iteration of the ADA when it is applied to the inner-loop master problem")
xiQ_y = Variable(m, name='xiQ_y', domain=[Y], description="Auxiliary variable of the second problem solved at each iteration of the ADA when it is applied to the inner-loop master problem")
rho_y = Variable(m, name='rho_y', domain=[Y], description="Auxiliary variable of the outer-loop master problem")
aD_dy = Variable(m, name='aD_dy', domain=[D,Y], description="Continuous variable associated with the deviation that the peak power consumption of load d can experience from its forecast value in year y")
aGC_gy = Variable(m, name='aGC_gy', domain=[G,Y], description="Continuous variable associated with the deviation that the marginal production cost of conventional generating unit g can experience from its forecast value in year y")
aGP_gy = Variable(m, name='aGP_gy', domain=[G,Y], description="Continuous variable associated with the deviation that the capacity of conventional generating unit g can experience from its forecast value in year y")
aR_ry = Variable(m, name='aR_ry', domain=[R,Y], description="Continuous variable associated with the deviation that the capacity of renewable generating unit r can experience from its forecast value in year y")
cG_gy = Variable(m, name='cG_gy', domain=[G,Y], description="Worst-case realization of the marginal production cost of conventional generating unit g")
cO_y = Variable(m, name='cO_y', domain=[Y], description="Operating costs")
cOWC_y = Variable(m, name='cOWC_y', domain=[Y], description="Worst case operating costs")
eS_syth = Variable(m, name='eS_syth', domain=[S,Y,T,H], description="Energy stored in storage facility s")
pD_dy = Variable(m, name='pD_dy', domain=[D, Y], description="Worst-case realization of the peak power consumption of load d")
pG_gyth = Variable(m, name='pG_gyth', domain=[G,Y,T,H], description="Power produced by conventional generating unit g")
pG_gy = Variable(m, name='pG_gy', domain=[G, Y], description="Worst-case realization of the capacity of conventional generating unit g")
pL_lyth = Variable(m, name='pL_lyth', domain=[L,Y,T,H], description="Power flow through transmission line l")
pLS_dyth = Variable(m, name='pLS_dyth', domain=[D,Y,T,H], description="Unserved demand of load d")
pR_ryth = Variable(m, name='pR_ryth', domain=[R,Y,T,H], description="Power produced by renewable generating unit r")
pR_ry = Variable(m, name='pR_ry', domain=[R, Y], description="Worst-case realization of the capacity of renewable generating unit r")
pSC_syth = Variable(m, name='pSC_syth', domain=[S,Y,T,H], description="Charging power of storage facility s")
pSD_syth = Variable(m, name='pSD_syth', domain=[S,Y,T,H], description="Discharging power of storage facility s")
uG_gyth = Variable(m, name='uG_gyth', type='binary', domain=[G,Y,T,H], description="Binary variable used to model the commitment status of conventional unit g")
uS_syth = Variable(m, name='uS_syth', type='binary', domain=[S,Y,T,H], description="Binary variable used to used to avoid the simultaneous charging and discharging of storage facility s")
vL_ly = Variable(m, name='vL_ly', type='binary', domain=[L_can,Y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y, which is otherwise 0")
vL_ly_prev = Variable(m, name='vL_ly_prev', type='binary', domain=[L_can,Y], description="Binary variable that is equal to 1 if candidate transmission line l is built in year y or in previous years, which is otherwise 0")
zD_dy = Variable(m, name='zD_dy', type='binary', domain=[D,Y], description="Binary variable that is equal to 1 if the worst-case realization of the peak power consumption of load 𝑑 is equal to its upper bound, which is otherwise 0")
zGC_gy = Variable(m, name='zGC_gy', type='binary', domain=[G,Y], description="Binary variable that is equal to 1 if the worst-case realization of the marginal production cost of conventional generating unit g is equal to its upper bound, which is otherwise 0")
zGP_gy = Variable(m, name='zGP_gy', type='binary', domain=[G,Y], description="Binary variable that is equal to 1 if the worst-case realization of the capacity of conventional generating unit g is equal to its upper bound, which is otherwise 0")
zR_ry = Variable(m, name='zR_ry', type='binary', domain=[R,Y], description="Binary variable that is equal to 1 if the worst-case realization of he capacity of renewable generating unit r is equal to its upper bound, which is otherwise 0")

# Dual variables
lambdaN_nyth = Variable(m, name='lambdaN_nyth', domain=[N,Y,T,H], description="Dual variable associated with the power balance equation at bus n")
muD_dyth_up = Variable(m, name='muD_dyth_up', domain=[D,Y,T,H], description="Dual variable associated with the constraint imposing the upper bound for the unserved demand of load d")
muG_gyth_lo = Variable(m, name='muG_gyth_lo', domain=[G,Y,T,H], description="Dual variable associated with the constraint imposing the lower bound for the power produced by conventional generating unit g")
muG_gyth_up = Variable(m, name='muG_gyth_up', domain=[G,Y,T,H], description="Dual variable associated with the constraint imposing the upper bound for the power produced by conventional generating unit g")
muGD_gyth = Variable(m, name='muGD_gyth', domain=[G,Y,T,H], description="Dual variable associated with the constraint imposing the ramp-down limit of conventional generating unit g, being ℎ greater than 1")
muGU_gyth = Variable(m, name='muGU_gyth', domain=[G,Y,T,H], description="Dual variable associated with the constraint imposing the ramp-up limit of conventional generating unit g, being ℎ greater than 1")
muL_lyth_exist = Variable(m, name='muL_lyth_exist', domain=[L_exist,Y,T,H], description="Dual variable associated with the power flow through existing transmission line l")
muL_lyth_can = Variable(m, name='muL_lyth_can', domain=[L_can,Y,T,H], description="Dual variable associated with the power flow through candidate transmission line l")
muL_lyth_lo = Variable(m, name='muL_lyth_lo', domain=[L,Y,T,H], description="Dual variable associated with the constraint imposing the lower bound for the power flow through transmission line l")
muL_lyth_up = Variable(m, name='muL_lyth_up', domain=[L,Y,T,H], description="Dual variable associated with the constraint imposing the upper bound for the power flow through transmission line l")
muR_ryth_up = Variable(m, name='muR_ryth_up', domain=[R,Y,T,H], description="Dual variable associated with the constraint imposing the upper bound for the power produced by renewable generating unit r")
muS_syth = Variable(m, name='muS_syth', domain=[S,Y,T,H], description="Dual variable associated with the energy stored in storage facility s, being h greater than 1")
muS_syth_lo = Variable(m, name='muS_syth_lo', domain=[S,Y,T,H], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s")
muS_syth_up = Variable(m, name='muS_syth_up', domain=[S,Y,T,H], description="Dual variable associated with the constraint imposing the upper bound for the energy stored in storage facility s")
muSC_syth_up = Variable(m, name='muSC_syth_up', domain=[S,Y,T,H], description="Dual variable associated with the constraint imposing the upper bound for the charging power of storage facility s")
muSD_syth_up = Variable(m, name='muSD_syth_up', domain=[S,Y,T,H], description="Dual variable associated with the constraint imposing the upper bound for the discharging power of storage facility s")
PhiS_syt = Variable(m, name='PhiS_syt', domain=[S,Y,T], description="Dual variable associated with the energy stored in storage facility s at the first RTP of RD")
PhiS_syt_lo = Variable(m, name='PhiS_syt_lo', domain=[S,Y,T], description="Dual variable associated with the constraint imposing the lower bound for the energy stored in storage facility s at the last RTP of RD")
phiN_nyth = Variable(m, name='phiN_nyth', domain=[N,Y,T,H], description="Dual variable associated with the definition of the reference bus n")

min_inv_cost_wc = Variable(m, name="min_inv_cost_wc", description="Worst-case investment costs")
max_op_cost_wc = Variable(m, name="max_op_cost_wc", description="Worst-case operating costs")

# EQUATIONS #
# Outer-loop master problem OF and constraints
# Initialize uncertain parameters to forecast values
cG_gy[G,Y].fx = CG_g_fc[G]
pD_dy[D,Y].fx = PD_d_fc[D]
pG_gy[G,Y].fx = PG_g_fc[G]
pR_ry[R,Y].fx = PR_r_fc[R]

OF_olmp = Equation(m, name="OF_olmp", type="regular")
OF_olmp[...] = min_inv_cost_wc == Sum(Y, rho_y[Y]/power(1.0+kappa, Y.val) +\
              (1.0/power(1.0+kappa, Y.val-1))*Sum(L_can, IL_l[L_can]*vL_ly[L_can,Y]))
con_1c = Equation(m, name="con_1c")
con_1c[...] = Sum(L_can, Sum(Y, (1.0/power(1.0+kappa, Y.val-1))*IL_l[L_can]*vL_ly[L_can,Y])) <= IT
con_1d = Equation(m, name="con_1d", domain=[L_can])
con_1d[L_can] = Sum(Y, vL_ly[L_can,Y]) <= 1
con_1e = Equation(m, name="con_1e", domain=[L_can,Y])
con_1e[L_can,Y] = vL_ly_prev[L_can,Y] == Sum(Yp.where[Yp.val<=Y.val], vL_ly[L_can,Yp])

con_4c = Equation(m, name="con_4c", domain=[Y])
con_4c[Y] = rho_y[Y] >= Sum(T, sigma_yt[Y,T]*Sum(H, tau_yth[Y,T,H]*(Sum(G, CG_g_fc[G]*pG_gyth[G,Y,T,H])+\
            Sum(R, CR_r[R]*(gammaR_ryth[R,Y,T,H]*PR_r_fc[R]-pR_ryth[R,Y,T,H]))+Sum(D, CLS_d[D]*pLS_dyth[D,Y,T,H]))))
con_4d = Equation(m, name="con_4d", domain=[N,T,H,Y])
con_4d[N,T,H,Y] = Sum(G.where[G_n[G,N]], pG_gyth[G,Y,T,H]) + Sum(R.where[R_n[R,N]], pR_ryth[R,Y,T,H]) +\
                  Sum(L.where[SE_l[L,N]], pL_lyth[L,Y,T,H]) - Sum(L.where[RE_l[L,N]], pL_lyth[L,Y,T,H]) +\
                  Sum(S.where[S_n[S,N]], pSD_syth[S,Y,T,H] - pSC_syth[S,Y,T,H]) == \
                  Sum(D.where[D_n[D,N]], gammaD_dyth[D,Y,T,H]*PD_d_fc[D] - pLS_dyth[D,Y,T,H])
con_4e = Equation(m, name="con_4e", domain=[L_exist,T,H,Y])
con_4e[L_exist,T,H,Y] = pL_lyth[L_exist,Y,T,H] == (1.0/X_l[L_exist])*(Sum(N.where[SE_l[L_exist,N]], theta_nyth[N,Y,T,H]) - Sum(N.where[RE_l[L_exist,N]], theta_nyth[N,Y,T,H]))
con_4f_lin1 = Equation(m, name="con_4f_lin1", domain=[L_can,T,H,Y]) # Linearized
con_4f_lin1[L_can,T,H,Y] = pL_lyth[L_can,Y,T,H] - (1/X_l[L_can])*(Sum(N.where[SE_l[L_can,N]], theta_nyth[N,Y,T,H]) - Sum(N.where[RE_l[L_can,N]], theta_nyth[N,Y,T,H])) <= (1-vL_ly_prev[L_can,Y])*FL
con_4f_lin2 = Equation(m, name="con_4f_lin2", domain=[L_can,T,H,Y]) # Linearized
con_4f_lin2[L_can,T,H,Y] = pL_lyth[L_can,Y,T,H] - (1/X_l[L_can])*(Sum(N.where[SE_l[L_can,N]], theta_nyth[N,Y,T,H]) - Sum(N.where[RE_l[L_can,N]], theta_nyth[N,Y,T,H])) >= -(1-vL_ly_prev[L_can,Y])*FL
con_4g_exist_lin1 = Equation(m, name="con_4g_exist_lin1", domain=[L_exist,T,H,Y])
con_4g_exist_lin1[L_exist,T,H,Y] = pL_lyth[L_exist,Y,T,H] <= PL_l[L_exist]
con_4g_exist_lin2 = Equation(m, name="con_4g_exist_lin2", domain=[L_exist,T,H,Y])
con_4g_exist_lin2[L_exist,T,H,Y] = pL_lyth[L_exist,Y,T,H] >= -PL_l[L_exist]
con_4g_can_lin1 = Equation(m, name="con_4g_can_lin1", domain=[L_can,T,H,Y])
con_4g_can_lin1[L_can,T,H,Y] = pL_lyth[L_can,Y,T,H] <= vL_ly_prev[L_can,Y]*PL_l[L_can]
con_4g_can_lin2 = Equation(m, name="con_4g_can_lin2", domain=[L_can,T,H,Y])
con_4g_can_lin2[L_can,T,H,Y] = pL_lyth[L_can,Y,T,H] >= -vL_ly_prev[L_can,Y]*PL_l[L_can]
con_4h = Equation(m, name="con_4h", domain=[S,T,Y]) # H == 1
con_4h[S,T,Y] = eS_syth[S,Y,T,1] == ES_syt0[S,Y,T]+(pSC_syth[S,Y,T,1]*etaSC_s[S]-(pSD_syth[S,Y,T,1]/etaSD_s[S]))*tau_yth[Y,T,1]
con_4i = Equation(m, name="con_4i", domain=[S,T,H,Y]) # H =/= 1
con_4i[S,T,H,Y].where[Ord(H) > 1] = eS_syth[S,Y,T,H] == eS_syth[S,Y,T,H.lag(1)] + (pSC_syth[S,Y,T,H]*etaSC_s[S]-(pSD_syth[S,Y,T,H]/etaSD_s[S]))*tau_yth[Y,T,H]
con_4j = Equation(m, name="con_4j", domain=[S,T,H,Y]) # H == Hmax
con_4j[S,T,H,Y].where[Ord(H) == Card(H)] = ES_syt0[S,Y,T] <= eS_syth[S,Y,T,H]
con_4k1 = Equation(m, name="con_4k1", domain=[S,T,H,Y])
con_4k1[S,T,H,Y] = eS_syth[S,Y,T,H] <= ES_s_max[S]
con_4k2 = Equation(m, name="con_4k2", domain=[S,T,H,Y])
con_4k2[S,T,H,Y] = eS_syth[S,Y,T,H] >= ES_s_min[S]
# con_4l = Equation(m, name="con_4l", domain=[S,T,H,Y])
con_4m1 = Equation(m, name="con_4m1", domain=[S,T,H,Y])
con_4m1[S,T,H,Y] = pSC_syth[S,Y,T,H] <= uS_syth[S,Y,T,H]*PSC_s[S]
con_4m2 = Equation(m, name="con_4m2", domain=[S,T,H,Y])
con_4m2[S,T,H,Y] = pSC_syth[S,Y,T,H] >= 0
con_4n1 = Equation(m, name="con_4n1", domain=[S,T,H,Y])
con_4n1[S,T,H,Y] = pSD_syth[S,Y,T,H] <= (1-uS_syth[S,Y,T,H])*PSD_s[S]
con_4n2 = Equation(m, name="con_4n2", domain=[S,T,H,Y])
con_4n2[S,T,H,Y] = pSD_syth[S,Y,T,H] >= 0
con_4o1 = Equation(m, name="con_4o1", domain=[D,T,H,Y])
con_4o1[D,T,H,Y] = pLS_dyth[D,Y,T,H] <= gammaD_dyth[D,Y,T,H]*PD_d_fc[D]#pD_dy[D,Y]
con_4o2 = Equation(m, name="con_4o2", domain=[D,T,H,Y])
con_4o2[D,T,H,Y] = pLS_dyth[D,Y,T,H] >= 0
# con_4p = Equation(m, name="con_4p", domain=[G,T,H,Y])
con_4q1 = Equation(m, name="con_4q1", domain=[G,T,H,Y])
con_4q1[G,T,H,Y] = pG_gyth[G,Y,T,H] <= uG_gyth[G,Y,T,H]*PG_g_fc[G]#pG_gy[G,Y]
con_4q2 = Equation(m, name="con_4q2", domain=[G,T,H,Y])
con_4q2[G,T,H,Y] = pG_gyth[G,Y,T,H] >= uG_gyth[G,Y,T,H]*PG_g_min[G]
con_4r1 = Equation(m, name="con_4r1", domain=[G,T,H,Y]) # H =/= 1
con_4r1[G,T,H,Y].where[Ord(H) > 1] = pG_gyth[G,Y,T,H]-pG_gyth[G,Y,T,H.lag(1)] <= RGU_g[G]
con_4r2 = Equation(m, name="con_4r2", domain=[G,T,H,Y]) # H =/= 1
con_4r2[G,T,H,Y].where[Ord(H) > 1] = pG_gyth[G,Y,T,H]-pG_gyth[G,Y,T,H.lag(1)] >= -RGD_g[G]
con_4s1 = Equation(m, name="con_4s1", domain=[R,T,H,Y])
con_4s1[R,T,H,Y] = pR_ryth[R,Y,T,H] <= gammaR_ryth[R,Y,T,H]*PR_r_fc[R]#pR_ry[R,Y]
con_4s2 = Equation(m, name="con_4s2", domain=[R,T,H,Y])
con_4s2[R,T,H,Y] = pR_ryth[R,Y,T,H] >= 0
con_4t = Equation(m, name="con_4t", domain=[T,H,Y]) # N == ref bus
con_4t[T,H,Y] = theta_nyth[1,Y,T,H] == 0

# Outer-loop subproblem
# Inner-loop master problem OF and constraints
# OF_ilmp = Equation(m, name="OF_ilmp", type="regular")
con_2b = Equation(m, name="con_2b", domain=[G,Y])
con_2b[G,Y] = cG_gy[G,Y] == CG_g_fc[G]*power(1+zetaGC_g_fc[G], Y.val-1) + CG_g_max[G]*power(1+zetaGC_g_max[G], Y.val-1)*zGC_gy[G,Y]
con_2c = Equation(m, name="con_2c", domain=[D,Y])
con_2c[D,Y] = pD_dy[D,Y] == PD_d_fc[D]*power(1+zetaD_d_fc[D], Y.val-1) + PD_d_max[D]*power(1+zetaD_d_max[D], Y.val-1)*zD_dy[D,Y]
con_2d = Equation(m, name="con_2d", domain=[G,Y])
con_2d[G,Y] = pG_gy[G,Y] == PG_g_fc[G]*power(1+zetaGP_g_fc[G], Y.val-1) - PG_g_max[G]*power(1+zetaGP_g_max[G], Y.val-1)*zGP_gy[G,Y]
con_2e = Equation(m, name="con_2e", domain=[R,Y])
con_2e[R,Y] = pR_ry[R,Y] == PR_r_fc[R]*power(1+zetaR_r_fc[R], Y.val-1) - PR_r_max[R]*power(1+zetaR_r_max[R], Y.val-1)*zR_ry[R,Y]
# con_2f = Equation(m, name="con_2f", domain=[G,Y])
# con_2g = Equation(m, name="con_2g", domain=[D,Y])
# con_2h = Equation(m, name="con_2h", domain=[G,Y])
# con_2i = Equation(m, name="con_2i", domain=[R,Y])
con_2j = Equation(m, name="con_2j", domain=[Y])
con_2j[Y] = Sum(G, zGC_gy[G,Y]) <= GammaGC
con_2k = Equation(m, name="con_2k", domain=[Y])
con_2k[Y] = Sum(D, zD_dy[D,Y]) <= GammaD
con_2l = Equation(m, name="con_2l", domain=[Y])
con_2l[Y] = Sum(G, zGP_gy[G,Y]) <= GammaGP
con_2m = Equation(m, name="con_2m", domain=[Y])
con_2m[Y] = Sum(RS, zR_ry[RS,Y]) <= GammaRS
con_2n = Equation(m, name="con_2n", domain=[Y])
con_2n[Y] = Sum(RW, zR_ry[RW,Y]) <= GammaRW

yi = 1
con_5c = Equation(m, name="con_5c")
con_5c[...] = xi_y[yi] <= Sum(T, Sum(H, Sum(D, gammaD_dyth[D,yi,T,H]*pD_dy[D,yi]*Sum(N.where[D_n[D,N]], lambdaN_nyth[N,yi,T,H]))-\
                          Sum(L, PL_l[L]*(muL_lyth_lo[L,yi,T,H]+muL_lyth_up[L,yi,T,H]))-\
                          Sum(S, uS_syth[S,yi,T,H]*PSC_s[S]*muSC_syth_up[S,yi,T,H]+(1-uS_syth[S,yi,T,H])*PSD_s[S]*muSD_syth_up[S,yi,T,H]-ES_s_min[S]*muS_syth_lo[S,yi,T,H]+ES_s_max[S]*muS_syth_up[S,yi,T,H])+\
                          Sum(G, uG_gyth[G,yi,T,H]*(PG_g_min[G]*muG_gyth_lo[G,yi,T,H]-pG_gy[G,yi]*muG_gyth_up[G,yi,T,H]))-\
                          Sum(R, gammaR_ryth[R,yi,T,H]*pR_ry[R,yi]*(muR_ryth_up[R,yi,T,H]-sigma_yt[yi,T]*tau_yth[yi,T,H]*CR_r[R]))-\
                          Sum(D, gammaD_dyth[D,yi,T,H]*pD_dy[D,yi]*muD_dyth_up[D,yi,T,H]))+\
                          Sum(S, ES_syt0[S,yi,T]*(PhiS_syt[S,yi,T]+PhiS_syt_lo[S,yi,T]))-\
                          Sum(H.where[Ord(H)>1], Sum(G, RGD_g[G]*muGD_gyth[G,yi,T,H]+RGU_g[G]*muGU_gyth[G,yi,T,H])))
con_5def = Equation(m, name="con_5d", domain=[G,T,H])
con_5def[G,T,H] = Sum(N.where[G_n[G,N]], lambdaN_nyth[N,yi,T,H])+muG_gyth_lo[G,yi,T,H]-muG_gyth_up[G,yi,T,H]+muGD_gyth[G,yi,T,H] -\
              muGD_gyth[G,yi,T,H.lead(1)]-muGU_gyth[G,yi,T,H]+muGU_gyth[G,yi,T,H.lead(1)] == sigma_yt[yi,T]*tau_yth[yi,T,H]*cG_gy[G,yi]
con_5def[G,T,H].where[Ord(H)==1] = Sum(N.where[G_n[G,N]], lambdaN_nyth[N,yi,T,H])+muG_gyth_lo[G,yi,T,H]-muG_gyth_up[G,yi,T,H]-muGD_gyth[G,yi,T,H.lead(1)] +\
              muGU_gyth[G,yi,T,H.lead(1)] == sigma_yt[yi,T]*tau_yth[yi,T,H]*cG_gy[G,yi]
con_5def[G,T,H].where[Ord(H)==Card(H)] = Sum(N.where[G_n[G,N]], lambdaN_nyth[N,yi,T,H])+muG_gyth_lo[G,yi,T,H]-muG_gyth_up[G,yi,T,H]+muGD_gyth[G,yi,T,H] -\
              muGU_gyth[G,yi,T,H] == sigma_yt[yi,T]*tau_yth[yi,T,H]*cG_gy[G,yi]
# con_5g = Equation(m, name="con_5g", domain=[Y])
con_5h = Equation(m, name="con_5h", domain=[Y])
con_5i = Equation(m, name="con_5i", domain=[Y])
con_5j = Equation(m, name="con_5j", domain=[Y])
con_5k = Equation(m, name="con_5k", domain=[Y])
con_5l = Equation(m, name="con_5l", domain=[Y])
con_5m = Equation(m, name="con_5m", domain=[Y])
con_5n = Equation(m, name="con_5n", domain=[Y])
con_5o = Equation(m, name="con_5o", domain=[Y])
con_5p = Equation(m, name="con_5p", domain=[Y])
con_5q = Equation(m, name="con_5q", domain=[Y])
con_5r = Equation(m, name="con_5r", domain=[Y])
con_5s = Equation(m, name="con_5s", domain=[Y])
con_5t = Equation(m, name="con_5t", domain=[Y])
con_5u = Equation(m, name="con_5u", domain=[Y])
con_5v = Equation(m, name="con_5v", domain=[Y])
con_5w = Equation(m, name="con_5w", domain=[Y])
con_5x = Equation(m, name="con_5x", domain=[Y])
con_5y = Equation(m, name="con_5y", domain=[Y])
con_5z = Equation(m, name="con_5z", domain=[Y])

OLMP_model = Model(
    m,
    name="OLMP",
    description="Outer-loop subproblem",
    equations=m.getEquations(),
    problem='MIP',
    sense='min',
    objective=min_inv_cost_wc,
)

# summary = OLMP_model.solve(options=Options(relative_optimality_gap=0.005, mip="CPLEX", savepoint=1), output=sys.stdout)
# redirect output to a file
# print("Objective Function Value:  ", round(OLMP_model.objective_value, 3))
# print(vL_ly.records)
# with open("gamspy_solve_out", "w") as file:
#     OLMP_model.solve(output=file)



