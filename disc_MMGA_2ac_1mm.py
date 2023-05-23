# infinitesimal gradient ascent
# two-action, one-memory
#


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as animation

x_vc, y_vc = 0.7*np.ones(4), 0.7*np.ones(4)
#x_vc, y_vc = np.random.random(4), np.random.random(4)
print(x_vc)
print(y_vc)
xc_vc, yc_vc = np.copy(x_vc), np.copy(y_vc)
uo_vc, vo_vc = [1,-1,-1,1], [-1,1,1,-1]
u_vc, v_vc = np.array(uo_vc), np.array(vo_vc)
Tmax, NdT = 1000, 100
dp = 10**-6


# output textfile
txt = open('output.txt', 'w')
txt.write('#time:[Tmax,NdT] = ' + str([Tmax,NdT]) + '\n')
txt.write('#payoff:u = ' + str([uo_vc]) + ', v = ' + str([vo_vc]) + '\n')
txt.write('#[dp] = ' + str([dp]) + '\n')
txt.write('#strategies: one-memory vs one-memory' + '\n')
txt.write('#t, x_vc(4), y_vc(4), p_vc(4), ueq(1), veq(1)' + '\n')


# eigenvector
def EIGEN_VECTOR(M_mt):
    distance = 1
    p_vc = np.ones(4)/4
    while distance > 10**-9:
        pn_vc = np.dot(M_mt,p_vc)
        pn_vc = pn_vc/np.sum(pn_vc)
        distance = np.sum((pn_vc-p_vc)**2)**0.5
        p_vc = pn_vc
    return(p_vc)

# matrix generation
def M_MATRIX(x_vc, y_vc):
    M_mt = np.array([x_vc*y_vc,x_vc*(1-y_vc),(1-x_vc)*y_vc,(1-x_vc)*(1-y_vc)])
    return(M_mt)

ueqp_vc, veqp_vc = [], []
x_vct0, x_vct1, x_vct2, x_vct3 = [], [], [], []
y_vct0, y_vct1, y_vct2, y_vct3 = [], [], [], []
for t in range(0,int(Tmax*NdT)+1):
    M_mt = M_MATRIX(x_vc, y_vc)
    po_vc = EIGEN_VECTOR(M_mt)
    ueqo, veqo = np.dot(po_vc,u_vc), np.dot(po_vc,v_vc)
    ueqp_vc.append(ueqo)
    veqp_vc.append(veqo)
    txt.write(str(round(t/NdT,3))+'\t')
    for l in range(0,4):
        txt.write(str(x_vc[l])+'\t')
    for l in range(0,4):
        txt.write(str(y_vc[l])+'\t')
    for l in range(0,4):
        txt.write(str(po_vc[l])+'\t')
    txt.write(str(ueqo)+'\t'+str(veqo)+'\n')
    dueq_vc = []
    for j in range(0,4):
        xn_vc = np.copy(x_vc)
        xn_vc[j] += dp
        M_mt = M_MATRIX(xn_vc, y_vc)
        p_vc = EIGEN_VECTOR(M_mt)
        dueq = (np.dot(p_vc,u_vc)-ueqo)/dp
        dueq_vc.append(dueq)
    dveq_vc = []
    for j in range(0,4):
        yn_vc = np.copy(y_vc)
        yn_vc[j] += dp
        M_mt = M_MATRIX(x_vc, yn_vc)
        p_vc = EIGEN_VECTOR(M_mt)
        dveq = (np.dot(p_vc,v_vc)-veqo)/dp
        dveq_vc.append(dveq)
    x_vc += x_vc*(1-x_vc)*dueq_vc/NdT
    y_vc += y_vc*(1-y_vc)*dveq_vc/NdT
    if t%(NdT*100) == 0:
        print(t/NdT)


txt.close()


