# infinitesimal gradient ascent
# two-action, three-memory
#


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as animation


#x_vc, y_vc = np.random.random(64), np.random.random(64)
x_vc, y_vc = 0.7*np.ones(64), 0.7*np.ones(64)
x_vc, y_vc = np.array(x_vc), np.array(y_vc)
print(x_vc)
print(y_vc)
uo_vc, vo_vc = [1,-1,-1,1], [-1,1,1,-1]
u_vc, v_vc = np.outer(uo_vc,np.ones(16)), np.outer(vo_vc,np.ones(16))
u_vc, v_vc = np.reshape(u_vc, [64]), np.reshape(v_vc, [64])
Tmax, NdT = 1000, 100
dp = 10**-6


# output textfile
txt = open('output.txt', 'w')
txt.write('#time:[Tmax,NdT] = ' + str([Tmax,NdT]) + '\n')
txt.write('#payoff:u = ' + str([uo_vc]) + ', v = ' + str([vo_vc]) + '\n')
txt.write('#[dp] = ' + str([dp]) + '\n')
txt.write('#strategies: three-memory vs three-memory' + '\n')
txt.write('#t, x_vc(64), y_vc(64), p_vc(64), ueq(1), veq(1)' + '\n')


# eigenvector
def EIGEN_VECTOR(M_mt):
    distance = 1
    p_vc = np.ones(64)/64
    while distance > 10**-9:
        pn_vc = np.dot(M_mt,p_vc)
        pn_vc = pn_vc/np.sum(pn_vc)
        distance = np.sum((pn_vc-p_vc)**2)**0.5
        p_vc = pn_vc
    return(p_vc)


# matrix generation
def M_MATRIX(x_vc, y_vc):
    sixteen_sixtyfour_mt = np.outer(np.eye(16),np.ones(4))
    sixteen_sixtyfour_mt = np.reshape(sixteen_sixtyfour_mt, [16,64])
    M1_mt = x_vc*y_vc*sixteen_sixtyfour_mt
    M2_mt = x_vc*(1-y_vc)*sixteen_sixtyfour_mt
    M3_mt = (1-x_vc)*y_vc*sixteen_sixtyfour_mt
    M4_mt = (1-x_vc)*(1-y_vc)*sixteen_sixtyfour_mt
    M_mt = np.append(M1_mt, M2_mt, axis=0)
    M_mt = np.append(M_mt, M3_mt, axis=0)
    M_mt = np.append(M_mt, M4_mt, axis=0)
    return(M_mt)

ueqp_vc, veqp_vc = [], []
for t in range(0,int(Tmax*NdT)+1):
    M_mt = M_MATRIX(x_vc, y_vc)
    po_vc = EIGEN_VECTOR(M_mt)
    ueqo, veqo = np.dot(po_vc,u_vc), np.dot(po_vc,v_vc)
    ueqp_vc.append(ueqo)
    veqp_vc.append(veqo)
    txt.write(str(round(t/NdT,3))+'\t')
    for l in range(0,64):
        txt.write(str(x_vc[l])+'\t')
    for l in range(0,64):
        txt.write(str(y_vc[l])+'\t')
    for l in range(0,64):
        txt.write(str(po_vc[l])+'\t')
    txt.write(str(ueqo)+'\t'+str(veqo)+'\n')
    dueq_vc = []
    for j in range(0,64):
        xn_vc = np.copy(x_vc)
        xn_vc[j] += dp
        M_mt = M_MATRIX(xn_vc, y_vc)
        p_vc = EIGEN_VECTOR(M_mt)
        dueq = (np.dot(p_vc,u_vc)-ueqo)/dp
        dueq_vc.append(dueq)
    dveq_vc = []
    for j in range(0,64):
        yn_vc = np.copy(y_vc)
        yn_vc[j] += dp
        M_mt = M_MATRIX(x_vc, yn_vc)
        p_vc = EIGEN_VECTOR(M_mt)
        dveq = (np.dot(p_vc,v_vc)-veqo)/dp
        dveq_vc.append(dveq)
    x_vc += x_vc*(1-x_vc)*dueq_vc/NdT
    y_vc += y_vc*(1-y_vc)*dveq_vc/NdT
    if t%(NdT*10) == 0:
        print(t/NdT)

txt.close()




