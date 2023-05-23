# infinitesimal gradient ascent
# three-action, two-memory
#


import numpy as np


det = 2 # if 1: random, 2: for Fig.~4
if det == 1:
    RN_vc = np.sort(np.random.random([162,2]))
    x_vc = RN_vc[:81,0]
    x_vc = np.append(x_vc, RN_vc[:81,1]-RN_vc[:81,0])
    x_vc = np.append(x_vc, 1-RN_vc[:81,1])
    x_vc = np.reshape(x_vc, (3,81))
    x_vc = np.transpose(x_vc)
    x_vc = np.reshape(x_vc, (243))
    y_vc = RN_vc[81:,0]
    y_vc = np.append(y_vc, RN_vc[81:,1]-RN_vc[81:,0])
    y_vc = np.append(y_vc, 1-RN_vc[81:,1])
    y_vc = np.reshape(y_vc, (3,81))
    y_vc = np.transpose(y_vc)
    y_vc = np.reshape(y_vc, (243))
    x_vc, y_vc = np.array(x_vc), np.array(y_vc)
elif det == 2:
    x_vc = np.reshape(np.outer(np.ones(81),[0.7,0.15,0.15]),[243])
    y_vc = np.reshape(np.outer(np.ones(81),[0.15,0.7,0.15]),[243])
print(x_vc)
print(y_vc)
eps = 0
uo_vc, vo_vc = [eps,1,-1,-1,eps,1,1,-1,eps], [-eps,-1,1,1,-eps,-1,-1,1,-eps]
u_vc, v_vc = np.reshape(np.outer(np.array(uo_vc)*1.0,np.ones(9)), (81)), np.reshape(np.outer(np.array(vo_vc)*1.0,np.ones(9)), (81))
Tmax, NdT = 1000, 100
dp = 10**-6


# output textfile
txt = open('output.txt', 'w')
txt.write('#time:[Tmax,NdT] = ' + str([Tmax,NdT]) + '\n')
txt.write('#payoff:u = ' + str([uo_vc]) + ', v = ' + str([vo_vc]) + '\n')
txt.write('#[dp] = ' + str([dp]) + '\n')
txt.write('#strategies: two-memory vs two-memory' + '\n')
txt.write('#t, x_vc(243), y_vc(243), p_vc(81), ueq(1), veq(1)' + '\n')


# eigenvector
def EIGEN_VECTOR(M_mt):
    distance = 1
    p_vc = np.ones(81)/81
    while distance > 10**-9:
        pn_vc = np.dot(M_mt,p_vc)
        pn_vc = pn_vc/np.sum(pn_vc)
        distance = np.sum((pn_vc-p_vc)**2)**0.5
        p_vc = pn_vc
    return(p_vc)

# matrix generation
def M_MATRIX(x_vc, y_vc):
    M_mt = np.array([])
    for j in range(0,81):
        jmod9 = int(j/9)
        ejmod9_vc = np.zeros(9)
        ejmod9_vc[jmod9] += 1
        m_vc = np.reshape(np.outer(np.reshape(np.outer(x_vc[3*j:3*(j+1)],y_vc[3*j:3*(j+1)]), (9)),ejmod9_vc), (81))
        M_mt = np.append(M_mt, m_vc)
    M_mt = np.transpose(np.reshape(M_mt, (81,81)))
    return(M_mt)


ueqp_vc, veqp_vc = [], []
for t in range(0,int(Tmax*NdT)+1):
    M_mt = M_MATRIX(x_vc, y_vc)
    po_vc = EIGEN_VECTOR(M_mt)
    ueqo, veqo = np.dot(po_vc,u_vc), np.dot(po_vc,v_vc)
    ueqp_vc.append(ueqo)
    veqp_vc.append(veqo)
    txt.write(str(round(t/NdT,3))+'\t')
    for l in range(0,243):
        txt.write(str(x_vc[l])+'\t')
    for l in range(0,243):
        txt.write(str(y_vc[l])+'\t')
    for l in range(0,81):
        txt.write(str(po_vc[l])+'\t')
    txt.write(str(ueqo)+'\t'+str(veqo)+'\n')
    dueq_vc = []
    for j in range(0,243):
        xn_vc = np.copy(x_vc)
        xn_vc[j] += dp
        jmod3 = int(j/3)
        xn_vc[3*jmod3:3*(jmod3+1)] = xn_vc[3*jmod3:3*(jmod3+1)]/np.sum(xn_vc[3*jmod3:3*(jmod3+1)])
        M_mt = M_MATRIX(xn_vc, y_vc)
        p_vc = EIGEN_VECTOR(M_mt)
        dueq = (np.dot(p_vc,u_vc)-ueqo)/dp
        dueq_vc.append(dueq)
    dveq_vc = []
    for j in range(0,243):
        yn_vc = np.copy(y_vc)
        yn_vc[j] += dp
        jmod3 = int(j/3)
        yn_vc[3*jmod3:3*(jmod3+1)] = yn_vc[3*jmod3:3*(jmod3+1)]/np.sum(yn_vc[3*jmod3:3*(jmod3+1)])
        M_mt = M_MATRIX(x_vc, yn_vc)
        p_vc = EIGEN_VECTOR(M_mt)
        dveq = (np.dot(p_vc,v_vc)-veqo)/dp
        dveq_vc.append(dveq)
    x_vc += x_vc*dueq_vc/NdT
    y_vc += y_vc*dveq_vc/NdT
    x_vc = np.transpose(np.reshape(x_vc, (81,3)))
    x_vc = x_vc/np.sum(x_vc, axis=0)
    x_vc = np.reshape(np.transpose(x_vc), (243))
    y_vc = np.transpose(np.reshape(y_vc, (81,3)))
    y_vc = y_vc/np.sum(y_vc, axis=0)
    y_vc = np.reshape(np.transpose(y_vc), (243))
    if t%(NdT) == 0:
        print(t/NdT)

txt.close()




