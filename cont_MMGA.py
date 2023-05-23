# analyzing 2nd, 3rd, 4th-order approximations of one-memory penny-matching game
# comparing the approximation 
# output textfile
#


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation as animation


Tmax, NdT = 10000, 100
scale = 1.0*10**-1
Dxi_vc, Dyi_vc = scale*np.random.randn(4), scale*np.random.randn(4)
print(Dxi_vc, Dyi_vc)
u_vc, v_vc = np.array([+1,-1,-1,+1]),np.array([-1,+1,+1,-1])


# output textfile
txt = open('output.txt', 'w')
txt.write('#time:[Tmax,NdT] (Runge-Kutta 4) = ' + str([Tmax,NdT]) + '\n')
txt.write('#payoff:u = ' + str([u_vc]) + ', v = ' + str([v_vc]) + '\n')
txt.write('#strategies: one-memory vs one-memory' + '\n')
txt.write('#Dxi_vc = ' + str([Dxi_vc]) + 'Dyi_vc = ' + str([Dyi_vc]) + '\n')
txt.write('#t, DxT_vc(4), DyT_vc(4), DxO2_vc(4), DyO2_vc(4), DxO3_vc(4), DyO3_vc(4), DxO4_vc(4), DyO4_vc(4)' + '\n')


# define function (analytically calculate equilibrium state for fixed strategy)
def Q_AND_DQ_ANALYTICAL(x_vc, y_vc):
    x1, x2, x3, x4 = x_vc[0], x_vc[1], x_vc[2], x_vc[3]
    y1, y2, y3, y4 = y_vc[0], y_vc[1], y_vc[2], y_vc[3]
    q1 = (x4+(x3-x4)*y3)*(y4+(y2-y4)*x2)-x3*y2*(x2-x4)*(y3-y4)
    q2 = (x4+(x3-x4)*y4)*(1-y3-(y1-y3)*x1)-x4*(1-y1)*(x1-x3)*(y3-y4)
    q3 = (1-x2-(x1-x2)*y1)*(y4+(y2-y4)*x4)-(1-x1)*y4*(x2-x4)*(y1-y2)
    q4 = (1-x2-(x1-x2)*y2)*(1-y3-(y1-y3)*x3)-(1-x2)*(1-y3)*(x1-x3)*(y1-y2)
    q_vc = np.array([q1,q2,q3,q4])
    dq1dx2 = (y2-y4)*(x4+(x3-x4)*y3)-x3*y2*(y3-y4)
    dq1dx3 = y3*(y4+(y2-y4)*x2)-y2*(x2-x4)*(y3-y4)
    dq1dx4 = (1-y3)*(y4+(y2-y4)*x2)+x3*y2*(y3-y4)
    dq2dx1 = -(y1-y3)*(x4+(x3-x4)*y4)-x4*(1-y1)*(y3-y4)
    dq2dx3 = y4*(1-y3-(y1-y3)*x1)+x4*(1-y1)*(y3-y4)
    dq2dx4 = (1-y4)*(1-y3-(y1-y3)*x1)-(1-y1)*(x1-x3)*(y3-y4)
    dq3dx1 = -y1*(y4+(y2-y4)*x4)+y4*(x2-x4)*(y1-y2)
    dq3dx2 = -(1-y1)*(y4+(y2-y4)*x4)-(1-x1)*y4*(y1-y2)
    dq3dx4 = (y2-y4)*(1-x2-(x1-x2)*y1)+(1-x1)*y4*(y1-y2)
    dq4dx1 = -y2*(1-y3-(y1-y3)*x3)-(1-x2)*(1-y3)*(y1-y2)
    dq4dx2 = -(1-y2)*(1-y3-(y1-y3)*x3)+(1-y3)*(x1-x3)*(y1-y2)
    dq4dx3 = -(y1-y3)*(1-x2-(x1-x2)*y2)+(1-x2)*(1-y3)*(y1-y2)
    dqdx_mt = np.array([[0,dq2dx1,dq3dx1,dq4dx1],[dq1dx2,0,dq3dx2,dq4dx2],[dq1dx3,dq2dx3,0,dq4dx3],[dq1dx4,dq2dx4,dq3dx4,0]])
    dq1dy2 = x2*(x4+(x3-x4)*y3)-x3*(x2-x4)*(y3-y4)
    dq1dy3 = (x3-x4)*(y4+(y2-y4)*x2)-x3*y2*(x2-x4)
    dq1dy4 = (1-x2)*(x4+(x3-x4)*y3)+x3*y2*(x2-x4)
    dq2dy1 = -x1*(x4+(x3-x4)*y4)+x4*(x1-x3)*(y3-y4)
    dq2dy3 = -(1-x1)*(x4+(x3-x4)*y4)-x4*(1-y1)*(x1-x3)
    dq2dy4 = (x3-x4)*(1-y3-(y1-y3)*x1)+x4*(1-y1)*(x1-x3)
    dq3dy1 = -(x1-x2)*(y4+(y2-y4)*x4)-(1-x1)*y4*(x2-x4)
    dq3dy2 = x4*(1-x2-(x1-x2)*y1)+(1-x1)*y4*(x2-x4)
    dq3dy4 = (1-x4)*(1-x2-(x1-x2)*y1)-(1-x1)*(x2-x4)*(y1-y2)
    dq4dy1 = -x3*(1-x2-(x1-x2)*y2)-(1-x2)*(1-y3)*(x1-x3)
    dq4dy2 = -(x1-x2)*(1-y3-(y1-y3)*x3)+(1-x2)*(1-y3)*(x1-x3)
    dq4dy3 = -(1-x3)*(1-x2-(x1-x2)*y2)+(1-x2)*(x1-x3)*(y1-y2)
    dqdy_mt = np.array([[0,dq2dy1,dq3dy1,dq4dy1],[dq1dy2,0,dq3dy2,dq4dy2],[dq1dy3,dq2dy3,0,dq4dy3],[dq1dy4,dq2dy4,dq3dy4,0]])
    return(q_vc, dqdx_mt, dqdy_mt)

def SECOND_ORDER_ANALYTICAL(Dx_vc, Dy_vc):
    dO2_vc = Dy_vc/4
    eO2_vc = Dx_vc/4
    d_vc = +dO2_vc
    e_vc = -eO2_vc
    return(d_vc, e_vc)

def THIRD_ORDER_ANALYTICAL(Dx_vc, Dy_vc):
    vc_x, vc_y = np.array([+1,+1,-1,-1]), np.array([+1,-1,+1,-1])
    sumd = np.sum(Dx_vc)
    sume = np.sum(Dy_vc)
    sumde_x, sumde_y = np.sum(Dx_vc*Dy_vc*vc_x), np.sum(Dx_vc*Dy_vc*vc_y)
    dO2_vc = Dy_vc/4
    eO2_vc = Dx_vc/4
    dO3_vc = (Dy_vc*vc_x*sumd+Dy_vc*vc_y*sume+sumde_x)/8
    eO3_vc = (Dx_vc*vc_y*sume+Dx_vc*vc_x*sumd+sumde_y)/8
    d_vc = +dO2_vc+dO3_vc
    e_vc = -eO2_vc-eO3_vc
    return(d_vc, e_vc)

def FOURTH_ORDER_ANALYTICAL(Dx_vc, Dy_vc):
    vc_x, vc_y, vc_z = np.array([+1,+1,-1,-1]), np.array([+1,-1,+1,-1]), np.array([+1,-1,-1,+1])
    sumd, sumd_x, sumd_y = np.sum(Dx_vc), np.sum(Dx_vc*vc_x), np.sum(Dx_vc*vc_y)
    sume, sume_x, sume_y = np.sum(Dy_vc), np.sum(Dy_vc*vc_x), np.sum(Dy_vc*vc_y)
    sumde, sumde_x, sumde_y, sumde_z = np.sum(Dx_vc*Dy_vc), np.sum(Dx_vc*Dy_vc*vc_x), np.sum(Dx_vc*Dy_vc*vc_y), np.sum(Dx_vc*Dy_vc*vc_z)
    dO2_vc = Dy_vc/4
    eO2_vc = Dx_vc/4
    dO3_vc = (Dy_vc*vc_x*sumd+Dy_vc*vc_y*sume+sumde_x)/8
    eO3_vc = (Dx_vc*vc_y*sume+Dx_vc*vc_x*sumd+sumde_y)/8
    dO4_vc = (Dy_vc*vc_z*sumde+Dy_vc*sumde_z)/4+(Dy_vc*vc_x*(sumd_x*sumd+sumd_y*sume)+(vc_x*sumd+vc_y*sume+sumd_x)*sumde_x+Dy_vc*vc_y*(sume_x*sumd+sume_y*sume)+sume_x*sumde_y)/16-Dx_vc**2*Dy_vc
    eO4_vc = (Dx_vc*vc_z*sumde+Dx_vc*sumde_z)/4+(Dx_vc*vc_y*(sume_y*sume+sume_x*sumd)+(vc_y*sume+vc_x*sumd+sume_y)*sumde_y+Dx_vc*vc_x*(sumd_y*sume+sumd_x*sumd)+sumd_y*sumde_x)/16-Dy_vc**2*Dx_vc
    d_vc = +dO2_vc+dO3_vc+dO4_vc
    e_vc = -eO2_vc-eO3_vc-eO4_vc
    return(d_vc, e_vc)

xT_vc, yT_vc = 0.5+np.copy(Dxi_vc), 0.5+np.copy(Dyi_vc)
DxO2_vc, DyO2_vc = np.copy(Dxi_vc), np.copy(Dyi_vc)
DxO3_vc, DyO3_vc = np.copy(Dxi_vc), np.copy(Dyi_vc)
DxO4_vc, DyO4_vc = np.copy(Dxi_vc), np.copy(Dyi_vc)
for i in range(0,Tmax*NdT+1):
    # write txt
    txt.write(str(round(i/NdT,3))+'\t')
    for l in range(0,4):
        txt.write(str(xT_vc[l]-0.5)+'\t')
    for l in range(0,4):
        txt.write(str(yT_vc[l]-0.5)+'\t')
    for l in range(0,4):
        txt.write(str(DxO2_vc[l])+'\t')
    for l in range(0,4):
        txt.write(str(DyO2_vc[l])+'\t')
    for l in range(0,4):
        txt.write(str(DxO3_vc[l])+'\t')
    for l in range(0,4):
        txt.write(str(DyO3_vc[l])+'\t')
    for l in range(0,4):
        txt.write(str(DxO4_vc[l])+'\t')
    for l in range(0,3):
        txt.write(str(DyO4_vc[l])+'\t')
    txt.write(str(DyO4_vc[3])+'\n')

    # analytical
    # Runge-Kutta k
    q_vc, dqdx_mt, dqdy_mt = Q_AND_DQ_ANALYTICAL(xT_vc, yT_vc)
    dxTk_vc = xT_vc*(1-xT_vc)*(np.dot(dqdx_mt,u_vc)*np.sum(q_vc)-np.dot(q_vc,u_vc)*np.sum(dqdx_mt,axis=1))/np.sum(q_vc)**2
    dyTk_vc = yT_vc*(1-yT_vc)*(np.dot(dqdy_mt,v_vc)*np.sum(q_vc)-np.dot(q_vc,v_vc)*np.sum(dqdy_mt,axis=1))/np.sum(q_vc)**2
    xTk_vc, yTk_vc = xT_vc+dxTk_vc/NdT/2, yT_vc+dyTk_vc/NdT/2
    # Runge-Kutta l
    qk_vc, dqdxk_mt, dqdyk_mt = Q_AND_DQ_ANALYTICAL(xTk_vc, yTk_vc)
    dxTl_vc = xTk_vc*(1-xTk_vc)*(np.dot(dqdxk_mt,u_vc)*np.sum(qk_vc)-np.dot(qk_vc,u_vc)*np.sum(dqdxk_mt,axis=1))/np.sum(qk_vc)**2
    dyTl_vc = yTk_vc*(1-yTk_vc)*(np.dot(dqdyk_mt,v_vc)*np.sum(qk_vc)-np.dot(qk_vc,v_vc)*np.sum(dqdyk_mt,axis=1))/np.sum(qk_vc)**2
    xTl_vc, yTl_vc = xT_vc+dxTl_vc/NdT/2, yT_vc+dyTl_vc/NdT/2
    # Runge-Kutta m
    ql_vc, dqdxl_mt, dqdyl_mt = Q_AND_DQ_ANALYTICAL(xTl_vc, yTl_vc)
    dxTm_vc = xTl_vc*(1-xTl_vc)*(np.dot(dqdxl_mt,u_vc)*np.sum(ql_vc)-np.dot(ql_vc,u_vc)*np.sum(dqdxl_mt,axis=1))/np.sum(ql_vc)**2
    dyTm_vc = yTl_vc*(1-yTl_vc)*(np.dot(dqdyl_mt,v_vc)*np.sum(ql_vc)-np.dot(ql_vc,v_vc)*np.sum(dqdyl_mt,axis=1))/np.sum(ql_vc)**2
    xTm_vc, yTm_vc = xT_vc+dxTm_vc/NdT, yT_vc+dyTm_vc/NdT
    # Runge-Kutta n
    qm_vc, dqdxm_mt, dqdym_mt = Q_AND_DQ_ANALYTICAL(xTm_vc, yTm_vc)
    dxTn_vc = xTm_vc*(1-xTm_vc)*(np.dot(dqdxm_mt,u_vc)*np.sum(qm_vc)-np.dot(qm_vc,u_vc)*np.sum(dqdxm_mt,axis=1))/np.sum(qm_vc)**2
    dyTn_vc = yTm_vc*(1-yTm_vc)*(np.dot(dqdym_mt,v_vc)*np.sum(qm_vc)-np.dot(qm_vc,v_vc)*np.sum(dqdym_mt,axis=1))/np.sum(qm_vc)**2
    dxT_vc, dyT_vc = (dxTk_vc+2*dxTl_vc+2*dxTm_vc+dxTn_vc)/6, (dyTk_vc+2*dyTl_vc+2*dyTm_vc+dyTn_vc)/6
    xT_vc += dxT_vc/NdT
    yT_vc += dyT_vc/NdT

    # second-order approximation
    # Runge-Kutta k
    dxO2k_vc, dyO2k_vc = SECOND_ORDER_ANALYTICAL(DxO2_vc, DyO2_vc)
    DxO2k_vc, DyO2k_vc = DxO2_vc+dxO2k_vc/NdT/2, DyO2_vc+dyO2k_vc/NdT/2
    # Runge-Kutta l
    dxO2l_vc, dyO2l_vc = SECOND_ORDER_ANALYTICAL(DxO2k_vc, DyO2k_vc)
    DxO2l_vc, DyO2l_vc = DxO2_vc+dxO2l_vc/NdT/2, DyO2_vc+dyO2l_vc/NdT/2
    # Runge-Kutta m
    dxO2m_vc, dyO2m_vc = SECOND_ORDER_ANALYTICAL(DxO2l_vc, DyO2l_vc)
    DxO2m_vc, DyO2m_vc = DxO2_vc+dxO2m_vc/NdT, DyO2_vc+dyO2m_vc/NdT
    # Runge-Kutta n
    dxO2n_vc, dyO2n_vc = SECOND_ORDER_ANALYTICAL(DxO2m_vc, DyO2m_vc)
    dxO2_vc, dyO2_vc = (dxO2k_vc+2*dxO2l_vc+2*dxO2m_vc+dxO2n_vc)/6, (dyO2k_vc+2*dyO2l_vc+2*dyO2m_vc+dyO2n_vc)/6
    DxO2_vc += dxO2_vc/NdT
    DyO2_vc += dyO2_vc/NdT

    # third-order approximation
    # Runge-Kutta k
    dxO3k_vc, dyO3k_vc = THIRD_ORDER_ANALYTICAL(DxO3_vc, DyO3_vc)
    DxO3k_vc, DyO3k_vc = DxO3_vc+dxO3k_vc/NdT/2, DyO3_vc+dyO3k_vc/NdT/2
    # Runge-Kutta l
    dxO3l_vc, dyO3l_vc = THIRD_ORDER_ANALYTICAL(DxO3k_vc, DyO3k_vc)
    DxO3l_vc, DyO3l_vc = DxO3_vc+dxO3l_vc/NdT/2, DyO3_vc+dyO3l_vc/NdT/2
    # Runge-Kutta m
    dxO3m_vc, dyO3m_vc = THIRD_ORDER_ANALYTICAL(DxO3l_vc, DyO3l_vc)
    DxO3m_vc, DyO3m_vc = DxO3_vc+dxO3m_vc/NdT, DyO3_vc+dyO3m_vc/NdT
    # Runge-Kutta n
    dxO3n_vc, dyO3n_vc = THIRD_ORDER_ANALYTICAL(DxO3m_vc, DyO3m_vc)
    dxO3_vc, dyO3_vc = (dxO3k_vc+2*dxO3l_vc+2*dxO3m_vc+dxO3n_vc)/6, (dyO3k_vc+2*dyO3l_vc+2*dyO3m_vc+dyO3n_vc)/6
    DxO3_vc += dxO3_vc/NdT
    DyO3_vc += dyO3_vc/NdT

    # fourth-order approximation
    # Runge-Kutta k
    dxO4k_vc, dyO4k_vc = FOURTH_ORDER_ANALYTICAL(DxO4_vc, DyO4_vc)
    DxO4k_vc, DyO4k_vc = DxO4_vc+dxO4k_vc/NdT/2, DyO4_vc+dyO4k_vc/NdT/2
    # Runge-Kutta l
    dxO4l_vc, dyO4l_vc = FOURTH_ORDER_ANALYTICAL(DxO4k_vc, DyO4k_vc)
    DxO4l_vc, DyO4l_vc = DxO4_vc+dxO4l_vc/NdT/2, DyO4_vc+dyO4l_vc/NdT/2
    # Runge-Kutta m
    dxO4m_vc, dyO4m_vc = FOURTH_ORDER_ANALYTICAL(DxO4l_vc, DyO4l_vc)
    DxO4m_vc, DyO4m_vc = DxO4_vc+dxO4m_vc/NdT, DyO4_vc+dyO4m_vc/NdT
    # Runge-Kutta n
    dxO4n_vc, dyO4n_vc = FOURTH_ORDER_ANALYTICAL(DxO4m_vc, DyO4m_vc)
    dxO4_vc, dyO4_vc = (dxO4k_vc+2*dxO4l_vc+2*dxO4m_vc+dxO4n_vc)/6, (dyO4k_vc+2*dyO4l_vc+2*dyO4m_vc+dyO4n_vc)/6
    DxO4_vc += dxO4_vc/NdT
    DyO4_vc += dyO4_vc/NdT

    if i%(NdT*1000) == 0:
        print(i/NdT)

txt.close()


