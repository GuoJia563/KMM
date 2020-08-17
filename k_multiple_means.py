"""
Created on Sat May  2 09:20:19 2020
K-Multiple-Means方法，处理多原型的聚类问题
论文题目：K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters
出自 KDD论文集 2019
2020/5/23 20：54 能够实现一个双月形状的连线二部图了呢 
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import time
import math

def dbmoon(N, d, r, w):
    """
    生成一个two-moon dataset
    """
    N1 = 10*N
    w2 = w/2
    done = True
    data = np.empty(0)
    while done:
        #generate Rectangular data
        tmp_x = 2*(r+w2)*(np.random.random([N1, 1])-0.5)
        tmp_y = (r+w2)*np.random.random([N1, 1])
        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
        tmp_ds = np.sqrt(tmp_x*tmp_x + tmp_y*tmp_y)
        #generate double moon data ---upper
        idx = np.logical_and(tmp_ds > (r-w2), tmp_ds < (r+w2))
        idx = (idx.nonzero())[0]
 
        if data.shape[0] == 0:
            data = tmp.take(idx, axis=0)
        else:
            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
        if data.shape[0] >= N:
            done = False
    #print (data)
    db_moon = data[0:N, :]
    #print (db_moon)
    #generate double moon data ----down
    data_t = np.empty([N, 2])
    data_t[:, 0] = data[0:N, 0] + r
    data_t[:, 1] = -data[0:N, 1] - d
    db_moon = np.concatenate((db_moon, data_t), axis=0)
    return db_moon

def A_initialize(X_i,m_i):
    """
    对原型矩阵进行初始化
    随机挑选m_i个数据点作为初始的原型点
    X_i为数据点矩阵
    输出为原型矩阵A_i
    """
    A_i=[]
    selected=[]
    for i in range(0,X_i.shape[0]):
        selected.append(i)
    select=[]  #用于存放随机挑选出的数据点在X_i中的编号
    for i in range (0,m_i):
        ra=random.choice(selected)
        select.append(ra)
        selected.remove(ra)
    for i in range(0,m_i):
        A_i.append( X_i[select[i]].tolist()[0] )
    A_i=np.mat(A_i)
    #plt.plot(A[0][ 0], A[0][ 1], 'r*')
   # plt.show()
    
    return A_i

def distant_calculation(A_dis,Xi_dis):
    """
    A_dis为m行矩阵，一行是一个原型点的坐标
    该函数返回m个原型到数据点xi_dis的距离
    返回值应是一个向量，m*1的，每个元素为一个距离数值
    """
    d_dis=[]
    for i in range(0,len(A_dis)):
        d_dis_i= np.linalg.norm(A_dis[i] - Xi_dis)
        d_dis.append(d_dis_i)
    return d_dis

def distant_calculation1(A_dis,Xi_dis):
    """
    A_dis为m行矩阵，一行是一个原型点的坐标
    该函数返回m个原型到数据点xi_dis的距离 的平方
    返回值应是一个列表，m维的，每个元素为一个距离数值
    """
    d_dis=[]
    for i in range(0,len(A_dis)):
        d_dis_i= math.pow( np.linalg.norm(A_dis[i] - Xi_dis) ,2)
        d_dis.append(d_dis_i)
    return d_dis

def ThSerialCalcu(huang_thser):
    """
    对一个向量进行排序
    输入huang是一个向量
    返回值为一个列表，是其排序结果，用元素序号表示的
    """
    selected=[]
    for i in range(0,len(huang_thser)):
        selected.append(i)
    th_serial=[]
    for i in range(0,len(huang_thser)):
        hu=selected[0]
        for j in selected:
            if huang_thser[j]<huang_thser[hu]:
                hu=j
        th_serial.append(hu)
        selected.remove(hu)
    return th_serial

def v_calculate(F_vcal,D_diagonal_vcal,i_vcal,m_vcal,n_vcal):
    """
    专门用于计算问题（20）处用到的v向量
    输入F_vcal:(2*N+m)*k，D_diagonal_vcal:n+m维的数列，元素是矩阵D的对角线元素
    返回值v_vcal:m维列表
    """
    v_vcal=[]
    for j in range(0,m_vcal):
        Subtracted=F_vcal[i_vcal]/math.sqrt(D_diagonal_vcal[i_vcal])  ##
        Minus=F_vcal[n_vcal+j]/math.sqrt(D_diagonal_vcal[n_vcal+j])
        v_vcal.append( math.pow( np.linalg.norm(Subtracted - Minus) ,2) )  #这个平方不平方的问题。。。？？？
    
    return v_vcal
    
start_time = time.time()    

c=math.sqrt( 2 )/2  #会用到的常数
N = 200 
n=2*N
data = dbmoon(N, -2, 5, 3)  #这里的data是一个多维数组，ndarray类型，（2N)行*2列的
#画数据点
plt.plot(data[0:N, 0], data[0:N, 1], 'r*', data[N:2*N, 0], data[N:2*N, 1], 'b*')
plt.show()
#初始化
X=np.mat(data)  #数据点矩阵，(2N)行*2列，即有2*N个二维的数据点
k=2  #kmm，k指的是最终想要把数据分成几个聚类，双月数据集下，k固定为2
m=30  #m，指的是欲分成m个子聚类,即有m个原型
λ=10000  #参数，λ足够大时，将能够满足rank约束，即有k个聚类的约束
A=A_initialize(X,m)  #矩阵A是原型矩阵，对其进行初始化，方法是随机选m个数据点
kl=5  #k~,指一个数据点可以有几个邻原型
S=np.mat(np.zeros((n,m)))  #S的第ij个元素指的是第i个数据点与第j个原型相连接的概率，这里初始化成了全0矩阵
#画初始原型点
plt.plot(A[0:20, 0], A[0:20, 1], 'r*')
plt.show() 

plt.plot(data[0:N, 0], data[0:N, 1], 'b+', data[N:2*N, 0], data[N:2*N, 1], 'b+',A[0:20, 0], A[0:20, 1], 'r*')
plt.show() 

#大循环
huangbaoyu=1
while huangbaoyu==1:  #收敛条件：A矩阵不在更新变化
#for huangbaoyu in range(0,20):
    #用计算式（4）的方法计算S矩阵，即概率矩阵
    #Sij表示 第i个数据点xi 与 第j个原型aj 相连作为邻原型的概率
    #对每一个数据点xi，相邻原型的分配是独立的，分别对其邻原型进行分配
    for i in range(0,n):
        distant=distant_calculation(A,X[i])  #m个原型到第i个数据点的距离
        distant_th_Serial=ThSerialCalcu(distant)  #对distant矩阵进行从小到大的排序，得到顺序号，比如9，4，5，就是第9个元素第一小
        distant_th=[]  #排序好的distant矩阵
        for j in range(0,m):
            distant_th.append(distant[ distant_th_Serial[j] ])
        for j in range(0,kl):
            fenzi=distant_th[kl+1]-distant_th[j]
            he=0
            for ii in range(0,kl):
                he+=distant_th[ii]
            fenmu=kl*distant_th[kl+1]-he
            S[i,distant_th_Serial[j] ]=fenzi/fenmu
        for j in range(kl,m):
            S[i,distant_th_Serial[j] ]=0
    
    #λ=10000
    Lsl_rank=n+m
    #子循环
    while Lsl_rank!=(n+m-k):
    #for huangbaoyuyu in range (0,15):
        #更新D_U和D_V
        du=[]
        for i in range(0,n):
            hu=0
            for j in range(0,m):
                hu+=S[i,j]
            du.append(hu)
        D_U=np.mat(np.diag(du))
        dv=[]
        for j in range(0,m):
            hu=0
            for i in range(0,n):
                hu+=S[i,j]
            dv.append(hu)
        D_V=np.mat(np.diag(dv))
        
        #算F
        du=[]
        for i in range(0,n):
            du.append(1/ ( math.sqrt( D_U[i,i] ) ) )   ##
        D_U1=np.mat(np.diag(du))
        dv=[]
        for i in range(0,m):
            dv.append(1/ ( math.sqrt( D_V[i,i] ) ) )
        D_V1=np.mat(np.diag(dv))
        Sl=D_U1*S*D_V1
        u,s,vt=np.linalg.svd(Sl)  #得到的s是从大到小排列的array
        U=[]
        V=[]
        for i in range(0,k):
            U.append( u.T[i].tolist()[0] )
            V.append( vt[i].tolist()[0] )
        U=( np.mat(U)*c ).T  #U:400*2
        V=( np.mat(V)*c ).T  #V:m*2
        F=np.vstack((U, V))  #对U,V矩阵进行垂直方向的连接 F应为k列的矩阵
        
        #更新S的每一行
        D_diagonal=[]
        for i in range(0,n):
            D_diagonal.append(D_U[i,i])
        for i in range(0,m):
            D_diagonal.append(D_V[i,i])
            
        for i in range(0,n):
            #先算di~
            v=v_calculate(F,D_diagonal,i,m,n)  #计算v，m维列表
            v=np.mat(v)
            distant_l=np.mat( distant_calculation1(A,X[i]) )+λ*v  #di~，1*20
            distant_l=distant_l.T
            distant_lth_Serial=ThSerialCalcu(distant_l)  #对distant矩阵进行从小到大的排序，得到顺序号，比如9，4，5，就是第9个元素第一小
            distant_lth=[]  #排序好的distant矩阵
            for j in range(0,m):
                distant_lth.append(distant_l[ distant_lth_Serial[j] ])
            for j in range(0,kl):
                fenzi=distant_lth[kl+1]-distant_lth[j]
                he=0
                for ii in range(0,kl):
                    he+=distant_lth[ii]
                fenmu=kl*distant_lth[kl+1]-he
                S[i,distant_lth_Serial[j] ]=fenzi/fenmu
            for j in range(kl,m):
                S[i,distant_lth_Serial[j] ]=0  #验证过了，这么更新之后的S，满足S*向量1=向量1，即S每行的和均为1
                
        #计算子循环的收敛条件是否达到
        #与S有关的归一化拉普拉斯矩阵Ls~的秩=（n+m-k）时，停止迭代
        I=np.mat(np.eye(n+m))  #单位阵，(n+m)*(n+m)
        dl=[]
        for i in range(0,n):
            dl.append(D_U1[i,i])
        for i in range(0,m):
            dl.append(D_V1[i,i])
        Dl=np.mat(np.diag(dl))
        P=np.mat(np.zeros((n+m,n+m)))  #构造P矩阵
        for i in range(0,2*N):
            for j in range(n,n+m):
                P[i,j]=S[i, (j-2*N) ]
        for i in range(n,n+m):
            for j in range(0,n):
                P[i,j]=S[ j, (i-(n))]
        Lsl=I-Dl*P*Dl  #430*430
        Lsl_rank=np.linalg.matrix_rank(Lsl) #返回矩阵的秩，返回秩函数是对的  
#        if Lsl_rank<(n+m-k):
#            λ=λ*1.2
#        elif Lsl_rank>(n+m-k):
#            λ=λ*0.8
            
    A1=np.mat(np.zeros((A.shape[0],A.shape[1])))
    #更新A
    for j in range(0,m):
        fenzi=[0,0]
        fenmu=0
        for i in range(0,n):
            fenzi+=S[i,j]*X[i]
        for i in range(0,n):
            fenmu+=S[i,j]
        A1[j]=fenzi/fenmu
    sos=0
    huangbaoyu=1
    for i in range(0,A.shape[0]):
         for j in range(0,A.shape[1]):
            if math.pow( (A[i,j]-A1[i,j]),2 )>0.01:
                sos+=1
    if sos<1:
        huangbaoyu=0
    A=A1

#把最后的聚类结果展示出来
#试试在数据点和原型之间把线连起来
for i in range(0,2*N):
    for j in range(0,m):
        if S[i,j]!=0:
            plt.plot([X.tolist()[i][0],A.tolist()[j][0]],[X.tolist()[i][1],A.tolist()[j][1]],linewidth = '0.2',color='#3399FF')
plt.plot(data[0:N, 0], data[0:N, 1], 'b+', data[N:2*N, 0], data[N:2*N, 1], 'b+',A[0:20, 0], A[0:20, 1], 'm*')
plt.show()

end_time= time.time()
print('程序运行时间:',end_time-start_time,'s')

##进行svd分解的实验
#q=np.mat([[1,2,3],[4,5,6],[2,3,4],[2,2,2]])
#w,e,r=np.linalg.svd(q)
#Q=[]
#W=[]
#for i in range(0,2):
#    Q.append( w.T[i].tolist()[0] )
#    W.append( r[i].tolist()[0] )
#Q0=( np.mat(Q)*c ).T  #U:400*2
#W0=( np.mat(W)*c ).T  #V:20*2
#Q1=( np.mat(Q) ).T  #U:400*2
#W1=( np.mat(W) ).T  #V:20*2
#
##看矩阵Q与S是否完全相同的实验
#Q=np.mat(np.zeros((S.shape[0],S.shape[1])))
#for i in range(0,S.shape[0]):
#    for j in range(0,S.shape[1]):
#        Q[i,j]=S[i,j]
#
#sos=0
#for i in range(0,S.shape[0]):
#    for j in range(0,S.shape[1]):
#        if Q[i,j]!=S[i,j]:
#            sos=1
#print(sos)