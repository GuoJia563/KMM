"""
Created on Fri Jun  5 08:31:50 2020
K-Multiple-Means方法，处理多原型的聚类问题
论文题目：K-Multiple-Means: A Multiple-Means Clustering Method with Specified K Clusters
出自 KDD论文集 2019
将算法部分编成函数，然后使用wine数据集对其进行测试
使用双月形数据集进行测试的代码在最后，已设为注释
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import load_wine  #sklearn数据集中的wine数据集

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

def KMM(X,k,m,λ,kl):
    """
    输入量：
    X为被处理的数据点所组成的矩阵，一行是一个点
    k为输入的先验知识，欲将数据点分成k个大聚类
    m自己设置，指的是欲分成m个子聚类,即有m个原型
    λ为参数，λ足够大时，将能够满足rank约束，即有k个聚类的约束
    kl,即k~,指一个数据点可以有几个邻原型
    输出量：
    聚类结果
    需要返回的结果为列表results[]
    原型点矩阵A
    连接矩阵S
    """
    n=X.shape[0]
    A=A_initialize(X,m)  #矩阵A是原型矩阵，对其进行初始化，方法是随机选m个数据点
    #kl=5  #k~,指一个数据点可以有几个邻原型
    S=np.mat(np.zeros((n,m)))  #S的第ij个元素指的是第i个数据点与第j个原型相连接的概率，这里初始化成了全0矩阵
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
            U=( np.mat(U)*c ).T  #U:n*d
            V=( np.mat(V)*c ).T  #V:m*d
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
            for i in range(0,n):
                for j in range(n,n+m):
                    P[i,j]=S[i, (j-n) ]
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
            fenzi=[]  #维数需要修改的地方找到了
            for p in range(0,A.shape[1]):
                fenzi.append(0)
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
    #输出
    result=[]  #输出         
    result.append(A)
    result.append(S)
    return result

def ClusterResult_Standardize(k,result_list):
    """
    输入：聚类数k，KMM函数的输出，一个list，内容有两项，分别是原型点矩阵A和连接矩阵S
    输出：一个list，output[]
    聚到一起的数据点cluster_result[]
    按聚类分的原型点prototype_result[]
    该输出是一个三层的列表
    output结果
    output[0]数据点聚类结果 output[1]原型点聚类结果
    output[0][0]第一个聚类内部的数据点们 output[1][0]第一个聚类内部的原型点们
    output[0][0][0]第一个聚类内部的第一个数据点在数据点矩阵内的排序序号/行数
    """
    output=[]
    A=result_list[0]
    S=result_list[1]
    m=len(A)
    n=len(S)
    cluster_result=[]  #聚类数据点,其每个元素也是列表,内部存放一个簇中的点在数据点矩阵中的行数
    prototype_result=[]  #聚类原型点，其每个元素也是列表，内部存放一个簇中原型点在A矩阵中的行数
    X_selected=[]  #存放X矩阵的所有行数，备选
    for i in range(0,n):
        X_selected.append(i)
    A_selected=[]  #存放A矩阵的所有行数，备选
    for i in range(0,m):
        A_selected.append(i)
    for kk in range(0,k):  #每个大聚类     
        child_cluster=[]
        child_prototype=[]
        data_new=[]
        prototype_new=[]
        if X_selected!=[]:
            child_cluster.append(X_selected[0])
            data_new.append(X_selected[0])
            X_selected.remove(X_selected[0])
        baoyu=1
        while baoyu==1:
            for i in range(0,len(data_new)):
                for j in range(0,m):
                    if S[data_new[i],j]!=0 and (j in A_selected):
                        child_prototype.append(j)
                        prototype_new.append(j)
                        A_selected.remove(j)
            data_new.clear()
            for i in range(0,len(prototype_new)):
                for j in range(0,n):
                    if S[j,prototype_new[i]]!=0 and (j in X_selected):
                        child_cluster.append(j)
                        data_new.append(j)
                        X_selected.remove(j)
            prototype_new.clear()
            if data_new==[]:  #当不再有新的数据点被找出来，则这一聚类已被找完
                baoyu=0
        child_cluster.sort()
        cluster_result.append(child_cluster)
        child_prototype.sort()
        prototype_result.append(child_prototype)
    output.append(cluster_result)
    output.append(prototype_result)
    return output

def KMM_test(ClusterResult,TestTarget,k):
    """
    验证聚类结果的好坏
    输入：
    简化后的聚类结果，ClusterResult_Standardize函数的输出的第一项，数据点聚类结果
    标签,是一个数组
    k,聚类数，即标签的种类数
    输出：
    聚类分类效果，保存的量是在生成的每一个聚类中的每一种标签对应数据的个数
    """
    result=[]
    #还是应该先把k种标签找出来，然后具体分
    label=[]  #唯一地存放着全部种类的标签
    label.append(TestTarget[0])
    for i in range(0,len(TestTarget)):
        if (TestTarget[i] not in label):
            label.append(TestTarget[i])
        if (len(label)==k):
            break
    #开始统计
    for i in range(0,k):  #ClusterResult的第i项，即第i个聚类
        number_tar=[]  #数列，用于存放该聚类里每种标签的个数
        for tar in range(0,k):
            number_tar.append(0)  #number_tar的初始值均为0，从0开始计数
        for tar in range(0,k):
            for j in range(0,len(ClusterResult[i])):
                if TestTarget[ ClusterResult[i][j] ]==label[tar]:
                    number_tar[tar]+=1
        result.append(number_tar)
    return result

def data_normalize(mat):
    """
    对数据点矩阵的每列数据（即每一维的特征）进行归一化处理
    """
    mat1=np.mat(np.zeros((mat.shape[0],mat.shape[1])))
    for j in range(0,mat.shape[1]):
        num=0
        for i in range(0,mat.shape[0]):
            num+=mat[i,j]
        for i in range(0,mat.shape[0]):
            mat1[i,j]=(mat[i,j]*1000)/num
    return mat1

#用wine数据集进行测试
c=math.sqrt( 2 )/2  #会用到的常数
#数据集，wine.data是数据，178*13的ndarray；wine.target是标签，内容为0，1，2，的ndarray
wine = load_wine()
data_mat=np.mat(wine.data)  #178*13的矩阵
DataMatNormalized=data_normalize(data_mat)#进行数据的标准归一化处理
re=KMM(DataMatNormalized,3,20,10000,5)  #用标准化后的数据进行聚类
#re=KMM(data_mat,3,20,10000000,5)  #用数据集中的数据直接进行聚类
cluResult=ClusterResult_Standardize(3,re)
final_result=KMM_test(cluResult[0],wine.target,3)

##以下为用双月数据集进行测试的部分
#def dbmoon(N, d, r, w):
#    """
#    生成一个two-moon dataset
#    """
#    N1 = 10*N
#    w2 = w/2
#    done = True
#    data = np.empty(0)
#    while done:
#        #generate Rectangular data
#        tmp_x = 2*(r+w2)*(np.random.random([N1, 1])-0.5)
#        tmp_y = (r+w2)*np.random.random([N1, 1])
#        tmp = np.concatenate((tmp_x, tmp_y), axis=1)
#        tmp_ds = np.sqrt(tmp_x*tmp_x + tmp_y*tmp_y)
#        #generate double moon data ---upper
#        idx = np.logical_and(tmp_ds > (r-w2), tmp_ds < (r+w2))
#        idx = (idx.nonzero())[0]
# 
#        if data.shape[0] == 0:
#            data = tmp.take(idx, axis=0)
#        else:
#            data = np.concatenate((data, tmp.take(idx, axis=0)), axis=0)
#        if data.shape[0] >= N:
#            done = False
#    #print (data)
#    db_moon = data[0:N, :]
#    #print (db_moon)
#    #generate double moon data ----down
#    data_t = np.empty([N, 2])
#    data_t[:, 0] = data[0:N, 0] + r
#    data_t[:, 1] = -data[0:N, 1] - d
#    db_moon = np.concatenate((db_moon, data_t), axis=0)
#    return db_moon
#c=math.sqrt( 2 )/2  #会用到的常数
#N = 200
#data = dbmoon(N, -2, 5, 3) 
#data_mat=np.mat(data)  #数据点矩阵，(2N)行*2列，即有2*N个二维的数据点
#re=KMM(data_mat,2,30,10000000,5)
#cluResult=ClusterResult_Standardize(2,re)
##用上述函数做出来的聚类结果图
#plt.plot(data[cluResult[0][0][0:N], 0], data[cluResult[0][0][0:N], 1], 'b+',data[cluResult[0][1][0:N], 0], data[cluResult[0][1][0:N], 1], 'm+')
#plt.show()
##用直接连线的形式做出来的聚类结果图
#for i in range(0,2*N):
#    for j in range(0,30):
#        if re[1][i,j]!=0:
#            plt.plot([data_mat.tolist()[i][0],re[0].tolist()[j][0]],[data_mat.tolist()[i][1],re[0].tolist()[j][1]],linewidth = '0.2',color='#3399FF')
#plt.plot(data[0:N, 0], data[0:N, 1], 'b+', data[N:2*N, 0], data[N:2*N, 1], 'b+',re[0][0:20, 0], re[0][0:20, 1], 'm*')
#plt.show()