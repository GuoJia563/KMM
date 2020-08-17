#two moon dataset #
import numpy as np
import matplotlib.pyplot as plt

def dbmoon(N=100, d=2, r=10, w=2):
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

N = 200  #
d = -2
r = 5
w = 3
#a = 0.1
#num_MSE = []
#num_step = []
data = dbmoon(N, d, r, w)  #这里的data是一个多维数组，ndarray类型，（2N)行*2列的
#type(data)
#print("data:",data)
#data.size

plt.plot(data[0:N, 0], data[0:N, 1], 'r*', data[N:2*N, 0], data[N:2*N, 1], 'b*')
plt.show()
