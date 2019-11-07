import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import mixfit
from scipy import stats


"""<==========TEST EM_1==========>"""


tau = 0.5
mu1 = 0.1
sigma1 = 0.4
mu2 = 0.5
sigma2 = 0.3

dat = mixfit.data(tau, mu1, sigma1, mu2, sigma2)

print('data_shape: ', dat.shape)
x1 = mixfit.max_likelihood(dat, 0.3, 0.3, 0.5, 0.3, 0.4)
print('max_likelihood: ', np.round(x1, 5))
x2 = mixfit.em_double_gauss(dat, 0.3, 0.3, 0.5, 0.3, 0.4)
print('EM_1 method: ', np.round(x2, 5))


"""<==========TEST EM_2==========>"""


tau = 0.4
mu1 = np.array([0.5, 0.5])
mu2 = np.array([-0.5, -0.5])
sigma1 = 0.4
sigma2 = 0.3
n = 10000
n_1 = int(n * tau)
n_2 = n - n_1

x_n1 = stats.multivariate_normal(mu1, sigma1**2).rvs(n_1)
x_n2 = stats.multivariate_normal(mu2, sigma2**2).rvs(n_2)
x = np.vstack((x_n1, x_n2))
plt.hist2d(*x.T)
plt.plot(*x.T, '.', color='red')


# print('x.shape: ', x.shape)
print('EM_1 method_X: ', np.round(mixfit.em_double_gauss(x[:, 0], 0.6, -0.5, 0.3, 0.5, 0.4), 5))
print('EM_1 method_Y: ', np.round(mixfit.em_double_gauss(x[:, 1], 0.6, -0.5, 0.3, 0.5, 0.4), 5))
arg = 0.6, [-0.5, -0.5], 0.4, 0.3, [0.5, 0.5], 0.5
print('EM method №2: ', mixfit.em_double_cluster(x, (max(x[:, 1]) - min(x[:, 1])) * (max(x[:, 0]) - min(x[:, 0])), *arg))
plt.show()



