import numpy as np
from scipy import stats, optimize

import matplotlib.pyplot as plt


def data(tau, mu1, sigma1, mu2, sigma2):
    k = 10000
    n1 = stats.norm.rvs(loc=mu1, scale=sigma1, size=int(k * tau))  # Первое нормальное распределение
    n2 = stats.norm.rvs(loc=mu2, scale=sigma2, size=int(k * (1 - tau)))  # Второе нормальное распределение
    return np.concatenate((n1, n2))


# Итоговая плотность склеенных нормальных распределений
def final_d(arg, x):
    tau, mu1, sigma1, mu2, sigma2 = arg
    d1 = stats.norm.pdf(x, loc=mu1, scale=np.abs(sigma1))
    d2 = stats.norm.pdf(x, loc=mu2, scale=np.abs(sigma2))
    res = tau * d1 + (1 - tau) * d2
    return res, d1, d2


"""<==========likelihood==========>"""


def sub_likelihood(arg, x):
    res, d1, d2 = final_d(arg, x)
    return -np.sum(np.log(np.abs(res)))  # Ищем минимум от -F(x) или максимум от F(x)


def max_likelihood(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    result = optimize.minimize(sub_likelihood, np.array([tau, mu1, sigma1, mu2, sigma2]), args=x, tol=rtol)
    return result.x


"""<==========EM==========>"""


def step(arg, x):
    """Старые параметры"""
    tau, mu1, sigma1, mu2, sigma2 = arg
    res, d1, d2 = final_d(arg, x)
    theta1, theta2 = tau * d1 / res, (1 - tau) * d2 / res
    """Новые параметры"""
    tau = (np.sum(theta1) /
           x.size)
    mu1 = (np.sum(theta1 * x) /
           np.sum(theta1))
    sigma1 = (np.sqrt((np.sum(theta1 * (x - mu1) ** 2)) /
                      np.sum(theta1)))
    mu2 = (np.sum(theta2 * x) /
           np.sum(theta2))
    sigma2 = (np.sqrt(np.sum(theta2 * (x - mu2) ** 2) /
                      np.sum(theta2)))
    return tau, mu1, sigma1, mu2, sigma2


def em_double_gauss(x, tau, mu1, sigma1, mu2, sigma2, rtol=1e-3):
    new = tau, mu1, sigma1, mu2, sigma2
    while 1:
        old = new  # Запоминаю предыдущее значение
        new = step(old, x)  # Новое значение
        if np.allclose(new, old, rtol=rtol, atol=0):  # Сравниваю новое со старым на велчину не более rtol
            return np.asarray(new)


"""<==========EM_2==========>"""


def em_double_cluster(x, uniform_dens, tau1, mu1, sigma1, tau2, mu2, sigma2, rtol=1e-3):
    def tau_(t_n, t_u, tau):
        return np.sum(t_n * (1 - tau)) / np.sum(t_n + t_u)

    def mu_(t_n):
        return [np.sum(t_n * x) / np.sum(t_n), np.sum(t_n * y) / np.sum(t_n)]

    def sigma_(t_n, mu_x, mu_y):
        return np.sqrt(np.sum(t_n * ((x - mu_x) ** 2 + (y - mu_y) ** 2)) / (2 * np.sum(t_n)))

    def uniform(tau1, tau2):
        return np.asarray(((1 - tau1 - tau2) / uniform_dens) + np.zeros(n))

    def t_n(tau, mu_x, mu_y, sigma):
        return tau / (2 * np.pi * sigma ** 2) * np.exp(- ((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))

    def t(x, tau1, mu1, sigma1, tau2, mu2, sigma2):
        t_n1 = t_n(tau1, mu1[0], mu1[1], sigma1)
        t_n2 = t_n(tau2, mu2[0], mu2[1], sigma2)
        t_u = uniform(tau1, tau2)
        return t_n1 / (t_u + t_n1 + t_n2),\
               t_n2 / (t_u + t_n1 + t_n2),\
               t_u / (t_u + t_n1 + t_n2)

    def theta(x, tau1, mu1, sigma1, tau2, mu2, sigma2):
        t_n1, t_n2, t_u = t(x, tau1, mu1, sigma1, tau2, mu2, sigma2)
        tau1, tau2 = tau_(t_n1, t_u, tau2), tau_(t_n2, t_u, tau1)
        mu1, mu2 = mu_(t_n1), mu_(t_n2)
        sigma1, sigma2 = sigma_(t_n1, mu1[0], mu1[1]), sigma_(t_n2, mu2[0], mu2[1])
        return tau1, mu1, sigma1, tau2, mu2, sigma2

    def transform(new, old):
        tau1, mu1, sigma1, tau2, mu2, sigma2 = new
        mu1, mu2 = np.sqrt(mu1[0] ** 2 + mu1[1] ** 2), np.sqrt(mu2[0] ** 2 + mu2[1] ** 2)
        tmp = tau1, mu1, sigma1, tau2, mu2, sigma2
        tau1, mu1, sigma1, tau2, mu2, sigma2 = old
        mu1, mu2 = np.sqrt(mu1[0] ** 2 + mu1[1] ** 2), np.sqrt(mu2[0] ** 2 + mu2[1] ** 2)
        tmp_old = tau1, mu1, sigma1, tau2, mu2, sigma2
        return tmp, tmp_old

    y = x[:, 1]
    x = x[:, 0]
    n = x.shape[0]
    new = tau1, mu1, sigma1, tau2, mu2, sigma2
    while 1:
        # print(new)
        old = new  # Запоминаю предыдущее значение
        new = theta(x, *old)  # Новое значение
        if np.allclose(*transform(new, old), rtol=rtol, atol=0):  # Сравниваю новое со старым на велчину не более rtol
            return {'tau1': new[0], 'mu1': new[1], 'sigma1': new[2],
                    'tau2': new[3], 'mu2': new[4], 'sigma2': new[5]}
