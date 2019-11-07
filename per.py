import astropy.units as u
from astropy.coordinates import SkyCoord
from astroquery.vizier import Vizier
import numpy as np

import mixfit
import matplotlib.pyplot as plt
import json
from scipy.stats import norm
from mpl_toolkits.mplot3d import Axes3D

center_coord = SkyCoord('02h21m00s +57d07m42s')
vizier = Vizier(column_filters={'Bmag': '<13'},  # число больше — звёзд больше
                row_limit=10000)
stars = vizier.query_region(center_coord,
                            width=1.5 * u.deg,
                            height=1.5 * u.deg,
                            catalog='USNO-A2.0')[0]
# print(stars)
ra = stars['RAJ2000'].data  # прямое восхождение, аналог долготы
dec = stars['DEJ2000'].data  # склонение, аналог широты
x1 = (ra - ra.mean()) * np.cos(dec / 180 * np.pi)
x2 = dec


def data(tmp):
    for i in range(len(x1)):
        tmp.append([x1[i], x2[i]])
    tmp = np.asarray(tmp)
    return tmp


def plot():
    global x
    fig = plt.figure()

    """ax1 = fig.add_subplot(221)
    a = np.linspace(x1.min(), x1.max(), 100)
    ax1.plot(a, norm.pdf(a, res['mu1'][0], res['sigma1']))
    ax1.plot(a, norm.pdf(a, res['mu2'][0], res['sigma2']))
    ax1.hist(x1, 100)
    ax1.title.set_text('Распределение ra')

    ax2 = fig.add_subplot(222)
    a = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
    ax2.plot(a, norm.pdf(a, res['mu1'][1], res['sigma1']))
    ax2.plot(a, norm.pdf(a, res['mu2'][1], res['sigma2']))
    ax2.hist(x[:, 1], 100)
    ax2.title.set_text('Распределение dec')"""

    ax3 = fig.add_subplot(121, projection='3d')
    a = np.linspace(x1.min(), x1.max(), 100)
    ax3.plot(a, norm.pdf(a, res['mu1'][0], res['sigma1']), zs=x[:, 1].max(), zdir='y')
    ax3.plot(a, norm.pdf(a, res['mu2'][0], res['sigma2']), zs=x[:, 1].max(), zdir='y')
    ax3.plot(x1, x[:, 1], '.', color='dodgerblue', label='stars')
    a = np.linspace(x[:, 1].min(), x[:, 1].max(), 100)
    ax3.plot(a, norm.pdf(a, res['mu1'][1], res['sigma1']), zs=x1.min(), zdir='x')
    ax3.plot(a, norm.pdf(a, res['mu2'][1], res['sigma2']), zs=x1.min(), zdir='x')
    ax3.set_xlabel('ra', color='r')
    ax3.set_ylabel('dec', color='r')
    ax3.title.set_text('Вид в 3D')

    ax4 = fig.add_subplot(122)
    ax4.hist2d(*x.T, bins=12)
    ax4.plot(*x.T, '.', color='dodgerblue')
    ax4.plot(res['mu1'][0], res['mu1'][1], 'o', color='r')
    ax4.plot(res['mu2'][0], res['mu2'][1], 'o', color='r')
    ax4.set_xlabel('ra', color='r')
    ax4.set_ylabel('dec', color='r')
    ax4.title.set_text('Космос')

    fig.set_size_inches(12, 6)
    plt.savefig('per.png', quality=100, figsize=(1000, 1000))
    plt.show()


def save():
    global res
    global x
    x = []
    x = data(x)
    x[:, 1] = x[:, 1] - x[:, 1].mean()
    arg = 0.3, [-0.5, -0.4], 0.5, 0.6, [0.5, 0.5], 0.7
    res = mixfit.em_double_cluster(x, 1.5 * 1.5, *arg)  # (max(x[:, 1]) - min(x[:, 1])) * (max(x[:, 0]) - min(x[:, 0]))
    print(res)
    plot()
    res['mu1'][1] += dec.mean()
    res['mu2'][1] += dec.mean()
    res['mu1'][0] += res['mu1'][0] / np.cos(res['mu1'][1] / 180 * np.pi) + ra.mean()
    res['mu2'][0] += res['mu2'][0] / np.cos(res['mu1'][1] / 180 * np.pi) + ra.mean()
    with open('per.json', 'w') as file:
        json.dump({"size_ratio": 1.5, "clusters": [
                       {"center": {
                           'ra': res['mu1'][0],
                           'dec': res['mu1'][1]},
                           "sigma": res['sigma1'],
                           "tau": res['tau1']},
                       {"center": {
                           'ra': res['mu2'][0],
                           'dec': res['mu2'][1]},
                        "sigma": res['sigma2'],
                        "tau": res['tau2']}]},
                  file,
                  indent=4,
                  separators=(',', ': '))


save()
