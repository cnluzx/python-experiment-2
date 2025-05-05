from matplotlib import pyplot as plt
import time
import numpy as np
from matplotlib import animation


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


####################################
## fixed point iteration function ##
def fixpt(f, x, epsilon=1E-5, N=500, store=False):
    y = f(x)
    n = 0
    if store:
        Values = [(x, y)]
    while abs(y - x) >= epsilon and n < N:
        x = f(x)
        n += 1
        y = f(x)
        if store:
            Values.append((x, y))
    if store:
        return y, Values
    else:
        if n >= N:
            return "No fixed point for given start value"
        else:
            return x, n, y
        
# define f(x)
def f(x):
    return ((-58 * x - 3) / (7 * x ** 3 - 13 * x ** 2 - 21 * x - 12)) ** (1 / 2)


def update1(i):
    a = 0.2 / (4 ** (i))
    x, y = points[i]
    print(y)
    result_x = round(x, 6)
    result_y = round(y, 6)
    plt.xlim(x - a, x + a)
    plt.ylim(f(x) - a, f(x) + a)
    plt.plot([x, x], [x, y], 'g')

    plt.text(x, y, (result_x, result_y))
    



def update2(i):
    a = 0.2 / (4 ** (i))
    x, y = points[i]

    plt.plot([x, y], [y, y], 'g')

    plt.xlim(y - a, y + a)
    plt.ylim(y - a, y + a)

    plt.title(f"第{i + 1}次迭代")
    
res, points = fixpt(f, 1.5, store=True)

xx = np.arange(1.3, 2.8, 1e-5)
fig = plt.figure()
plt.grid()

plt.plot(xx, f(xx), 'b')
plt.plot(xx, xx, 'r')

plt.pause(2)
ani1 = animation.FuncAnimation(fig, update1, frames=np.arange(0, len(points)), interval=2000, blit=False, repeat=False)
plt.pause(2)
ani2 = animation.FuncAnimation(fig, update2, frames=np.arange(0, len(points)), interval=2000, blit=False, repeat=False)

# show result
plt.show()

###
