import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 99, 10)

plt.figure()
plt.plot(range(len(x)), x)
for i in range(len(x)):
    plt.text(i, x[i], f'#{i}\n★', verticalalignment='bottom', horizontalalignment='center')

plt.xlim(0, len(x))
plt.ylim(0, 99)
# plt.savefig('./testing/1.png')
# plt.plot(range(len(x)), -x, 'k*')
# plt.savefig('./testing/2.png')

plt.show()