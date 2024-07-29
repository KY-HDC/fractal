import matplotlib.pyplot as plt
import numpy as np
import porespy as ps


im = ps.generators.sierpinski_foam(dmin=5, n=5, ndim=2)
fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.imshow(im, interpolation='none', origin='lower')
ax.axis(False)
plt.show()
plt.close()

b = ps.metrics.boxcount(im=im)
print(b)
print(b.size)
print(b.count)
print(b.slope)

fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.loglog(b.size, b.count)
ax.set_xlabel('box length')
ax.set_ylabel('number of partially filled boxes')
plt.show()
plt.close()

# linear regression
log_size = np.log10(b.size)
log_count = np.log10(b.count)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(log_size.reshape((-1, 1)), log_count)
# model.fit(b.size.reshape((-1, 1)), b.count)
print(model.intercept_) # bias
print(model.coef_)      # coefficient

log_y_hat = model.predict(log_size.reshape((-1, 1)))
y_hat = 10. ** (log_y_hat)

fig, ax = plt.subplots(1, 1, figsize=[6, 6])
ax.loglog(b.size, b.count, 'ok', markersize=2)
ax.loglog(b.size, y_hat, '-k', linewidth=0.8, label=f'slope={abs(float(model.coef_)):.3f}')

for i in range(len(b.size)):
    ax.loglog([b.size[i], b.size[i]], [b.count[i], y_hat[i]], 
              'k', linewidth=0.7, alpha=0.4
              )

# ax.loglog(
#     [(locx, locx) for locx in b.size],
#     [(b.count[i], y_hat[i]) for i in range(len(b.size))]
#     )
ax.set_xlabel('box length')
ax.set_ylabel('number of partially filled boxes')
plt.legend()
plt.show()
plt.close()
