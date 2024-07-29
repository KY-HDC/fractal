import jax
import jax.numpy as jnp
import numpy as np
import porespy as ps
from functools import partial

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
import os

# convergence detection
@partial(jax.vmap, in_axes=(0,), out_axes=0)
def convergence_measure(v, max_val=1e6):
    fin = jnp.isfinite(v)
    v = v * fin + max_val * (1-fin)
    v /= (v[0] + 1e-6)
    exceeds = (v > max_val)
    v = v * (1-exceeds) + max_val * exceeds
    # converged = (jnp.mean(v[-20:]) < 1)
    
    return -(1-jnp.mean(v))


# Interploating
def cdf_img(x, x_ref, buffer=0.25):
    """
    rescale x, relative to x_ref (x_ref is often the same as x), to achieve a uniform
    distribution over values with positive and negative intensities, but also to
    preserve the sign of x. This makes for a visualization that shows more
    structure.
    """
    u = jnp.sort(x_ref.ravel())
    num_neg = jnp.sum(u<0)
    num_nonneg = u.shape[0] - num_neg
    v = jnp.concatenate((jnp.linspace(-1,-buffer,num_neg), jnp.linspace(buffer,1,num_nonneg)), axis=0)
    y = jnp.interp(x, u, v)
    return -y


# plotting img
def plot_img(img, mnmx,
             figsize=(8, 8), dpi=100,
             savename=None,
             cmap='Spectral',
             title=""
             ):

    mn1, mx1, mn2, mx2 = mnmx

    img = cdf_img(img, img)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    im = ax.imshow(img,
                    extent=[mn2, mx2, mn1, mx1],
                    origin='lower',
                    vmin=-1, vmax=1,
                    cmap=cmap,
                    aspect='auto',
                    interpolation='nearest'
                    )

    fig.suptitle(title)
    fig.supxlabel('Input layer weight offset')
    fig.supylabel('Learning rate')

    rect = patches.Rectangle((mn2, mn1), mx2-mn2, mx1-mn1, linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.set_xticks(*tickslabels([mn2, mx2]))
    ax.set_yticks(*tickslabels([mn1, mx1]), rotation=90)
    im.set_extent([mn2, mx2, mn1, mx1])
    im.set_data(img)
    rect.set_xy((mn2, mn1))
    rect.set_width(mx2-mn2)
    rect.set_height(mx1-mn1)

    if savename:
        plt.savefig(savename)
    plt.close()

    return fig, ax, im


# Notation
def truncate_sci_notation(numbers):
    """
    keeping enough significant digits that the
    numbers disagree in four digits
    """

    # Convert numbers to scientific notation
    n1_sci, n2_sci = "{:.15e}".format(numbers[0]), "{:.15e}".format(numbers[1])

    # Extract the significant parts and exponents
    sig_n1, exp_n1 = n1_sci.split('e')
    sig_n2, exp_n2 = n2_sci.split('e')

    # Find the first position at which they disagree
    min_len = min(len(sig_n1), len(sig_n2))
    truncate_index = min_len

    for i in range(min_len):
        if (sig_n1[i] != sig_n2[i]) or (exp_n1 != exp_n2):
            # +4 accounts for 4 digits after the first disagreement
            truncate_index = i + 4
            if i == 0:
                truncate_index += 1 # Account for decimal point
        break

    exp_n1 = exp_n1[0] + exp_n1[2]
    exp_n2 = exp_n2[0] + exp_n2[2]
    if (exp_n1 == "+00") and (exp_n2 == "+00"):
        # don't bother with scientific notation if exponent is 0
        return [sig_n1[:truncate_index], sig_n2[:truncate_index]]

    # Truncate and reconstruct the scientific notation
    truncated_n1 = "{}e{}".format(sig_n1[:truncate_index], exp_n1)
    truncated_n2 = "{}e{}".format(sig_n2[:truncate_index], exp_n2)

    return [truncated_n1, truncated_n2]

def tickslabels(mnmx):
    return mnmx, truncate_sci_notation(10.**np.array(mnmx))


### FD calculation ###

# Measure the fractal dim
def extract_edges(X):
    """
    define edges as sign changes in the scalar representing convergence or
    divergence rate -- on one side of the edge training converges,
    while on the other side of the edge training diverges
    """

    Y = jnp.stack((X[1:,1:], X[:-1,1:], X[1:,:-1], X[:-1,:-1]), axis=-1)
    Z = jnp.sign(jnp.max(Y, axis=-1)*jnp.min(Y, axis=-1))
    return Z<0

def estimate_fractal_dimension(img, resolution, saveas=None):
    img = img.reshape((resolution, resolution))
    edge = extract_edges(img)
    bc = ps.metrics.boxcount(edge)
    mfd = np.median(bc.slope)

    if saveas:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        ax1.set_yscale('log')
        ax1.set_xscale('log')
        ax1.set_xlabel('box edge length')
        ax1.set_ylabel('number of boxes spanning phases')
        ax1.plot(bc.size, bc.count, '-o', alpha=0.7, label=f'mfd={mfd:.3f}')
        plt.legend()
        plt.savefig(saveas)
        plt.cla()
        plt.clf()
        plt.close()

    return mfd

# Making gif
def animate_sketches(directory):
    fnames = list(sorted(os.listdir(directory + '*.png')))
    frames = [imageio.imread(fn) for fn in fnames]
    dir_export = "/".join(directory.split('/')[:-1])
    imageio.mimsave(dir_export, frames, format='GIF', duration_rate=1)
