import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def plot_trend(target_deque, ylabel, saveas=None):
    """Plot the loss of TOP5-convergence
        Args:
            loss_deque(collections.deque): This deque contains of the dictionary of losses. deque[{'px': (epoch,)}]
    """
    # colors = sns.color_palette("blend:#7AB,#EDA", len(target_deque))
    # colors = sns.color_palette("PRGn", len(target_deque))
    colors = sns.color_palette("blend:#F55,#07B", len(target_deque))

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    xmax = 1
    ymax = 0.

    for i, px in enumerate(target_deque):
        for k, v in px.items():
            xmax = len(v) - 1
            ax.plot(range(len(v)), v, color=colors[i], linewidth=1, label=f"{k}px")
            if ymax < v.max():
                ymax = v.max()
    # print(xmax, ymax)
    ax.set_xlabel('Epochs')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax*1.05)
    # ax.xaxis.set_major_locator(ticker.IndexLocator(1, 0))
    ax.legend(loc='upper right')
    
    if saveas:
        fig.savefig(saveas)
        plt.close(fig)
    else:
        plt.show()
    
    return 0

if __name__ == "__main__":
    import numpy as np
    from collections import deque
    import matplotlib as mpl
    mpl.use('ipympl')

    l = np.array([1,2,3,4,5])
    loss_deque = deque(maxlen=5)
    for _ in range(5):
        loss = {    # px: loss array
            f"{_}": l*_,
        }
        loss_deque.appendleft(loss)
    plot_trend(loss_deque, ylabel='Loss')
    plot_trend(loss_deque, ylabel='Loss')
