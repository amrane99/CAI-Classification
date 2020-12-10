# ------------------------------------------------------------------------------
# Plots results.
# Substantial portions from https://github.com/camgbus/medical_pytorch project.
# ------------------------------------------------------------------------------

import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_numpy(result, save_path=None, save_name=None, 
    title=None, ending='.png', x_name='Epoch', y_name='Value',
    ylog=False, figsize=(10,5), xints=int, yints=int):
    """Plots a dataframe

    Args:
        save_path (str): path to save plot. If None, plot is shown.
        save_name (str): name with which plot is saved
        title (str): the title that will appear on the plot
        x_name (str): the name of the x axis and column name in df
        y_name (str): the name of the y axis and column name in df
        ending (str): can be '.png' or '.svg'
        ylog (bool): apply logarithm to y axis
        figsize (tuple[int]): figure size
    """
    df = result
    assert type(xints)==type(yints), 'xints and yints need to be the same type!'
    df[x_name] = df[x_name].astype(xints)
    df[y_name] = df[y_name].astype(yints)

    # Start a new figure so that different plots do not overlap
    plt.figure()
    sns.set(rc={'figure.figsize':figsize})
    # Plot
    ax = sns.lineplot(x=df[x_name], 
        y=df[y_name], 
        alpha=0.7, 
        data=df)
    ax = sns.scatterplot(x=df[x_name], 
        y=df[y_name], 
        alpha=1., 
        data=df)
    # Optional logarithmic scale
    if ylog:
        ax.set_yscale('log')
    # Style legend
    titles = ['Metric', 'Data']
    
    # Set title
    if title:
        ax.set_title(title)

    # Save image
    if save_path:
        file_name = save_name if save_name is not None else result.name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = file_name.split('.')[0]+ending
        plt.savefig(os.path.join(save_path, file_name), facecolor='w', 
            bbox_inches="tight", dpi = 300)