import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.ticker import ScalarFormatter

# theme
plt.rc('legend', fontsize='large')
plt.rc('text', usetex=True)
sns.set_context('paper')
sns.set_style('ticks')
colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#ffff99']
sns.set_palette(palette=colors)

nf = ["Classifier", "Dropper", "Redirector", "Scheduler", "Shaper"]
title_sec = ['anf', 'vnf']
res_sec = ['cpu', 'mem']
lamb = [1, 50, 100, 150]
limx = [[50, 100], [10000, 20000], [10000, 20000], [10000, 20000]]
ft = 12

data_folder = r"../data/"


def plot_loading_time():
    load = pd.read_csv(data_folder + "loading_time.csv")
    f = plt.figure(figsize=(4, 1.8))
    ax = f.add_subplot(111)
    sns.barplot(x="Time", y="Function", hue="Type", data=load, ax=ax)
    ax.set_xscale('symlog')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("")
    ax.grid(True)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    plt.savefig(r'../figs/fig2.svg', bbox_inches='tight')
    plt.show()


def plot_resources_usage():
    figname = ['c', 'd']
    for i in range(2):
        f, ax = plt.subplots(1, 2, figsize=(4, 1.8))
        for j in range(2):
            load = pd.read_csv(data_folder + str(title_sec[i]) + "_" + str(res_sec[j]) + ".csv")
            load[str(res_sec[j])] /= 100
            sns.lineplot(x="rate", y=res_sec[j], hue="nf", data=load, ax=ax[j], style="nf", legend=False)
            ax[j].set_title("")
            ax[j].set_xlabel(r'$\lambda$')
            ax[j].set_ylabel(str(res_sec[j].upper()) + ' usage')
            ax[j].set_ylim(0, 1.1)
            ax[j].set_xticks(lamb)
            ax[j].set_xticklabels(lamb)
            ax[j].grid(True)
            ax[j].legend(nf, loc=0, ncol=1, fontsize=ft, prop={'size': 7})
        plt.tight_layout()
        plt.savefig(r'../figs/fig3' + figname[i] + '.svg', bbox_inches='tight')
        plt.show()


def plot_processing_time():
    figname = ['a', 'b']
    for u in range(2):
        f, ax = plt.subplots(1, 4, figsize=(8, 2))
        for j in range(4):
            load = pd.read_csv(data_folder + str(title_sec[u]) + "_" + str(lamb[j]) + ".csv")
            sns.lineplot(x="pt", y="cdf", hue="type", data=load, ax=ax[j], style="type", legend=False)
            ax[j].set_title("$\lambda =" + str(lamb[j]) + "$")
            ax[j].set_xlabel('Time (ms)')
            ax[j].set_ylabel('CDF')
            ax[j].grid(True)
            ax[j].set_ylim(0, 1.1)
            ax[j].legend(nf, loc=0, ncol=1, fontsize=ft, prop={'size': 7})
        plt.tight_layout()
        plt.savefig(r'../figs/fig3' + figname[u] + '.svg', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    plot_loading_time()
    plot_resources_usage()
    plot_processing_time()
