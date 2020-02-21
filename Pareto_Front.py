from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns;
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter

# theme
plt.rc('legend', fontsize='large')
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
sns.set_context('paper')
sns.set_style('ticks')
colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a',
          '#ffff99', '#b15928', 'k']
sns.set_palette(palette=colors)
alpha = 0.5
f = ["Conditioner", "Redirector", "Dropper", "Scheduler", "Shaper", "Scale Up", "Scale In"]
xlab = ['Server', 'GW C', 'GW C1', 'GW C11']
lim = 200
# delta = ["l", "u", "t"]
figname = ['a', 'b', 'c', 'd']
fignumb = ['10', '11', '12', '13']
patterns = cycle(["oo", "xx", "||||", "++", "//"])
psolutions = []
problems = []
xlabel = ['E2E Resource Usage', 'E2E Actions Costs']
ft = 10
applabel = ['Teleoperated driving', 'Cooperative maneuvers', 'Traffic efficiency']
slolabel = ['Latency (ms)', r'Availability ($\%$)', 'Throughput (req/sec)']
num_locations = 3
plt.rcParams['hatch.linewidth'] = 0.1
alg_titile = ['FCFS', 'AS', 'QoSEF', 'QoSEFe', 'QoS4NIP']

app_nb = 3
host_nb = 4
host_size = 7
n_obj = app_nb * 3 + 2
n_vars = host_size * host_nb
cond_row = 0
sha_row = 2
sch_row = 3
drop_row = 4
xtra_line = app_nb + 2


class visualisation():

    def __init__(self, nds, problems):
        self.psolutions = nds
        self.problems = problems

    def displayQoS(self):
        psolutions_list = self.psolutions
        problems = self.problems
        df1 = pd.DataFrame(columns=['Application', slolabel[0], 'Scheme'])
        df2 = pd.DataFrame(columns=['Application', slolabel[1], 'Scheme'])
        df3 = pd.DataFrame(columns=['Application', slolabel[2], 'Scheme'])

        for rep in range(1):
            psolutions = psolutions_list  # [rep]
            if min([psolutions[0].__len__(), psolutions[1].__len__(), psolutions[2].__len__(),
                    psolutions[3].__len__()]) >= lim:
                for l in range(lim):
                    data_0 = np.concatenate((np.array(psolutions[0][l].variables, dtype=np.int).reshape(
                        (host_nb, 1, host_size)), problems[0].effect_on_apps, problems[0].coef, problems[0].xtra),
                        axis=1)
                    data_1 = np.concatenate((np.array(psolutions[1][l].variables, dtype=np.int).reshape(
                        (host_nb, 1, host_size)), problems[1].effect_on_apps, problems[1].coef, problems[1].xtra),
                        axis=1)
                    data_2 = np.concatenate((np.array(psolutions[2][l].variables, dtype=np.int).reshape(
                        (host_nb, 1, host_size)), problems[2].effect_on_apps, problems[2].coef, problems[2].xtra),
                        axis=1)
                    data_3 = np.concatenate((np.array(psolutions[3][l].variables, dtype=np.int).reshape(
                        (host_nb, 1, host_size)), problems[3].effect_on_apps, problems[3].coef, problems[3].xtra),
                        axis=1)
                    algz = pd.DataFrame({'Application': applabel, slolabel[0]: np.sum(problems[0].lat, axis=1)})
                    alg0 = pd.DataFrame({'Application': applabel, slolabel[0]: calc_lats(problems[0], data_0)})
                    alg1 = pd.DataFrame({'Application': applabel, slolabel[0]: calc_lats(problems[1], data_1)})
                    alg2 = pd.DataFrame({'Application': applabel, slolabel[0]: calc_lats(problems[2], data_2)})
                    alg3 = pd.DataFrame({'Application': applabel, slolabel[0]: calc_lats(problems[3], data_3)})
                    algz['Scheme'] = algz.apply(lambda x: alg_titile[0], axis=1)
                    alg0['Scheme'] = alg0.apply(lambda x: alg_titile[1], axis=1)
                    alg1['Scheme'] = alg1.apply(lambda x: alg_titile[2], axis=1)
                    alg2['Scheme'] = alg2.apply(lambda x: alg_titile[3], axis=1)
                    alg3['Scheme'] = alg3.apply(lambda x: alg_titile[4], axis=1)
                    df1 = pd.concat([df1, algz.append([alg0, alg1, alg2, alg3], ignore_index=True)], ignore_index=True)
                    algz = pd.DataFrame({'Application': applabel, slolabel[1]: [100, 100, 100]})
                    alg0 = pd.DataFrame({'Application': applabel, slolabel[1]: 100 - calc_errs(problems[0], data_0)})
                    alg1 = pd.DataFrame({'Application': applabel, slolabel[1]: 100 - calc_errs(problems[1], data_1)})
                    alg2 = pd.DataFrame({'Application': applabel, slolabel[1]: 100 - calc_errs(problems[2], data_2)})
                    alg3 = pd.DataFrame({'Application': applabel, slolabel[1]: 100 - calc_errs(problems[3], data_3)})
                    algz['Scheme'] = algz.apply(lambda x: alg_titile[0], axis=1)
                    alg0['Scheme'] = alg0.apply(lambda x: alg_titile[1], axis=1)
                    alg1['Scheme'] = alg1.apply(lambda x: alg_titile[2], axis=1)
                    alg2['Scheme'] = alg2.apply(lambda x: alg_titile[3], axis=1)
                    alg3['Scheme'] = alg3.apply(lambda x: alg_titile[4], axis=1)
                    df2 = pd.concat([df2, algz.append([alg0, alg1, alg2, alg3], ignore_index=True)], ignore_index=True)
                    algz = pd.DataFrame({'Application': applabel, slolabel[2]: np.amin(problems[0].thr, axis=1)})
                    alg0 = pd.DataFrame({'Application': applabel, slolabel[2]: calc_thrs(problems[0], data_0)})
                    alg1 = pd.DataFrame({'Application': applabel, slolabel[2]: calc_thrs(problems[1], data_1)})
                    alg2 = pd.DataFrame({'Application': applabel, slolabel[2]: calc_thrs(problems[2], data_2)})
                    alg3 = pd.DataFrame({'Application': applabel, slolabel[2]: calc_thrs(problems[3], data_3)})
                    algz['Scheme'] = algz.apply(lambda x: alg_titile[0], axis=1)
                    alg0['Scheme'] = alg0.apply(lambda x: alg_titile[1], axis=1)
                    alg1['Scheme'] = alg1.apply(lambda x: alg_titile[2], axis=1)
                    alg2['Scheme'] = alg2.apply(lambda x: alg_titile[3], axis=1)
                    alg3['Scheme'] = alg3.apply(lambda x: alg_titile[4], axis=1)
                    df3 = pd.concat([df3, algz.append([alg0, alg1, alg2, alg3], ignore_index=True)], ignore_index=True)

            #print(df1)
            #print(df2)
            #print(df3)

            f, ax = plt.subplots(figsize=(4.4, 3.1))
            sns.catplot(x='Application', y=slolabel[0], hue='Scheme', data=df1, ax=ax, kind="bar", ci="sd",
                        errwidth=0.5,
                        errcolor='k', palette=[colors[1], colors[3], colors[5], colors[7], colors[9]], capsize=.1)
            for i, patch in enumerate(ax.patches):
                if i % num_locations == 0:
                    hatch = next(patterns)
                patch.set_hatch(hatch)
                patch.set_edgecolor('w')
            firts_legend = ax.legend(title='Scheme', loc='best')
            axhline = ax.axhline(y=20, ls='--', c='red', xmin=0.03, xmax=0.31, label='Required')
            ax.axhline(y=100, ls='--', c='red', xmin=0.35, xmax=0.65)
            ax.axhline(y=1000, ls='--', c='red', xmin=0.68, xmax=1)
            second_legend = ax.legend(handles=[axhline], loc=9)
            ax.add_artist(second_legend)
            ax.add_artist(firts_legend)
            ax.grid(True)
            f.show()
            f.savefig(r'figs/fig9a.svg', bbox_inches='tight')

            f, ax = plt.subplots(figsize=(4.4, 3.1))
            sns.catplot(x='Application', y=slolabel[1], hue='Scheme', data=df2, ax=ax, kind="bar", ci="sd",
                        errwidth=0.5,
                        errcolor='k', palette=[colors[1], colors[3], colors[5], colors[7], colors[9]], capsize=.1)
            for i, patch in enumerate(ax.patches):
                if i % num_locations == 0:
                    hatch = next(patterns)
                patch.set_hatch(hatch)
                patch.set_edgecolor('w')
            firts_legend = ax.legend(title='Scheme', loc='best')
            axhline = ax.axhline(y=100 - 1, ls='--', c='red', xmin=0.03, xmax=0.31, label='Required')
            ax.axhline(y=100 - 1, ls='--', c='red', xmin=0.35, xmax=0.65)
            ax.axhline(y=100 - 10, ls='--', c='red', xmin=0.68, xmax=1)
            second_legend = ax.legend(handles=[axhline], loc=9)
            ax.add_artist(second_legend)
            ax.add_artist(firts_legend)
            ax.set_ylim(85, None)
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.d'))
            ax.grid(True)
            f.show()
            f.savefig(r'figs/fig9b.svg', bbox_inches='tight')

            f, ax = plt.subplots(figsize=(4.4, 3.1))
            sns.catplot(x='Application', y=slolabel[2], hue='Scheme', data=df3, ax=ax, kind="bar", ci="sd",
                        errwidth=0.5,
                        errcolor='k', palette=[colors[1], colors[3], colors[5], colors[7], colors[9]], capsize=.1)
            for i, patch in enumerate(ax.patches):
                if i % num_locations == 0:
                    hatch = next(patterns)
                patch.set_hatch(hatch)
                patch.set_edgecolor('w')
            firts_legend = ax.legend(title='Scheme', loc='best')
            axhline = ax.axhline(y=25, ls='--', c='red', xmin=0.03, xmax=0.31, label='Required')
            ax.axhline(y=10, ls='--', c='red', xmin=0.35, xmax=0.65)
            ax.axhline(y=10, ls='--', c='red', xmin=0.68, xmax=1)
            second_legend = ax.legend(handles=[axhline], loc=9)
            ax.add_artist(second_legend)
            ax.add_artist(firts_legend)
            ax.grid(True)
            f.show()
            f.savefig(r'figs/fig9c.svg', bbox_inches='tight')

    def displayCosts(self):
        psolutions = self.psolutions
        select = []
        xy_arr = []
        xy_arr.append((0, 0))
        if min([psolutions[0].__len__(), psolutions[1].__len__(), psolutions[2].__len__(),
                psolutions[3].__len__()]) >= lim:
            ####################################################################
            algz = pd.DataFrame({xlabel[0]: [0], xlabel[1]: [0]})
            alg0 = pd.DataFrame({xlabel[0]: [s.objectives[n_obj - 1] for s in psolutions[0][0:lim]],
                                 xlabel[1]: [s.objectives[n_obj - 2] for s in psolutions[0][0:lim]]})
            alg1 = pd.DataFrame({xlabel[0]: [s.objectives[n_obj - 1] for s in psolutions[1][0:lim]],
                                 xlabel[1]: [s.objectives[n_obj - 2] for s in psolutions[1][0:lim]]})
            alg2 = pd.DataFrame({xlabel[0]: [s.objectives[n_obj - 1] for s in psolutions[2][0:lim]],
                                 xlabel[1]: [s.objectives[n_obj - 2] for s in psolutions[2][0:lim]]})
            alg3 = pd.DataFrame({xlabel[0]: [s.objectives[n_obj - 1] for s in psolutions[3][0:lim]],
                                 xlabel[1]: [s.objectives[n_obj - 2] for s in psolutions[3][0:lim]]})
            algz['Scheme'] = algz.apply(lambda x: alg_titile[0], axis=1)
            alg0['Scheme'] = alg0.apply(lambda x: alg_titile[1], axis=1)
            alg1['Scheme'] = alg1.apply(lambda x: alg_titile[2], axis=1)
            alg2['Scheme'] = alg2.apply(lambda x: alg_titile[3], axis=1)
            alg3['Scheme'] = alg3.apply(lambda x: alg_titile[4], axis=1)

            df = algz.append([alg0, alg1, alg2, alg3], ignore_index=True)
            f, ax = plt.subplots(figsize=(3, 3))
            sns.scatterplot(x=xlabel[0], y=xlabel[1], hue='Scheme', style='Scheme', alpha=0.5,
                            s=40 * df[str(xlabel[1])] + 10, data=df, ax=ax, edgecolor=None,
                            palette=[colors[1], colors[3], colors[5], colors[7], colors[9]])
            for a in range(4):
                x = np.array([s.objectives[n_obj - 2] for s in psolutions[a][0:lim]])
                y = np.array([s.objectives[n_obj - 1] for s in psolutions[a][0:lim]])
                if a == 0:
                    sel = np.argmin(x)
                elif a == 3:
                    x_winner = np.where(x == x.min())
                    sel = np.argwhere(y == np.min(np.take(y, x_winner)))
                    sel = sel.flatten()[0]
                else:
                    sel = np.argmin(y)
                v, u = (x[sel], y[sel])
                select.append(sel)
                xy_arr.append((u, v))
                ax.plot(u, v, 'o', ms=5 * 2, mec='k', mfc='none', mew=1)
            ax.plot(0, 0, 'o', ms=5 * 2, mec='k', mfc='none', mew=1)
            for a in range(5):
                if a == 4:
                    ax.annotate('Selection', xy=xy_arr[a], xycoords='data', color='k', xytext=(0.7, 0.25),
                                textcoords='axes fraction',
                                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.2", ls='--', fc="w",
                                                color='k'))
                else:
                    ax.annotate('Selection', xy=xy_arr[a], xycoords='data', color='w', xytext=(0.7, 0.25),
                                textcoords='axes fraction',
                                arrowprops=dict(arrowstyle="-|>", connectionstyle="arc3,rad=0.2", ls='--', fc="w",
                                                color='k'))

            ax.grid(True)
            f.savefig(r'figs/fig8a.svg', bbox_inches='tight')
            f.show()

            # ******************************************#
            cmap = LinearSegmentedColormap.from_list('Custom', ('#deebf7', '#9ecae1'), 2)
            objs = []
            for i in range(0, host_nb):
                for j in range(0, host_size):
                    objs.insert(i * host_size + j, '$x^{' + str(j + 1) + '}_{' + str(i + 1) + '}$')

            f, ax = plt.subplots(figsize=(6, 1))
            data_z = np.zeros((1, n_vars), dtype=int)
            data_0 = np.array([np.reshape(np.array(psolutions[0][select[0]].variables).astype(int), n_vars)])
            data_1 = np.array([np.reshape(np.array(psolutions[1][select[1]].variables).astype(int), n_vars)])
            data_2 = np.array([np.reshape(np.array(psolutions[2][select[2]].variables).astype(int), n_vars)])
            data_3 = np.array([np.reshape(np.array(psolutions[3][select[3]].variables).astype(int), n_vars)])
            data = np.concatenate((data_z, data_0, data_1, data_2, data_3), axis=0)
            sns.heatmap(data, ax=ax, cmap=cmap, linewidths=True, cbar=False, annot=True,
                        fmt="d", square=True)
            ax.set_xlabel('Decision Variables')
            ax.tick_params(left=False, bottom=True)
            ax.set_xticklabels(objs, rotation=0)
            ax.set_ylim(len(alg_titile), 0)
            ax.set_xlim(0, len(objs))
            ax.set_yticklabels(alg_titile, va='center', rotation=0)
            f.savefig(r'figs/fig8b.svg', bbox_inches='tight')
            f.show()


def calc_lats(p, x):
    lat_line = app_nb + 1
    e2e_lats = np.zeros(app_nb)
    for i in range(0, app_nb):
        e2e_lat = 0
        for h in range(0, host_nb):
            host_ben_lat = 0
            # function ben
            for j in range(1, host_size - 2):
                host_ben_lat += x[h][i][j] * x[h][lat_line][j] * x[h][0][j] * 1 / 100
            for j in range(host_size - 2, host_size):
                host_ben_lat += x[h][i][j] * x[h][lat_line][j] * x[h][0][j] * 1 / 100

            if host_ben_lat >= 1:  # >=
                host_ben_lat = 0
            else:
                host_ben_lat = (1 - host_ben_lat) * p.lat[i][h]
                if x[h][0][sha_row] == 1:
                    host_ben_lat += x[h][xtra_line][sha_row]
            e2e_lat += host_ben_lat
        e2e_lats[i] = e2e_lat
    return e2e_lats


def calc_errs(p, x):
    e2e_errs = np.zeros(app_nb)
    for i in range(0, app_nb):
        app_effect_line = i + 1
        benefice = np.zeros(host_nb)
        for h in range(0, host_nb):
            benefice_past = 0
            for j in range(0, h):
                benefice_past += benefice[j]
            if x[h][app_effect_line][drop_row] < 0:
                benefice[h] = - (100 - benefice_past) * x[h][xtra_line][drop_row] * x[h][app_effect_line][drop_row] * \
                              x[h][0][drop_row] * 1 / 100
        e2e_err = np.sum(benefice)
        e2e_errs[i] = e2e_err
    return e2e_errs


def calc_thrs(p, x):
    e2e_thrs = np.zeros(app_nb)
    for i in range(0, app_nb):
        benefice = np.zeros(host_nb)
        for h in range(0, host_nb):
            if x[h][0][sch_row] == 1:
                benefice[h] = (100 + x[h][xtra_line][sch_row]) * p.thr[i][h] * 1 / 100
            else:
                benefice[h] = p.thr[i][h]
            for j in range(host_size - 2, host_size):
                if x[h][0][j]:
                    benefice[h] *= 2
        e2e_thr = np.min(benefice)
        e2e_thrs[i] = e2e_thr
    # print(e2e_thrs)
    return e2e_thrs
