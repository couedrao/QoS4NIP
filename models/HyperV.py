from platypus import *
from math import pi
import seaborn as sns

import matplotlib.pyplot as plt
from models import QoS4NIP
import numpy as np

# theme
plt.rc('legend', fontsize='large')
colors = ['#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a', '#ffff99']
sns.set(context="paper", style="ticks")
sns.set_palette(palette=colors)


def display_hypervolume(results=None, ndigits=None, seeds=None):
    categories = []
    for i in range(seeds):
        categories.append(str(i + 1))
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    f, ax = plt.subplots(figsize=(4, 2), subplot_kw=dict(projection='polar'))
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories)
    ax.set_rlabel_position(0)
    plt.yticks(color="grey", size=6)
    for algorithm in six.iterkeys(results):
        print(algorithm)
        for problem in six.iterkeys(results[algorithm]):
            if isinstance(results[algorithm][problem], dict):
                print("   ", problem)
                for indicator in six.iterkeys(results[algorithm][problem]):
                    values = list(map(functools.partial(round, ndigits=ndigits),
                                      results[algorithm][problem][indicator]))
                    print("      ", values)
                    values[:] = [x * 1000 for x in values]
                    values += values[:1]
                    ax.plot(angles, values, linewidth=1, linestyle='solid', label=algorithm)
                    ax.fill(angles, values, alpha=0.1)
    plt.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=2.5, fontsize=12)
    f.show()
    f.savefig(r'../figs/fig6a.svg', bbox_inches='tight')


app_nb = 3
n_obj = app_nb * 3 + 2
seeds = 30
pop_size = 100
if __name__ == '__main__':
    problems = [QoS4NIP.QoS4NIP_Scheduler()]
    algorithms = [(SPEA2, {"population_size": pop_size}),
                  (NSGAII, {"population_size": pop_size}),
                  (NSGAIII, {"population_size": pop_size, "divisions_outer": 8})]
    with ProcessPoolEvaluator() as evaluator:
        hyp = Hypervolume(minimum=np.zeros(n_obj), maximum=np.ones(n_obj))
        results = experiment(algorithms, problems, seeds=seeds, nfe=10000, evaluator=evaluator, display_stats=True)
        hyp_result = calculate(results, hyp, evaluator=evaluator)
        display_hypervolume(hyp_result, ndigits=10, seeds=seeds)
