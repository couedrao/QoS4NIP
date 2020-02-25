from platypus import *

import Pareto_Front
from models import QoSEF, QoS4NIP, AS, QoSEFe
import numpy as np

app_nb = 3
n_obj = app_nb * 3 + 2

if __name__ == '__main__':
    problem1 = AS.AS_Scheduler()
    problem2 = QoSEF.QoSEF_Scheduler()
    problem3 = QoSEFe.QoSEFe_Scheduler()
    problem4 = QoS4NIP.QoS4NIP_Scheduler()
    problems = [problem1, problem2, problem3, problem4]
    pb = ['AS', 'QoSEF', 'QoSEFe', 'QoS4NIP']
    all_fronts = []
    algorithms = [(NSGAII, {"population_size": 200})]

    # run the experiment using Python 3's concurrent futures for parallel evaluation
    with ProcessPoolEvaluator() as evaluator:
        results = experiment(algorithms, problems, seeds=1, nfe=10000, evaluator=evaluator, display_stats=True)

    # rearrange the results for visualisation
    for algorithm in six.iterkeys(results):
        print('**********' + str(algorithm) + '**********')
        for problem in six.iterkeys(results[algorithm]):
            solutions = results[algorithm][problem][0]
            front = nondominated([s for s in solutions if s.feasible])
            print(str(problem) + ' : ' + str(front.__len__()))
            if front.__len__() < 200:
                break
            all_fronts.append(front)

    # visualize the results
    visualize = Pareto_Front.visualisation(all_fronts, problems)
    visualize.displayQoS()
    visualize.displayCosts()
