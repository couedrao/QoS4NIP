import numpy as np
from platypus import *
# input Values
app_nb = 3
host_nb = 4
host_size = 7
f = ["Con", "Red", "Dro", "Sch", "Del"]
n_obj = app_nb * 3 + 2

#np.random.seed(42)
# random.seed(a=42)
penalty = 1000

# initial rand
h_r_usage = np.random.randint(15, 25, size=(2, host_nb))
vnf_r_usage = np.random.randint(penalty, penalty + 1, size=(2, 5))
anf_r_usage = np.random.randint(5, 10, size=(2, 5))
slo_lat = np.array([[20], [100], [1000]])  # np.random.randint(100, 1000, size=(app_nb, 1))
slo_err = np.array([[1], [1], [10]])  # np.random.randint(10, size=(app_nb, 1))
slo_thr = np.array([[25], [10], [10]])  # np.random.randint(1, 10, size=(app_nb, 1))

slo = np.concatenate((slo_lat, slo_err, slo_thr), axis=1)
lat1 = 10 * np.ones((1, host_nb), dtype=int)
lat2 = 30 * np.ones((1, host_nb), dtype=int)
lat3 = 80 * np.ones((1, host_nb), dtype=int)
thr1 = 20 * np.ones((1, host_nb), dtype=int)
thr2 = 10 * np.ones((1, host_nb), dtype=int)
thr3 = 10 * np.ones((1, host_nb), dtype=int)

lat = np.concatenate((lat1, lat2, lat3), axis=1)
lat = lat.reshape((app_nb, host_nb))
thr = np.concatenate((thr1, thr2, thr3), axis=1)
thr = thr.reshape((app_nb, host_nb))
# lat = np.random.randint(5, 6, size=(app_nb, host_nb))
# thr = np.random.randint(20, 30, size=(app_nb, host_nb))
# Supported Action
ai_cloud_fog = np.full((host_nb - 1, 2), True, dtype=bool)
ai_edge = np.full((1, 2), True, dtype=bool)
ai = np.concatenate((ai_cloud_fog, ai_edge), axis=0)
# Supported vnf and anf

################### chromosome
cond_eff = np.ones((host_nb, app_nb, 1), dtype=int)
app1 = np.ones((host_nb, 1, 1), dtype=int)
appx = -1 * np.ones((host_nb, app_nb - 1, 1), dtype=int)
red_eff = np.concatenate((app1, appx), axis=1)
sch_eff = np.concatenate((app1, appx), axis=1)
sha_eff = np.concatenate((app1, appx), axis=1)
drop_eff = np.concatenate((app1, appx), axis=1)
actu_eff = np.ones((host_nb, app_nb, 1), dtype=int)
acti_eff = np.ones((host_nb, app_nb, 1), dtype=int)
effect_on_apps = np.concatenate((cond_eff, red_eff, sha_eff, sch_eff, drop_eff, actu_eff, acti_eff), axis=2)
effect_on_apps = effect_on_apps.reshape((host_nb, app_nb, host_size))
# Coefficient
# coef = np.random.randint(20, 35, size=(host_nb, 1, host_size))
cond_coef = np.ones((host_nb, 1, 1), dtype=int)
red_coef = np.ones((host_nb, 1, 1), dtype=int)
drop_coef = 41 * np.ones((host_nb, 1, 1), dtype=int)
sch_coef = 35 * np.ones((host_nb, 1, 1), dtype=int)
sha_coef = 35 * np.ones((host_nb, 1, 1), dtype=int)
actu_coef = 50 * np.ones((host_nb, 1, 1), dtype=int)
acti_coef = 50 * np.ones((host_nb, 1, 1), dtype=int)
coef = np.concatenate((cond_coef, red_coef, sha_coef, sch_coef, drop_coef, actu_coef, acti_coef), axis=2)
coef = coef.reshape((host_nb, 1, host_size))
# EXTRA
# xtra = np.random.randint(10, size=(host_nb, 1, host_size))
cond_xtra = np.zeros((host_nb, 1, 1), dtype=int)
red_xtra = np.ones((host_nb, 1, 1), dtype=int)
drop_xtra = np.ones((host_nb, 1, 1), dtype=int)
sch_xtra = 30 * np.ones((host_nb, 1, 1), dtype=int)
sha_xtra = np.random.randint(10, size=(host_nb, 1, 1))
actu_xtra = np.random.randint(1, 3, size=(host_nb, 1, 1))
acti_xtra = np.random.randint(1, 3, size=(host_nb, 1, 1))
xtra = np.concatenate((cond_xtra, red_xtra, sha_xtra, sch_xtra, drop_xtra, actu_xtra, acti_xtra), axis=2)
xtra = xtra.reshape((host_nb, 1, host_size))
# x0 = np.random.randint(2, size=(host_nb, 1, host_size))
cond_row = 0
red_row = 1
sha_row = 2
sch_row = 3
drop_row = 4
xtra_line = app_nb + 2


class AS_Scheduler(Problem):

    def __init__(self):
        super(AS_Scheduler, self).__init__(1, n_obj, n_obj)

        self.effect_on_apps = effect_on_apps
        self.coef = coef
        self.xtra = xtra
        self.penalty = penalty
        self.lat = lat
        self.slo = slo
        self.thr = thr
        self.actu_xtra = actu_xtra
        self.acti_xtra = acti_xtra
        self.types[:] = Binary(host_nb * host_size)
        for i in range(n_obj):
            self.constraints[i] = Constraint('<=', 1)
        # self.constraints[n_obj - 1] = Constraint('<', penalty)
        # self.constraints[n_obj - 2] = Constraint('<', penalty)

    def evaluate(self, solution):
        # print('Act eval running ...')
        data = np.array(solution.variables[:], dtype=np.int).reshape((host_nb, 1, host_size))
        x = np.concatenate((data, effect_on_apps, coef, xtra), axis=1)
        s = np.concatenate((calc_lats(x), calc_errs(x), calc_thrs(x), calc_cact(x), calc_cfct(x)), axis=None)
        solution.objectives[:] = s
        solution.constraints[:] = s


def calc_lats(x):
    e2e_lats = np.zeros(app_nb)
    lat_line = app_nb + 1

    for h in range(0, host_nb - 2):
        if x[h][0][red_row] == 1 and np.sum(x[h + 1][0][0:host_size]) > 0:
            return penalty * np.ones(app_nb)

    if x[host_nb - 1][0][red_row] == 1 or x[host_nb - 2][0][red_row] == 1:
        return penalty * np.ones(app_nb)

    for i in range(0, app_nb):
        e2e_lat = 0
        for h in range(0, host_nb):
            host_ben_lat = 0

            # function ben
            for j in range(1, host_size - 2):
                if x[h][0][j] == 1 and x[h][0][cond_row] == 0:
                    return penalty * np.ones(app_nb)
                if x[h][0][cond_row] == 1 and np.sum(x[h][0][1:host_size - 2]) == 0:
                    return penalty * np.ones(app_nb)
                host_ben_lat += x[h][i][j] * x[h][lat_line][j] * x[h][0][j] * 1 / 100

            for j in range(host_size - 2, host_size):
                host_ben_lat += x[h][i][j] * x[h][lat_line][j] * x[h][0][j] * 1 / 100

            if host_ben_lat >= 1:  # >=
                host_ben_lat = 0
            else:
                host_ben_lat = (1 - host_ben_lat) * lat[i][h]
                if x[h][0][sha_row] == 1:
                    host_ben_lat += x[h][xtra_line][sha_row]
            e2e_lat += host_ben_lat
        e2e_lats[i] = e2e_lat / slo[i][0]
    return e2e_lats


def calc_errs(x):
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
        if e2e_err > slo[i][1]:
            return penalty * np.ones(app_nb)
        e2e_errs[i] = e2e_err / slo[i][1]
    return e2e_errs


def calc_thrs(x):
    e2e_thrs = np.zeros(app_nb)
    for i in range(0, app_nb):
        benefice = np.zeros(host_nb)
        for h in range(0, host_nb):
            if x[h][0][sch_row] == 1:
                # if x[h][i][0]:
                benefice[h] = (100 + x[h][xtra_line][sch_row]) * thr[i][h] * 1 / 100
            # else:
            #   benefice[h] = penalty
            else:
                benefice[h] = thr[i][h]

            for j in range(host_size - 2, host_size):
                if x[h][0][j]:
                    benefice[h] *= 2

        e2e_thr = np.min(benefice)
        e2e_thrs[i] = slo[i][2] / e2e_thr
    # print(e2e_thrs)
    return e2e_thrs


def calc_cfct(x):
    e2e_cfct = 0
    max_f_cost = 200
    for h in range(0, host_nb):
        cost_h_cpu = 0
        cost_h_ram = 0
        for j in range(0, host_size - 2):
            cost_h_cpu += vnf_r_usage[0][j] * x[h][0][j]
            cost_h_ram += vnf_r_usage[1][j] * x[h][0][j]

        action_ben_cpu = h_r_usage[0][h]
        action_ben_ram = h_r_usage[1][h]
        action_ben = 0
        for j in range(host_size - 2, host_size):
            if ai[h][j - host_size + 2]:
                action_ben += x[h][0][j]
        if action_ben > 0:
            action_ben *= 2
            action_ben_cpu /= action_ben
            action_ben_ram /= action_ben
            cost_h_cpu /= action_ben
            cost_h_ram /= action_ben
        a = action_ben_cpu + cost_h_cpu
        b = action_ben_ram + cost_h_ram
        if a < 100 and b < 100:
            host_cfct = (cost_h_cpu + cost_h_ram) / max_f_cost
            e2e_cfct += host_cfct / 2
        else:
            e2e_cfct += penalty
    return e2e_cfct


def calc_cact(x):
    e2e_cact = 0
    max_a_cost = np.sum(actu_xtra + acti_xtra)
    for h in range(0, host_nb):
        host_cost = 0
        for j in range(host_size - 2, host_size):
            if x[h][0][j] == 1:
                if ai[h][j - host_size + 2]:
                    host_cost += x[h][xtra_line][j] / max_a_cost
                else:
                    host_cost = penalty

        e2e_cact += host_cost
    return e2e_cact
