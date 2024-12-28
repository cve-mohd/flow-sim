import numpy as np
from river import River
from scipy.constants import g


class LaxModel:
    def __init__(self,
                 river: River,
                 delta_t: int | float,
                 delta_x: int | float,
                 duration: int | float):

        self.river = river
        self.delta_t, self.delta_x = delta_t, delta_x
        self.celerity = self.delta_x / float(self.delta_t)

        self.n_nodes = int(self.river.length / self.delta_x + 1)
        self.duration = duration

        self.A_previous = []
        self.Q_previous = []

        self.A_current = []
        self.Q_current = []

        self.resultsA = []
        self.resultsQ = []

        self.initialize_t0()

    def initialize_t0(self):
        self.river.initialize_conditions(self.n_nodes)

        for x, (A, Q) in enumerate(self.river.initial_conditions):
            self.A_previous.append(A)
            self.Q_previous.append(Q)

            self.A_current.append(0)
            self.Q_current.append(0)

        self.resultsA.append(self.A_previous)
        self.resultsQ.append(self.Q_previous)

    def lax_A(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):

        A = 0.5 * (A_i_plus_1 + A_i_minus_1) - (0.5 / self.celerity) * (Q_i_plus_1 - Q_i_minus_1)

        return A

    def lax_Q(self, A_i_minus_1, A_i_plus_1, Q_i_minus_1, Q_i_plus_1):
        W = self.river.width
        n = self.river.manning_co
        S_0 = self.river.bed_slope

        Q = (-g / (4 * W * self.celerity) * (A_i_plus_1 ** 2 - A_i_minus_1 ** 2)
             + 0.5 * g * S_0 * self.delta_t * (A_i_plus_1 + A_i_minus_1)
             + 0.5 * (Q_i_plus_1 + Q_i_minus_1)
             - (0.5 / self.celerity) * (Q_i_plus_1 ** 2 / A_i_plus_1 - Q_i_minus_1 ** 2 / A_i_minus_1)
             - 0.5 * g * W ** (4 / 3.) * n ** 2 * self.delta_t * (
                         Q_i_plus_1 ** 2 / A_i_plus_1 ** (7 / 3.) + Q_i_minus_1 ** 2 / A_i_minus_1 ** (7 / 3.)))

        return Q

    def manning_Q(self, A, S_f):
        W = self.river.width
        n = self.river.manning_co

        Q = ((1 / n) * A ** (5. / 3) * S_f ** 0.5
             / (W + 2 * A / W) ** (2. / 3))

        return Q

    def solve(self):
        for time in range(self.delta_t, self.duration + 1, self.delta_t):
            self.Q_current[0] = self.river.inflow_Q(time / 3600.)
            self.A_current[0] = self.lax_A(self.A_previous[0] * 2 - self.A_previous[1],
                                           self.A_previous[1],
                                           self.Q_previous[0] * 2 - self.Q_previous[1],
                                           self.Q_previous[1])

            for i in range(1, self.n_nodes - 1):
                self.A_current[i] = self.lax_A(self.A_previous[i - 1],
                                               self.A_previous[i + 1],
                                               self.Q_previous[i - 1],
                                               self.Q_previous[i + 1])

                self.Q_current[i] = self.lax_Q(self.A_previous[i - 1],
                                               self.A_previous[i + 1],
                                               self.Q_previous[i - 1],
                                               self.Q_previous[i + 1])

            self.A_current[-1] = self.lax_A(self.A_previous[-2],
                                            self.A_previous[-1] * 2 - self.A_previous[-2],
                                            self.Q_previous[-2],
                                            self.Q_previous[-1] * 2 - self.Q_previous[-2])

            self.Q_current[-1] = self.lax_Q(self.A_previous[-2],
                                            self.A_previous[-1] * 2 - self.A_previous[-2],
                                            self.Q_previous[-2],
                                            self.Q_previous[-1] * 2 - self.Q_previous[-2])

            Sf_ds = self.river.friction_slope(self.A_current[-1],
                                              self.Q_current[-1])
            self.Q_current[-1] = self.manning_Q(self.A_current[-1], Sf_ds)

            self.A_previous = [a for a in self.A_current]
            self.Q_previous = [q for q in self.Q_current]

            self.resultsA.append(self.A_previous)
            self.resultsQ.append(self.Q_previous)

    def save_results(self, time_steps_to_save):
        A = self.resultsA[::len(self.resultsA) // (time_steps_to_save - 1)]
        Q = self.resultsQ[::len(self.resultsQ) // (time_steps_to_save - 1)]

        A, Q = str(A), str(Q)

        A = A.replace('], [', '\n')
        Q = Q.replace('], [', '\n')
        for c in "[]' ":
            A = A.replace(c, '')
            Q = Q.replace(c, '')

        with open('area.csv', 'w') as output_file:
            output_file.write(A)

        with open('discharge.csv', 'w') as output_file:
            output_file.write(Q)


def quadratic_extrapolation(indices, values, target_index):
    # Ax = B
    A = [np.array(indices) ** i for i in range(len(indices) - 1, 0, -1)]

    A.append(np.ones(len(indices)))
    A = np.vstack(A).T
    B = np.array(values)

    coeffs = np.linalg.solve(A, B)

    target_value = 0
    for i in range(len(indices)):
        target_value += coeffs[i] * target_index ** (len(indices) - i - 1)

    return target_value
