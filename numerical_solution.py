import math
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(2000)


class Solver:
    def __init__(self, beta, omg, x0, x_left, x_right, y0, y_left, y_right, y_prime, step):
        self.beta = beta
        self.omg = omg
        self.x0 = x0
        self.x_left = x_left
        self.x_right = x_right
        self.y0 = y0
        self.y_left = y_left
        self.y_right = y_right
        self.y_prime = y_prime
        self.step = step
        # for delta step:
        self.y_minus_coef = -self.beta / (2*self.step) + 1/(self.step ** 2)
        self.y_plus_coef = self.beta / (2*self.step) + 1/(self.step ** 2)
        self.y_coef = -2/(self.step ** 2) + (self.omg ** 2)

    # # TODO: for what?
    # def func(self, x, y, y_prime):
    #     return -self.beta*y_prime-(self.omg ** 2)*y

    def right_step(self, y_minus, y):
        return (-y_minus*self.y_minus_coef - y*self.y_coef) / self.y_plus_coef

    def left_step(self, y, y_plus):
        return (-y_plus*self.y_plus_coef - y*self.y_coef) / self.y_minus_coef

    def right_shot(self, y_minus, y, x, values=[]):
        if x > self.x_right:
            return y, values
        values.append(y)
        return self.right_shot(y, self.right_step(y_minus, y), x+self.step, values)

    def left_shot(self, y, y_plus, x, values=[]):
        if x < self.x_left:
            return y, values
        values.append(y)
        return self.left_shot(self.left_step(y, y_plus), y, x-self.step, values)

    def Newton_step(self, y0, y_prime, clear_v=True):
        y_right_res, _ = self.right_shot(y0, y0+y_prime*step, x0, values=[])
        y_left_res, _ = self.left_shot(y0, y0+y_prime*step, x0, values=[])
        if clear_v:
            values = []
            _ = []
        right_discrep = self.y_right - y_right_res
        left_discrep = self.y_left - y_right_res
        return euclid(right_discrep, left_discrep)

    def Newton_GD(self, y0, y_prime, goal_old, tau, eps, tau_decr=0.99, clear_v=True):
        y_step = goal_old - self.Newton_step(y0+tau, y_prime)
        y_prime_step = goal_old - self.Newton_step(y0, y_prime+tau)
        mes = euclid(y_step, y_prime_step) + 0.0001
        y_new = y0 + tau * y_step / mes
        y_prime_new = y_prime + tau * y_prime_step / mes
        goal = self.Newton_step(y_new, y_prime_new)
        if goal < eps:
            return y_new, y_prime_new
        return self.Newton_GD(y_new, y_prime_new, goal, tau*tau_decr, eps)


def euclid(d1, d2):
    return math.sqrt(d1 ** 2 + d2 ** 2)


if __name__ == "__main__":
    beta = 0.5
    omg = 5
    x0 = 0
    x_left = -2
    x_right = 2
    y0 = 0
    y_left = 1
    y_right = 1
    y_prime = 1
    step = 0.1

    sol = Solver(beta=beta, omg=omg, x0=x0, x_left=x_left, x_right=x_right,
                y0=y0, y_left=y_left, y_right=y_right, y_prime=y_prime, step=step)

    goal_old = sol.Newton_step(y0, y_prime)
    y_res, y_prime_res = sol.Newton_GD(tau=0.05, y0=y0, goal_old=goal_old, y_prime=y_prime, eps=0.01)

    values = []
    y_left_res, left_array = sol.left_shot(y_res, y_res+y_prime_res*step, x0, values=[])
    y_right_res, right_array = sol.right_shot(y_res, y_res+y_prime_res*step, x0, values=[])

    # print(left_array)
    left_array.reverse()
    all_array = left_array+right_array
    plt.plot(all_array)
    plt.show()
