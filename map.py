import pandas as pd
import cvxpy as cp
import numpy as np
from lazy import lazy


class mgrallocator(object):
    '''
    mgrallocator solves the manager allocation problem.
    It aligns the model setting explained in notebook.
    '''

    def __init__(self, P, Q, y_s, F, Omega, lam, ):
        '''

        :param P: int, number of manager groups
        :param Q: int, number of factors/dimensions
        :param y_s: np array (Q-size vector), targeted exposures on factors/dimensions
        :param F: np.array (Q*P matrix), mapping manager allocation to corresponding factor exposures
        :param Omega: np.array (Q*Q matrix, symmetric semi-positive definite), penalty of factor deviation
        :param lam: float, penalty of concentration. Usually small comparing to deviation penalty
        '''

        self.P= P
        self.Q= Q
        self.y_s= y_s
        self.F= F
        self.Omega= Omega
        self.lam= lam


        self.global_optimum= None


    def solve_grid (self):


        return 0


    def _cal_exp(self, y):
        '''
        calculate factor exposure given manager allocation

        :param y: np array (P-size vector), the manager allocation
        :return: np array (Q-size vector), the corresponding factor exposure
        '''

        return np.matmul(self.F, y).flatten()



    def solve_global (self, ):


        return 0






