import numpy as np
import cvxpy as cp

__all__ = [
    'MinPayOptimizer',
    'MinBudgetOptimizer',
    'MinVarianceOptimizer',
    'InterpolationOptimizer',
]

class LPContractOptimizer:
    solver_params = {}
    
    @classmethod
    def objective(cls,F,c,t):
        raise NotImplemented

    @classmethod
    def constraints(cls,F,c,t,monotone):
        '''
        :param F: size nxm numpy ndarray of outcome distributions
        :param c: size n numpy array of costs
        :param t: size m numpy array of contract transfers
        :param monotone: Flag indicating that the monotonicity constraint must be enforced
        :return: A list of constraints
        '''
        out = [
            (F[-1] - F[:-1])@t >= c[-1]-c[:-1],
            t>=0,
        ]
        if monotone:
            out.append(t[:-1]<=t[1:])
        return out
        
    @classmethod
    def solve(cls,F,c,monotone=False,**kwargs):
        n,m = F.shape
        t = cp.Variable(m)
        obj = cls.objective(F,c,t,**kwargs)
        constraints = cls.constraints(F,c,t,monotone)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve(**cls.solver_params)
        t_opt = t.value
        return t_opt - t_opt.min()


class MinPayOptimizer(LPContractOptimizer):
    solver_params = {'solver': cp.CLARABEL}
    
    @classmethod
    def objective(cls,F,c,t):
        return F[-1]@t


class MinBudgetOptimizer(LPContractOptimizer):
    solver_params = {'solver': cp.CLARABEL}

    @classmethod
    def objective(cls,F,c,t):
        return cp.max(t)


class MinVarianceOptimizer(LPContractOptimizer):
    @classmethod
    def objective(cls,F,c,t):
        n,m = F.shape
        P = F[-1]
        R = np.diag(np.sqrt(P))@(np.eye(m)-np.tile(P,(m,1)))
        V = R.T@R
        return cp.quad_form(t,V)

class InterpolationOptimizer(LPContractOptimizer):
    solver_params = {'solver': cp.CLARABEL} # supports LP and QP


    _minVariance_objective = MinVarianceOptimizer.objective
    _minPay_objective = MinPayOptimizer.objective
    _minBudget_objective = MinBudgetOptimizer.objective
    @classmethod
    def objective(cls,F,c,t,**kwargs):
        alpha = kwargs.get('alpha', None)
        beta = kwargs.get('beta', None)
        if beta is not None:
            assert alpha is not None, 'Beta must be provided with alpha'
            assert alpha+beta<=1, 'Alpha + Beta must be less than or equal to 1'
        obj = kwargs.get('obj', None)
        objs = {'PB': cls._PB_objective,
                'PV': cls._PV_objective,
                'BV' : cls._BV_objective,
                'PBV': cls._PBV_objective,
                }
        assert obj in objs.keys(), f'Invalid objective: {obj}'
        return objs[obj](F,c,t,alpha,beta)
    @classmethod
    def _PB_objective(cls,F,c,t,alpha,beta):
        return alpha*cls._minPay_objective(F,c,t) + (1-alpha)*cls._minBudget_objective(F,c,t)
    @classmethod
    def _PV_objective(cls,F,c,t,alpha,beta):
        return alpha*cls._minPay_objective(F,c,t) + (1-alpha)*cls._minVariance_objective(F,c,t)
    @classmethod
    def _BV_objective(cls,F,c,t,alpha,beta):
        return alpha*cls._minBudget_objective(F,c,t) + (1-alpha)*cls._minVariance_objective(F,c,t)

    @classmethod
    def _PBV_objective(cls,F,c,t,alpha,beta):
        return alpha*cls._minPay_objective(F,c,t) + beta*cls._minBudget_objective(F,c,t) \
               + (1-alpha-beta)*cls._minVariance_objective(F,c,t)

class ThresholdContractOptimizer():
    @classmethod
    def solve(cls,F,c):
        thresh_contracts_sparse = []
        thresh_contracts = []
        thresholds = np.arange(0, 10, 1)
        for j in thresholds:
            thresh_contracts_sparse.append(cls.solve_threshold_contract(j, F, c))
        for i, cont in enumerate(thresh_contracts_sparse):
            contract = np.array([0] * i + [cont] * len(thresh_contracts_sparse[i:]))
            # print(f'contract for threshold {i}: {contract}')
            thresh_contracts.append(contract)
        # remove contracts with None values
        thresh_contracts = [t for t in thresh_contracts if None not in t]
        # get the objective values of each contract:
        P = F[-1]
        exp_pay = lambda t: P @ t
        budget = lambda t: np.max(t)
        stdev = lambda t: np.sqrt(P @ (t ** 2) - (P @ t) ** 2)

        opt_thresh_contracts = {
            'min-pay': thresh_contracts[np.argmin([exp_pay(t) for t in thresh_contracts])],
            'min-budget': thresh_contracts[np.argmin([budget(t) for t in thresh_contracts])],
            'min-variance': thresh_contracts[np.argmin([stdev(t) for t in thresh_contracts])]
        }
        return opt_thresh_contracts
    @classmethod
    def solve_threshold_contract(cls, j, F, cost):
        sum_f_nj = np.sum(F[-1, j:])
        t = 0
        for i in range(len(F) - 1):
            denom = sum_f_nj - np.sum(F[i, j:])
            if denom <= 0:
                return None
            tmp = (cost[-1] - cost[i]) / denom
            if tmp > t:
                t = tmp
        return t

