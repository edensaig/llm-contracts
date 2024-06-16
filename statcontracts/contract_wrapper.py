from .optimize import *
import numpy as np
import pandas as pd


contracts = {
    'min-pay': MinPayOptimizer.solve,
    'min-budget': MinBudgetOptimizer.solve,
    'min-variance': MinVarianceOptimizer.solve,
}

#For binary contracts
bnmk_col = 'pass@1'
bnmk_name = lambda bnmk: "humaneval" if "eval"  in bnmk else "mbpp"
bnmks = ['heval', 'mbpp', 'heval_plus', 'mbpp_plus']



class HumanEvalContractWrapper:
    def __init__(self,json_f_path):
        raw_df = pd.read_json(json_f_path).transpose()
        raw_df.index.name = 'model'
        self.contract_df = (
            raw_df
            .assign(heval = lambda df: df[bnmk_col].apply(lambda dct: dct['humaneval']),
                    mbpp = lambda df: df[bnmk_col].apply(lambda dct: dct['mbpp']),
                    heval_plus = lambda df: df[bnmk_col].apply(lambda dct: dct['humaneval']),
                    mbpp_plus = lambda df: df[bnmk_col].apply(lambda dct: dct['mbpp+'])
                    )
            [['heval', 'mbpp', 'heval_plus', 'mbpp_plus']]
        )
    def get_possible_models(self):
        return self.contract_df.index.tolist()

    def get_possible_benchmarks(self):
        return bnmks

    def get_distributions(self, rlvnt_models, rlvnt_bnmks):
        model_predicate = lambda st: any(x == st for x in rlvnt_models)
        return (
            self.contract_df
            .query('model.map(@model_predicate)')
            [rlvnt_bnmks]
        )
    def solve_contract(self, rlvnt_models, costs, rlvnt_bnmks):
        '''
        :param rlvnt_models: list of models to consider
        :param costs: list of costs of each model
        :param rlvnt_bnmks: list of benchmarks to consider
        :return: t_topt_dct: dictionary of the optimal contract for each benchmark
        '''
        assert len(costs) == len(rlvnt_models),\
                'cost array must have the same length as the number of relevant models'
        assert set(rlvnt_bnmks).issubset(set(bnmks)), 'invalid benchmark'

        cost_dct = {mdl: c for mdl, c in zip(rlvnt_models, costs)}
        model_predicate = lambda st: any(x == st for x in rlvnt_models)
        data_df = (
            self.contract_df
            .query('model.map(@model_predicate)')
            .sort_index(key = lambda m: m.map(lambda s: cost_dct[str(s)]))
            [rlvnt_bnmks]
        )

        # check that all models have valid data
        idxs = list(np.where(data_df.isna())[0])
        cols = list(np.where(data_df.isna())[1])
        for idx, col in zip(idxs, cols):
            print(f'{data_df.index[idx]} does not have data for {data_df.columns[col]} benchmark')
        assert data_df.isna().values.sum()==0 , 'Incomplete data'# just to exit if there is any missing data

        c = np.sort(np.array(costs)) # sort the costs along with the models
        t_opt_dct = {}
        for bnmk in rlvnt_bnmks:  # compute optimal contract for each benchmark
            F = data_df[bnmk].to_numpy()
            f_fail  = 100-F
            F = np.concatenate([f_fail.reshape(-1,1), F.reshape(-1,1)], axis = 1)
            F = F / F.sum(axis=1,keepdims=True)
            t_opt_dct[bnmk] = MinPayOptimizer.solve(F,c)  # only need this because all objectives are the same in binary outcome

        return data_df, c, t_opt_dct

class MTBenchContractWrapper:
    def __init__(self,json_f_path):
        raw_df = pd.read_json(json_f_path, lines = True)
        self.contract_df = (
            raw_df
            .assign(
                outcome=lambda df: df['score'].apply(lambda s: int(np.round(np.clip(s, 1, 10)))),
                # Outcome: clip and round scores
            )
            .groupby(['model', 'outcome'])  # Compute scores histogram
            ['tstamp']
            .count()
            .unstack()
            .fillna(0)
            .apply(lambda row: row / row.sum(), axis=1)  # Normalize counts to obtain probabilities
        )
        self.contract_df.index.name = 'models'

    def get_possible_models(self):
        return self.contract_df.index.tolist()

    def get_distributions(self, rlvnt_models):
        model_predicate = lambda st: any(x == st for x in rlvnt_models)
        return self.contract_df.query('models.map(@model_predicate)')

    def solve_contract(self, rlvnt_models, costs,monotone=False):

        assert len(costs) == len(rlvnt_models),\
                'cost array must have the same length as the number of relevant models'
        cost_dct = {mdl: c for mdl, c in zip(rlvnt_models, costs)}
        model_predicate = lambda st: any(x == st for x in rlvnt_models)
        contract_df = (
            self.contract_df
            .query('models.map(@model_predicate)')
            .sort_index(key = lambda m: m.map(lambda s: cost_dct[str(s)]))
        )
        F = contract_df.to_numpy()
        c = np.sort(np.array(costs)) # sort the costs along with the models
        t_opt_dct = {
        contract_name: contract_f(F,c,monotone)
        for contract_name, contract_f in contracts.items()
        }
        return contract_df, c, t_opt_dct








