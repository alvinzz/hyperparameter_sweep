import numpy as np
import copy
import itertools

DEFAULTS = {
    'SWEEP_TYPE': 'exp',
    'SWEEP_NUM': 10,
}

class Sweeper(object):
    """ param_dict should contain entries of the form:
        {param_name: {type, range, sweep_type, sweep_num}},
        WHERE:
        param_name is:
            a string,
        type is:
            'discrete' or 'continuous',
        range is:
            EITHER
                a list of possible values (if type == 'discrete')
            OR
                the start and end of the possible range (if type == 'continuous')
                    (must have start > 0 and start < end)
        sweep_type (for type == 'continuous' only) is:
            'linear'
            'exp' (e.g. 1e-8, 1e-7, 1e-6, ...)
            ##### REV_EXP IS CURRENTLY BUGGY #####
            'rev_exp' (e.g. 1 - e-6, 1 - e-7, 1 - e-8, ...)
        sweep_num (for type == 'continuous' only) is:
            an integer
                (we check sweep_num values in the possible range,
                distributed according to sweep_type)

        model_create_fn should create a Model with desired parameters,
        WHERE:
            we can call Model.train() and Model.eval() on a predefined dataset

        squeeze_factor is a float describing how much to shrink the search range after each iteration
    """
    def __init__(self, param_dict, model_create_fn, squeeze_factor=0.8):
        self.param_dict = param_dict
        self.model_create_fn = model_create_fn
        self.results = {}
        self.squeeze_factor = squeeze_factor

    def eval(self, params):
        return self.eval(params)

    def gridSweep(self, num_iters=1):
        return self.sweep(num_iters=num_iters, method='deterministic')

    def randomSweep(self, num_iters=1):
        return self.sweep(num_iters=num_iters, method='stochastic')

    def sweep(self, num_iters=1, method='stochastic'):
        if not self.results:
            for param in copy.deepcopy(list(self.param_dict.keys())):
                if param.endswith('_'):
                    del self.param_dict[param]
            for param in copy.deepcopy(list(self.param_dict.keys())):
                if self.param_dict[param]['type'] == 'continuous':
                    self.param_dict[param+'_'] = copy.deepcopy(self.param_dict[param])
        if not num_iters:
            res = max(self.results.items(), key=lambda kv: kv[1])
            self.results = {}
            return sorted(res[0]), res[1]
        self.discretize_continuous(method=method)
        combs = []
        for i, param in enumerate(filter(lambda k: self.param_dict[k]['type'] == 'discrete', self.param_dict.keys())):
            if i == 0:
                combs = [(param, value) for value in self.param_dict[param]['range']]
            elif i == 1:
                newvals = [(param, value) for value in self.param_dict[param]['range']]
                combs = itertools.product(combs, newvals)
            else:
                newvals = [(param, value) for value in self.param_dict[param]['range']]
                combs = itertools.product(combs, newvals)
                combs = map(lambda c: c[0] + tuple(c[1:]), combs)
        thisResults = {}
        for comb in combs:
            params = {pv[0]: pv[1] for pv in comb}
            model = self.model_create_fn(params)
            model.train()
            self.results[tuple(sorted(params.items()))] = model.eval()
            thisResults[tuple(sorted(params.items()))] = self.results[tuple(sorted(params.items()))]
        # get top 20% of results, and narrow the range
        sortedResults = sorted(thisResults.keys(), key=lambda k: thisResults[k])
        sortedResults = list(reversed(sortedResults))[:len(sortedResults) // 5 + 1]
        combs = [{pv[0]: pv[1] for pv in comb} for comb in sortedResults]
        for param in self.param_dict:
            if param.endswith('_') and self.param_dict[param]['type'] == 'continuous':
                if self.param_dict[param]['sweep_type'] == 'linear':
                    #var = np.var([comb[param+'discrete'] for comb in combs])
                    mean = np.mean([comb[param+'discrete'] for comb in combs])
                    span = self.param_dict[param]['range'][1] - self.param_dict[param]['range'][0]
                    self.param_dict[param]['range'] = [
                        #mean - max(var, 2*span/self.param_dict[param]['sweep_num']),
                        #mean + max(var, 2*span/self.param_dict[param]['sweep_num'])
                        max(1e-12, mean - span/2.*self.squeeze_factor),
                        mean + span/2.*self.squeeze_factor
                    ]
                    #print(param, self.param_dict[param]['range'])
                elif self.param_dict[param]['sweep_type'] == 'exp':
                    #var = np.var([np.log(comb[param+'discrete']) for comb in combs])
                    mean = np.mean([np.log(comb[param+'discrete']) for comb in combs])
                    span = np.log(self.param_dict[param]['range'][1]) - np.log(self.param_dict[param]['range'][0])
                    self.param_dict[param]['range'] = [
                        #np.exp(mean - max(var, 2*span/self.param_dict[param]['sweep_num'])),
                        #np.exp(mean + max(var, 2*span/self.param_dict[param]['sweep_num']))
                        max(1e-12, np.exp(mean - span/2.*self.squeeze_factor)),
                        np.exp(mean + span/2.*self.squeeze_factor)
                    ]
                    #print(param, self.param_dict[param]['range'])
                elif self.param_dict[param]['sweep_type'] == 'rev_exp':
                    #var = np.var([np.log(sum(self.param_dict[param]['range']) - comb[param+'discrete']) for comb in combs])
                    mean = np.mean([np.log(self.param_dict[param]['range'][0]) + np.log(self.param_dict[param]['range'][1]) 
                        - np.log(comb[param+'discrete']) for comb in combs])
                    span = np.log(self.param_dict[param]['range'][1]) - np.log(self.param_dict[param]['range'][0])
                    self.param_dict[param]['range'] = [
                        #sum(self.param_dict[param]['range']) - np.exp(mean + max(var, 2*span/self.param_dict[param]['sweep_num'])),
                        #sum(self.param_dict[param]['range']) - np.exp(mean - max(var, 2*span/self.param_dict[param]['sweep_num']))
                        max(1e-12, sum(self.param_dict[param]['range']) - np.exp(mean + span/2.*self.squeeze_factor)),
                        sum(self.param_dict[param]['range']) - np.exp(mean - span/2.*self.squeeze_factor)
                    ]
                    #print(param, self.param_dict[param]['range'])
        return self.sweep(num_iters=num_iters-1, method=method)

    def discretize_continuous(self, method='stochastic'):
        for param in copy.deepcopy(list(self.param_dict.keys())):
            if self.param_dict[param]['type'] == 'continuous' and param.endswith('_'):
                range = self.param_dict[param]['range']
                try:
                    sweep_type = self.param_dict[param]['sweep_type']
                except:
                    sweep_type = DEFAULTS['SWEEP_TYPE']
                try:
                    sweep_num = self.param_dict[param]['sweep_num']
                except:
                    sweep_num = DEFAULTS['SWEEP_NUM']
                if sweep_type == 'linear':
                    if method == 'deterministic':
                        range = np.linspace(range[0], range[1], num=sweep_num, endpoint=True).tolist()
                    elif method == 'stochastic':
                        range = np.random.uniform(range[0], range[1], size=sweep_num).tolist()
                elif sweep_type == 'exp':
                    if method == 'deterministic':
                        range = np.exp(np.linspace(np.log(range[0]), np.log(range[1]), num=sweep_num, endpoint=True)).tolist()
                    elif method == 'stochastic':
                        range = np.exp(np.random.uniform(np.log(range[0]), np.log(range[1]), size=sweep_num)).tolist()
                elif sweep_type == 'rev_exp':
                    if method == 'deterministic':
                        range = (range[1] + range[0] -
                                    np.exp(np.linspace(np.log(range[0]), np.log(range[1]), num=sweep_num, endpoint=True))
                                ).tolist()
                    elif method == 'stochastic':
                        range = (range[1] + range[0] -
                                    np.exp(np.random.uniform(np.log(range[0]), np.log(range[1]), size=sweep_num))
                                ).tolist()
                self.param_dict[param+'discrete'] = {'type': 'discrete', 'range': range}

class Model():
    def __init__(self, params):
        self.params = params
        self.target = {
            'a': 4.7,
            'b_discrete': 3.5,
            'c_discrete': 10**1.2,
            'd_discrete': 10**3.7
        }
    def train(self):
        pass

    def eval(self):
        return -sum([(self.params[k] - self.target[k])**2 if k in ['a', 'b_discrete']
            else np.log(np.abs(self.params[k] - self.target[k]))**2
            for k in self.params.keys()])

if __name__ == '__main__':
    params = {
                'a': {'type': 'discrete', 'range': [4,5,6]},
                'b': {'type': 'continuous', 'range': [1, 5], 'sweep_type': 'linear', 'sweep_num': 10},
                'c': {'type': 'continuous', 'range': [1, 1e2], 'sweep_type': 'exp', 'sweep_num': 10},
                'd': {'type': 'continuous', 'range': [1e2, 1e4], 'sweep_type': 'exp', 'sweep_num': 10},
                #'d': {'type': 'continuous', 'range': [1e2, 1e4], 'sweep_type': 'rev_exp', 'sweep_num': 5},
                }
    s = Sweeper(params, Model)
    print('targs:', sorted(Model(None).target.items()))
    print('best, random:', s.randomSweep(num_iters=20))
    print('best, grid:', s.gridSweep(num_iters=20))

