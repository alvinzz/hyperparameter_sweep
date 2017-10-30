## Automated Hyperparameter Sweeper
### Usage:
- Create a Sweeper() instance from sweep.py
- Takes the following parameters:

param_dict should contain entries of the form:
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
        'log' (e.g. log(1e-2), log(2e-2), log(3e-2), ...)
    sweep_num (for type == 'continuous' only) is:
        an integer
            - in grid sweep, we check sweep_num values in the possible range,
                distributed according to sweep_type)
            - in random sweep, we use this to calculate how many values to
                sample

model_create_fn should create a Model with desired parameters,
    WHERE:
        we can call Model.train() and Model.eval() on a predefined dataset
            - Model.eval() should return the loss

squeeze_factor is:
    a float
        - how much to shrink the search range after each iteration

top_n is:
    a float
        - the fraction of the top scores from the current iteration
            that are used to determine the range for the next iteration

log_file is:
    a string
        - path to file to write log to
