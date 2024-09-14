import numpy as np

def richardson_lucy(data, model, response, response_norm = None, niter = 1):
    """
    Data size M

    Model since N

    Response size MxN

    Response_Norm size N

    response_norm =  None, then projected response
    """

    if response_norm is None:
        response_norm = np.sum(response, axis = 0)
    
    # RL iterations
    for t in range(niter):

        expectation = np.dot(response, model)

        # nansum skips cases where both data and expectation are 0
        coeff = np.nansum( (data/expectation)[:,None] * response, axis = 0)  

        model *= coeff / response_norm

    return model
