def poisson_binned_log_likelihood(data, expectation):

    # Same as this 
    #log_like = np.sum(np.log(np.power(expectation, data) * np.exp(-expectation) / factorial(data)))
    # The factorial is taken out since it's a constant and only likelihood *differences* matter
    
    log_like = np.sum(data*np.log(expectation) - expectation)
    
    return log_like

def unbinned_log_likelihood(data, expectation):

    """
    expectation  = expectation density = expected event per unit of measured phase space
    """
    
    log_like = -np.size(data) + np.sum(np.log(expectation))

    return log_like

    
