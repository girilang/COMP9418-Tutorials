# Necessary libraries
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph
# combinatorics
from itertools import product, combinations

from DiscreteFactors import Factor
from Graph import Graph 
from BayesNet import BayesNet 

def factorError(f1, f2):
    """
    argument 
    `f1`, factor with the current state probability distribution in the chain.
    `f2`, factor with the previous state probability distribution in the chain.
    
    Returns absolute error between f1 and f2. 
    """
    assert f1.domain == f2.domain
    return np.sum(np.abs(f1.table - f2.table))

class MarkovModel():
    def __init__(self, start_state, transition, variable_remap):
        '''
        Takes 3 arguments:
        - start_state: a factor representing the start state. E.g. domain might be ('A', 'B', 'C')
        - transition: a factor that represents the transition probs. E.g. P('A_next', 'B_next', 'C_next' | 'A', 'B', 'C')
        - variable_remap: a dictionary that maps new variable names to old variable names,
                          to reset the state after transition. E.g. {'A_next':'A', 'B_next':'B', 'C_next':'C'}
        '''
        self.state = start_state
        self.transition = transition
        self.remap = variable_remap

    def forward(self):
        # get state vars (to be marginalized later)
        state_vars = self.state.domain
        # join with transition factor
        f = self.state * self.transition
        # marginalize out old state vars, leaving only new state vars
        for var in state_vars:
            f = f.marginalize(var)
        # remap variables to their original names
        f.domain = tuple(self.remap[var] for var in f.domain)
        self.state = f
        return self.state

    def forwardBatch(self, n):
        ''' Do `n` steps, and return history list of states '''
        history = []
        for i in range(n):
            f = self.forward()
            history.append(f)
        return history

    def forwardUntilConvergence(self, max_iters=1000, eps=0.000001):
        '''
        Arguments:
        `n`, maximum number of time updates.
        `eps`, error threshold to determine convergence.
        Returns:
        A history of error values
        A factor that represents the current state of the chain after n time steps or the convergence error is less than eps.
        '''
        errors = []
        prevState = self.state
        for i in range(max_iters):
            # forward
            newState = self.forward() 
            # calculate error
            error = factorError(prevState, newState)
            # break loop if error is small
            if error < eps:
                break
            # append error to errors list
            errors.append(error)
            prevState = newState
        return errors, newState

class HiddenMarkovModel():
    def __init__(self, start_state, transition, emission, variable_remap):
        '''
        Takes 3 arguments:
        - start_state: a factor representing the start state. E.g. domain might be ('A', 'B', 'C')
        - transition: a factor that represents the transition probs. E.g. P('A_next', 'B_next', 'C_next' | 'A', 'B', 'C')
        - emission: emission probabilities. E.g. P('O' | 'A', 'B', 'C')
        - variable_remap: a dictionary that maps new variable names to old variable names,
                            to reset the state after transition. E.g. {'A_next':'A', 'B_next':'B', 'C_next':'C'}
        '''
        self.state = start_state
        self.transition = transition
        self.emission = emission
        self.remap = variable_remap

        # These lists will be used later to find the mostly likely sequence of states
        self.history = []
        self.prev_history = []

    def forward(self, **emission_evi):
        # get state vars (to be marginalized later)
        state_vars = self.state.domain

        # join with transition factor
        f = self.state * self.transition 

        # marginalize out old state vars, leaving only new state vars
        for var in state_vars:
            f = f.marginalize(var)

        # remap variables to their original names
        f.domain = tuple(self.remap[var] for var in f.domain)
        self.state = f

        # set emission evidence
        emissionFactor = self.emission.evidence(**emission_evi) 

        # join with state factor
        f = f * emissionFactor 

        # marginalize out emission vars
        for var in f.domain:
            if var not in state_vars:
                f = f.marginalize(var)
        self.state = f

        # normalize state (keep commented out for now)
        # self.state = self.state.normalize()

        return self.state

    def forwardBatch(self, n, **emission_evi):
        '''
        emission_evi: A dictionary of lists, each list containing the evidence list for a variable. 
                         Use `None` if no evidence for that timestep
        '''
        history = []
        for i in range(n):
            # select evidence for this timestep
            evi_dict = dict([(key, value[i]) for key, value in emission_evi.items() if value[i] is not None])
            
            # take a step forward
            state = self.forward(**evi_dict) 
            history.append(state)
        return history

    def viterbi(self, **emission_evi):
        '''
        This function is very similar to the forward algorithm. 
        For simplicity, we will assume that there is only one state variable, and one emission variable.
        '''

        # confirm that state and emission each have 1 variable 
        assert len(self.state.domain) == 1
        assert len(self.emission.domain) == 2
        assert len(self.transition.domain) == 2

        # get state and evidence var names (to be marginalized and maximised out later)
        state_var_name = self.state.domain[0]
        emission_vars = [v for v in self.emission.domain if v not in self.state.domain]
        emission_var_name = emission_vars[0]
        # print('state var name = ', state_var_name)
        # join with transition factor
        f = self.state * self.transition

        # maximize out old state vars, leaving only new state vars
        f, prev = f.maximize(state_var_name, return_prev = True) # (use return_prev to also return prev)
        self.prev_history.append(prev) # save prev for use in traceback

        # remap variables to their original names
        f.domain = tuple(self.remap[var] for var in f.domain)
        self.state = f
        # print('after maximum \n', f)
        # set emission evidence
        emissionFactor = self.emission.evidence(**emission_evi) #

        # join with state factor
        f = f * emissionFactor # 
        # print('after emmission factor \n', f)

        # marginalize out emission vars
        for var in f.domain:
            if var != state_var_name:
                f = f.marginalize(var)
        self.state = f

        # normalize state (keep commented out for now)
        # self.state = self.state.normalize()

        self.history.append(self.state)

        return self.state

    def viterbiBatch(self, n,  **emission_evi):
        '''
        emission_evi: A dictionary of lists, each list containing the evidence list for a variable. 
                         Use `None` if no evidence for that timestep
        '''
        for i in range(n):
            # get evidence for this timestep
            evi_dict = dict([(key, value[i]) for key, value in emission_evi.items() if value[i] is not None])
            self.viterbi(**evi_dict) # take a step using the `viterbi` method
        return self.history

    def traceBack(self):
        '''
        This function iterates backwards over the history to find the most 
        likely sequence of states.
        For simplicity, this function assumes there is one state variable
        '''
        # get most likely outcome of final state
        index = np.argmax(self.history[-1].table)
        
        # Go through "prev_history" in reverse
        indexList = []
        for prev in reversed(self.prev_history):
            indexList.append(index)
            index = prev[index]
        indexList = reversed(indexList)

        # translate the indicies into the outcomes they represent
        mleList = []
        stateVar = self.state.domain[0]
        for idx in indexList:
            mleList.append(self.state.outcomeSpace[stateVar][idx]) 
        return mleList

