#!/usr/bin/env python
# -*- coding: utf-8 -*

"""Calculate the solution to non-linear, linearized and the approximate power flow solutions using either root finding methods 
or the restricted newton method for any winding vector introduced in Appendix B. in the corresponding publication
Hartmann et al, 'Synchronized states of power grids and oscillator networks by convex optimization', 2024"""

import numpy as np
from scipy.optimize import root, newton
from scipy import sparse
from scipy.sparse import csgraph
import networkx

from typing import Tuple

import gzip
import pickle

from  opti_power_flow_solution.utils import formulate_matrices

""" Calculate the linear power flow. """

def find_lin_powerflow_incidence_matrix(Pvec, B_d, II, use_pseudo_inverse=False):
    """Solve the linear power flow by using the node edge incidence
    matrix II and the matrix collecting the susceptances 

    Args:
        Pvec (1d numpy array): vector collecting the effective power injections
        B_d (sparse matrix): Susceptance matrix
        II (sparse matrix): Node Edge Incidence matrix
        
    Returns:

        theta, flows: Linear phase angles, Power flows
    """
    
    laplacian = II @ B_d @ II.T
    
    if use_pseudo_inverse:
        theta_lin = np.linalg.pinv(laplacian) @ Pvec
        
    else:
        theta_lin = np.zeros(laplacian.shape[0])
        theta_lin[1:] = np.linalg.solve(laplacian[1:, 1:], Pvec[1:])
        #theta_lin[1:] = sparse.linalg.spsolve(laplacian[1:, 1:], Pvec[1:])
    
    flows_lin = B_d @ (II.T @ theta_lin)
    
    return theta_lin, flows_lin

""" Find non-linear power flow using root finding. """

def find_non_lin_powerflow_theta(power_injected, KK_matr, node_edge_incidence_matr,
                                 initial_guess_theta=None, tol=1e-10, verbose=False,
                                 solver_method="hybr"):
    """Find the non-linear power flow solution given by the power phase angles

    Args:
        power_injected (_type_): Vector with power injections
        KK_matr (_type_): Matrix that collect the transmission capacities on the diagonal.
        node_edge_incidence_matr (_type_): Node-edge incidence matrix
        initial_guess_theta (_type_, optional): initial guess of the phase angles as a vector. Defaults to None.
        tol (_type_, optional): Tolerance of the solver. Defaults to 1e-12.
        verbose (bool, optional): if True, print out extra details. Defaults to False.
        solver_method (str, optional): Solver methods used by scipy. 
        Possible values: 'hybr', 'broyden1' or 'newton_rhapson' Defaults to "newton_rhapson".
        
    Returns:
        theta_sol (array): Array with phase angles for each node. If 'np.nan' is returned,
        no solution was found by the solver.
    """
    
    def non_lin_prob(state):
        theta_arr = state

        power_flow = node_edge_incidence_matr @ KK_matr @ np.sin(node_edge_incidence_matr.T @ theta_arr)
        
        diff_power = power_injected - power_flow
        
        return diff_power
    
    NN = len(power_injected)
    
    if initial_guess_theta is None:
        initial_guess_theta = find_lin_powerflow_incidence_matrix(power_injected,
                                                                  KK_matr,
                                                                  node_edge_incidence_matr)
    
    if solver_method in ['broydn1', 'hybr']:
        sol = root(non_lin_prob, initial_guess_theta, method=solver_method, tol=tol)
    
        if sol.success:
            return sol.x
    
        else:
            return np.nan
        
    elif solver_method == 'newton_rhapson':
        sol_theta, did_converge, _ = newton(non_lin_prob,
                                            initial_guess_theta,
                                            tol=tol, full_output=True,
                                            maxiter=200)
        
        if did_converge.all():
            return sol_theta
        else:
            return np.nan
        
    else:
        raise NotImplementedError("Solver method '{}' not implemented!".format(solver_method))
        
def find_correction_term_linear_powerflow(linear_pf: np.ndarray, edge_weights: np.ndarray, 
                                          node_edge_incidence_matrix: np.ndarray,
                                          edge_cycle_incidence_matrix: np.ndarray) -> np.ndarray:
    """Find the correction term of the linear powerflow by evaluating the loop flow
    amplitudes by using approximation in equation . The correction to the powerflows
    is then given by 'edge_cycles_incidence_matrix@loop_flows_amplitude'.
    
    Args:
        linear_pf (1D array): power flows resulting form the linear power flow solution
        edge_weights (1D array): Weights of ech edges given by the admittance
        node_edge_incidence_matrix (2D array): Node edge incidence matrix.
        edge_cycle_incidence_matrix (m X C_n array): Edge cycle incidence matrix that describes
        which edge belongs to which cycle.
        verbose (bool, optional): If 'True' additional information is being printed. Defaults to False.

    Returns:
        loop_flow_amplitude (1D numpy.ndarray): Loop flow amplitude.
    """
    
    KK_e = edge_weights    
    KK_red = np.diag([(KK_e[xx]**2 - linear_pf[xx]**2)**-.5 for xx in range(len(KK_e))])
    
    # Rename matrices to be consistent with paper
    EE = node_edge_incidence_matrix
    CC = edge_cycle_incidence_matrix
    
    # Solve for loop flow amplitudes
    ll_lhs = CC.transpose() @ KK_red @ CC
    ll_rhs = - CC.transpose() @ np.arcsin(linear_pf/edge_weights)
    
    ll = np.linalg.solve(ll_lhs, ll_rhs)
    
    return ll


def translate_theta_node_to_edge(theta_node: np.ndarray, node_edge_incidence_matrix):
    """Translate the results for the phase angles to a delta that for each edge 
    with ordering and direction according to node_edge_incidence matrix.

    Args:
        theta_node (vector): _description_
        node_edge_incidence_matrix (matrix): _description_
    """

    theta_delta = node_edge_incidence_matrix.T @ theta_node
    
    return theta_delta

""" Find nonlinear power flow using restricted Newton """

def __gradient_F_z__(ll_vec, flow_vec, winding_vec, K_vec, C_matr):
    """
    Calculate the gradient of \mathcal{F}_z(\ell) at a flow vector f^{(0)}.

    Args:
        ll_vec: loop_flow vector \ell
        flow_vec: flow_vector f^{(0)}
        winding_vec: winding vector z. 
        K_vec: Vector of line capacities. 
        C_matr: cycle edge incidenc matrix. 
    """
    return C_matr.T @ np.arcsin((flow_vec + C_matr @ ll_vec)/K_vec) - 2 * np.pi * winding_vec

def __Hessian_F_z__(ll_vec, flow_vec, K_vec, C_matr):
    """
    Calculate the Hessian of \mathcal{F}_z(\ell) at a flow vector f^{(0)}.

    Args:
        ll_vec: loop_flow vector \ell
        flow_vec: flow_vector f^{(0)}
        K_vec: Vector of line capacities. 
        C_matr: cycle edge incidenc matrix. 
    """
    diagonal_hess_vec = 1 / np.sqrt(K_vec**2 - (flow_vec + C_matr @ ll_vec)**2)
    return C_matr.T @ np.diag(diagonal_hess_vec) @ C_matr

def __restricted_newton__(f_initial, winding_vec, K_vec, C_matr, maxiter=10, tol=1e-15, check_feasible=True, lambda_reg=1):
    """Calculate the solutions to the linear and nonlinear power flow
    equations using the restricted newton devised in Appendix B of the paper.

    Args:
        f_initial: inital guess for the flow vector. Muss fulfill the KCL. 
        winding_vector (list, optional): Winding vector that determines the loop flow solution. Defaults to None. 
        K_vec: Vector of line capacities. 
        C_matr: cycle edge incidenc matrix. 
        maxiter (int, optional): Maximal number of iterations of Newton steps. Defaults to 100. 
        tol (float, optional): Stops criterion, if difference in gradient is smaller than tol in consecutive iterations. Defaults to 1e-15. 
        check_feasible (bool, optional): Check inequality constraints, that is, no lines are overloaded. Breaks and returns not_feasible if 
            violated. Defaults to True. 
        lambda_reg (float, optional): Regularization as step size factor. Defaults to 1. 

    """
    # initialize 
    f_n = f_initial * 1
    if check_feasible is True: 
        is_feasible = True
    else: 
        is_feasible = None
    if tol is None:
        tol = 0

    # iterate
    for nn_iter in range(1, maxiter+1):
        ## Solve for the n-th iteration
        # get matrices
        hess = __Hessian_F_z__(ll_vec=np.zeros(len (winding_vec)), flow_vec=f_n, K_vec=K_vec, C_matr=C_matr)
        grad = __gradient_F_z__(ll_vec=np.zeros(len (winding_vec)), flow_vec=f_n, winding_vec=winding_vec, K_vec=K_vec, C_matr=C_matr)
        # solve for ll_vec(n)
        ll_vec_n = np.linalg.solve(hess, -1*grad)
        # update flows
        f_n_plus_1 = f_n + lambda_reg * C_matr @ ll_vec_n

        ## check inequaltiy constraint 
        if check_feasible is True:
            # break if line loadings are not respceted 
            if np.max(f_n_plus_1 - K_vec) > 0: 
                is_feasible = False
                break 
            else:
                f_n = f_n_plus_1 * 1
        else:
            f_n = f_n_plus_1 * 1

        ## break if gradient is not changing anymore 
        if nn_iter > 1: 
            if np.sum(grad - grad_nn_minus_1) <= tol:
                break
        grad_nn_minus_1 = grad * 1
            
    return is_feasible, f_n, nn_iter

""" Wrapper Functions  """

def calculate_power_flow_solutions(graph: networkx.Graph,
                                   save_it: bool = True,
                                   results_save_path:str = None,
                                   power_factor: float = 1.,
                                   use_pinv: bool = False,
                                   initialize_theta: list = None
                                   ) -> Tuple[np.ndarray, np.ndarray,
                                                                    np.ndarray, np.ndarray,
                                                                    np.ndarray]:
    """Calculate the solutions to the linear and nonlinear power flow
    equations as well as the correction term devised in the paper.

    Args:
        graph (networkx.Graph): Graph with electrical details.
        save_it (bool, optional): If 'True' results get saved. Defaults to True.
        results_save_path (str, optional): Path + filename to save the data.
        power_factor (float, optional): Factor that multiplies the power injections/extractions
            of each node, which effectively controls the loading of the grid. Defaults to 1..

    Returns:
        thetas_lin: array with solutions of the linear power flow eqs.
        thetas_nonlin: array with the solutions of the nonlinear power flow eqs
        flows_lin: flows for each edge resulting from thetas_lin
        flows_nonlin: flows for each edge resulting from thetas_nonlin
        flows_corrected: flows_lin + correction term that was derived in the paper
    """
       
    # Build power injections and adjacency matrix
    node_name_n_Pi = np.array(list(networkx.get_node_attributes(graph, 'Pi').items()))
    sort_idxs = np.argsort(node_name_n_Pi[:, 0])
    Pvec = node_name_n_Pi[sort_idxs, 1] * power_factor
    
    # Find node_edge_incidence and edge_cycle_incidence matrix
    _, node_edge_inci, BBvec = formulate_matrices.construct_node_edge_incidence_matrix_from_graph(graph)
    
    KK_matr = np.diag(BBvec)
    
    edge_cycle_inci = formulate_matrices.edge_cycle_incidence_matrix(node_edge_inci, 
                                                                     graph, use_minimal_cyc_basis=True)
    # Linear solution
    thetas_lin, flows_lin = find_lin_powerflow_incidence_matrix(Pvec, KK_matr, 
                                                           node_edge_inci, use_pseudo_inverse=use_pinv)


    # Non-linear solution
    thetas_nonlin = find_non_lin_powerflow_theta(Pvec, KK_matr, node_edge_inci,
                                                                     initial_guess_theta=thetas_lin)
    
    # if no solution has been found using linear solution as initial guess, use initial thetas from initialize_theta
    if np.isnan(thetas_nonlin).any():
        if initialize_theta is not None: 
            thetas_nonlin = find_non_lin_powerflow_theta(Pvec, KK_matr, node_edge_inci,
                                                                     initial_guess_theta=initialize_theta)
    else:
        pass
    
    # compute the flows
    if np.isnan(thetas_nonlin).any():
        flows_nonlin = np.nan
        print('No non linear solution for power factor {}.'.format(power_factor))
    
    else:
        flows_nonlin = KK_matr @ np.sin(node_edge_inci.T @ thetas_nonlin)
    
    # Correction
    thetas_lin_edge = node_edge_inci.T @ thetas_lin
    ll_correct = find_correction_term_linear_powerflow(flows_lin, np.diag(KK_matr),
                                                                           node_edge_inci,
                                                                           edge_cycle_inci)
    
    
    flows_corrected = flows_lin + edge_cycle_inci @ ll_correct
    
    # Save results
    if save_it:
        if results_save_path is None:
            script_path = os.path.dirname(__file__)
            results_save_path = script_path + "/results_data/last_test_graph/"
            if not os.path.exists(results_save_path):
                os.mkdir(results_save_path)
            results_save_path = results_save_path + "/powerflow_powerfac{0:.4f}.pklz".format(pwr_fac_r)

        out_path = results_save_path
        
        res_tup = (thetas_lin, thetas_nonlin,
                   flows_lin, flows_nonlin, flows_corrected)
        
        with gzip.open(out_path, 'wb') as fh_out:
            pickle.dump(res_tup, fh_out)
    
    return thetas_lin, thetas_nonlin, flows_lin, flows_nonlin, flows_corrected


def find_power_flow_solutions_restricted_newton(graph:networkx.Graph, 
                                                power_factor:int=1, 
                                                winding_vector:list=None, 
                                                maxiter:int=100, 
                                                tol:float=1e-15, 
                                                check_feasible:bool=True, 
                                                lambda_reg:float=1. ,
                                                verbose:int=0) -> Tuple[np.ndarray, np.ndarray,
                                                                    bool, np.ndarray]:
    """Calculate the solutions to the linear and nonlinear power flow
    equations using the restricted newton devised in Appendix B of the paper.

    Args:
        graph (networkx.Graph): Graph with electrical details.
        power_factor (float, optional): Factor that multiplies the power injections/extractions
            of each node, which effectively controls the loading of the grid. Defaults to 1..
        winding_vector (list, optional): Winding vector that determines the loop flow solution. Defaults to None. 
        maxiter (int, optional): Maximal number of iterations of Newton steps. Defaults to 100. 
        tol (float, optional): Stops criterion, if difference in gradient is smaller than tol in consecutive iterations. Defaults to 1e-15. 
        check_feasible (bool, optional): Check inequality constraints, that is, no lines are overloaded. Breaks and returns not_feasible if 
            violated. Defaults to True. 
        lambda_reg (float, optional): Regularization as step size factor. Defaults to 1. 
        verbose (int, optional): If > 1, print results in console. Defaults to 0. 

    Returns:
        flows_lin: linear flows for each edge 
        flows_nonlin: nonlinear flows for each edge for winding vector 
        is_feasible: boolean value, indicating whether a feasible nonlinear flow could have been found.
        C_matr: Edge cycle incidence matrix. Encodes the information about the cycle basis used. 
    """
    # Build power injections and adjacency matrix
    node_name_n_Pi = np.array(list(networkx.get_node_attributes(graph, 'Pi').items()))
    sort_idxs = np.argsort(node_name_n_Pi[:, 0])
    Pvec = node_name_n_Pi[sort_idxs, 1] * power_factor
   
    # Find node_edge_incidence and edge_cycle_incidence matrix
    _, E_matr, K_vec = formulate_matrices.construct_node_edge_incidence_matrix_from_graph(graph)
    
    K_matr = np.diag(K_vec)
    
    C_matr = formulate_matrices.edge_cycle_incidence_matrix(E_matr, graph, use_minimal_cyc_basis=True)
    N_cycles = C_matr.shape[1]

    if winding_vector is None: 
        winding_vector = np.zeros(N_cycles)
    else:
        assert len(winding_vector) == N_cycles, 'Winding vector does not match Cycle Space dimension.'

    # find initial guess uisng the linear flow
    _, flows_lin = find_lin_powerflow_incidence_matrix(Pvec, K_matr, 
                                                           E_matr, use_pseudo_inverse=False)
    
    # calculate power flow 
    is_feasible, flows_non_lin_z, n_iterations = __restricted_newton__(flows_lin, winding_vector, K_vec, C_matr, maxiter, tol, check_feasible=check_feasible, lambda_reg=lambda_reg)
    if verbose > 0: 
        print('Restricted Newton:')
        print('Found a feasible real power flow solution: ', is_feasible)
        print('Numer of iterations needed: ', n_iterations)

    return flows_lin, flows_non_lin_z, is_feasible, C_matr