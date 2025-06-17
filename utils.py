import numpy as np
import scipy
import cvxpy as cp

#leaves returned by sklearn are not incremental by one. 
# They can be for instance 1,3,4,7. This mapping should return (in this case) 0,1,2,3
#this will be needed (I think) to create the matrices in a better way. 
# We also need it to map test datapoints to correct leaves
def map_leaves_to_number(leaves):
    unique_leaves = sorted(np.unique(leaves))
    mapped_values = np.arange(len(unique_leaves))
    mapping = []
    for leaf in leaves:
        index = np.where(unique_leaves == leaf)[0][0]
        mapping.append(mapped_values[index])

    mapping = np.array(mapping)
    return mapping

def compute_rmse(predictions, targets):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.sqrt(np.mean((predictions - targets)**2))

def compute_mse(predictions, targets):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.mean((predictions - targets)**2)

def compute_mae(predictions, targets):
    """Compute Root Mean Squared Error (RMSE)."""
    return np.mean(np.abs(predictions - targets))

def compute_huber(predictions, targets, delta):
    residuals = np.abs(predictions - targets)
    lad_mask = residuals > delta   # True where residual > delta (outliers)
    mse_part = compute_mse(predictions[~lad_mask], targets[~lad_mask])
    lad_part = compute_mae(predictions[lad_mask], targets[lad_mask])*delta - delta**2 / 2
    return mse_part + lad_part


def early_stopping(es_rounds, val_loss_list, tol = 1e-6):
    recent_min = np.min(val_loss_list[-es_rounds:])
    global_min = np.min(val_loss_list)

    if recent_min > global_min + tol:
        return True  # No improvement recently → stop
    else:
        return False  # Still improving → continue

def find_gamma_gammahat(unique_leaves_clf, indexed_leaves_clf, 
                        unique_leaves_clfhat, indexed_leaves_clfhat,
                        y_train_target_residuals):
    
    R = np.zeros((len(unique_leaves_clf), len(unique_leaves_clf)))
    for leaf_clf in indexed_leaves_clf:
        R[leaf_clf, leaf_clf] += 1

    Rhat = np.zeros((len(unique_leaves_clfhat), len(unique_leaves_clfhat)))
    for leaf_clfhat in indexed_leaves_clfhat:
        Rhat[leaf_clfhat, leaf_clfhat] += 1

    N = np.zeros((len(unique_leaves_clf), len(unique_leaves_clfhat)))
    for leaf_clf, leaf_clfhat in zip(indexed_leaves_clf, indexed_leaves_clfhat):
        N[leaf_clf, leaf_clfhat] += 1

    upper = np.hstack((R, N))
    lower = np.hstack((N.T, Rhat))
    M = np.vstack((upper, lower))

    r1 = np.zeros(len(unique_leaves_clf))
    r2 = np.zeros(len(unique_leaves_clfhat))
    for index in sorted(np.unique(indexed_leaves_clf)):
        indixes = np.where(indexed_leaves_clf == index)[0]
        r1[index] = np.sum(y_train_target_residuals[indixes])
    for index in sorted(np.unique(indexed_leaves_clfhat)):
        indixes = np.where(indexed_leaves_clfhat == index)[0]
        r2[index] = np.sum(y_train_target_residuals[indixes])
    r = np.concatenate((r1, r2))

    gamma_vector = np.linalg.lstsq(M, r, rcond=None)[0]

    leaf_gamma = gamma_vector[:len(unique_leaves_clf)]
    leaf_gammahat = gamma_vector[len(unique_leaves_clf):]
    return leaf_gamma, leaf_gammahat

#In here will be the function for computing optimal coefficients for LAD
def find_gamma_gammahat_LAD(unique_leaves_clf, indexed_leaves_clf, 
                             unique_leaves_clfhat, indexed_leaves_clfhat,
                             y_train_target_residuals):

    # Objective: minimize sum of absolute residuals
    c1 = np.ones((1, len(indexed_leaves_clf)))  
    c2 = np.zeros((1, len(unique_leaves_clf) + len(unique_leaves_clfhat)))
    c = np.concatenate((c1, c2), axis=1).flatten()

    # Constraint matrix
    A1 = np.eye(len(indexed_leaves_clf))
    A2 = np.zeros((len(indexed_leaves_clf), len(unique_leaves_clf) + len(unique_leaves_clfhat)))

    for i, index in enumerate(indexed_leaves_clf):
        A2[i, index] = 1
    for i, index in enumerate(indexed_leaves_clfhat):
        A2[i, index + len(unique_leaves_clf)] = 1

    A = -np.concatenate((A1, A2), axis=1)


    # Constraint bounds
    b_ub = -y_train_target_residuals.copy()

    # Constraint matrix 2
    Ag = np.concatenate((-A1, A2), axis = 1)

    # Constraint bounds 2
    b_ub_g = y_train_target_residuals.copy()


    #Combine two one constraint matrix
    A = np.vstack((A, Ag))
    b_ub = np.concatenate((b_ub, b_ub_g))


    # Solve LP
    result = scipy.optimize.linprog(c, A_ub=A, b_ub=b_ub, bounds=(None, None), method='highs')

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    all_optimal_params = result.x

    gamma_vector = all_optimal_params[len(indexed_leaves_clf):]
    leaf_gamma = gamma_vector[:len(unique_leaves_clf)]
    leaf_gammahat = gamma_vector[len(unique_leaves_clf):]

    return leaf_gamma, leaf_gammahat


def find_lad_ls_indices_delta(y_train_target_residuals, quantile = 0.95):
    delta = np.quantile(np.abs(y_train_target_residuals), q = quantile)
    lad_indices = np.where(np.abs(y_train_target_residuals) > delta)[0]
    ls_indices = np.where(np.abs(y_train_target_residuals) <= delta)[0]


    return lad_indices, ls_indices, delta


#Huber Loss (written by GPT, needs improvement)
def find_gamma_gammahat_Huber(unique_leaves_clf, indexed_leaves_clf, 
                              unique_leaves_clfhat, indexed_leaves_clfhat,
                              y_train_target_residuals, lad_indices, ls_indices, delta):
    

    total_variable_len = len(unique_leaves_clf) + len(unique_leaves_clfhat) + len(indexed_leaves_clf)
    Q = np.eye(total_variable_len) #construct this matrix for the quadratic part
    Q[lad_indices] = 0
    c1 = np.zeros((1, len(indexed_leaves_clf)))  
    c2 = np.zeros((1, len(unique_leaves_clf) + len(unique_leaves_clfhat)))
    c = np.concatenate((c1, c2), axis=1).flatten()
    c[lad_indices] = 1*delta

    # Constraint matrix
    A1 = np.eye(len(indexed_leaves_clf))
    A2 = np.zeros((len(indexed_leaves_clf), len(unique_leaves_clf) + len(unique_leaves_clfhat)))

    for i, index in enumerate(indexed_leaves_clf):
        A2[i, index] = 1
    for i, index in enumerate(indexed_leaves_clfhat):
        A2[i, index + len(unique_leaves_clf)] = 1
    A = -np.concatenate((A1, A2), axis=1)
    A[ls_indices,:] = 0


    # Constraint bounds
    b_ub = -y_train_target_residuals.copy()
    b_ub[ls_indices] = 0

    # Constraint matrix 2
    Ag = np.concatenate((-A1, A2), axis = 1)
    Ag[ls_indices,:] = 0

    # Constraint bounds 2
    b_ub_g = y_train_target_residuals.copy()
    b_ub_g[ls_indices] = 0


    #Combine two one constraint matrix
    A = np.vstack((A, Ag))
    b_ub = np.concatenate((b_ub, b_ub_g))


    n_vars = total_variable_len
    x = cp.Variable(n_vars) 
    objective = cp.Minimize(0.5 * cp.quad_form(x, Q) + c.T @ x)
    constraints = [A @ x <= b_ub]

    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    gamma_vector = x.value[len(indexed_leaves_clf):]
    leaf_gamma = gamma_vector[:len(unique_leaves_clf)]
    leaf_gammahat = gamma_vector[len(unique_leaves_clf):]

    return leaf_gamma, leaf_gammahat
