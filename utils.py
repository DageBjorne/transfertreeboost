import numpy as np
import scipy

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



import numpy as np

def huber_weights(residuals, delta):
    """
    Compute weights for IRLS under Huber loss.

    Parameters:
    - residuals: array of residuals
    - delta: Huber threshold

    Returns:
    - weights: array of weights
    """
    abs_res = np.abs(residuals)
    weights = np.ones_like(residuals)
    mask = abs_res > delta
    weights[mask] = delta / abs_res[mask]
    return weights


#Huber Loss (written by GPT, needs improvement)
def find_gamma_gammahat_Huber(unique_leaves_clf, indexed_leaves_clf, 
                              unique_leaves_clfhat, indexed_leaves_clfhat,
                              y_train_target_residuals, 
                              delta=1.0, max_iter=10, tol=1e-6):

    J = len(unique_leaves_clf)
    K = len(unique_leaves_clfhat)
    N = len(y_train_target_residuals)

    # Build design matrix X: rows = data points, columns = J+K coefficients
    X = np.zeros((N, J+K))
    for i in range(N):
        X[i, indexed_leaves_clf[i]] = 1
        X[i, J + indexed_leaves_clfhat[i]] = 1

    # Initialize coefficients
    gamma_vector = np.zeros(J + K)

    # Initial residuals (without gamma offsets)
    residuals = y_train_target_residuals.copy()

    for iteration in range(max_iter):
        weights = huber_weights(residuals, delta)
        
        # Weighted least squares solution
        W = np.diag(weights)
        # Solve (X^T W X) gamma = X^T W y
        XTWX = X.T @ W @ X
        XTWy = X.T @ W @ y_train_target_residuals

        # Use least squares solver in case XTWX is singular
        gamma_new = np.linalg.lstsq(XTWX, XTWy, rcond=None)[0]

        # Update residuals
        residuals_new = y_train_target_residuals - X @ gamma_new

        # Check convergence
        if np.linalg.norm(gamma_new - gamma_vector) < tol:
            break
        
        gamma_vector = gamma_new
        residuals = residuals_new

    leaf_gamma = gamma_vector[:J]
    leaf_gammahat = gamma_vector[J:]

    return leaf_gamma, leaf_gammahat
