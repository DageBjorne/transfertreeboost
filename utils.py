import numpy as np

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

