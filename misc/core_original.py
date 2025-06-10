from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from utils import *

class LSTransferTreeBoost():
    def __init__(self, v=0.1, epochs=100, target_tree_size=2, source_tree_size=2,
                 alpha_0=1.0, decay_factor=0.99, min_samples_leaf=25):
        # Save hyperparameters
        self.v = v
        self.epochs = epochs
        self.target_tree_size = target_tree_size
        self.source_tree_size = source_tree_size
        self.alpha_0 = alpha_0
        self.decay_factor = decay_factor
        self.min_samples_leaf = min_samples_leaf
        self.initial_guess = None
        self.x_train_target_snapshot = None #to use in predict

        # Initialize containers to be set during fit
        self.model_tray_clf = []
        self.model_tray_clfhat = []
        self.leaf_gammas_tray = []
        self.leaf_gammashats_tray = []
        self.alpha_tray = []

    def predict(self, x_test):
        F_pred = np.full_like(
            x_test[:, 0], self.initial_guess)  # Initialize with training mean

        # Iterate over each model (clf, clfhat) in the model tray
        for i, clf in enumerate(self.model_tray_clf):
            clfhat = self.model_tray_clfhat[i]
            alpha = self.alpha_tray[i]
            v = self.v

            # Get leaf indices for train and test
            leaves_clf_train = clf.apply(self.x_train_target_snapshot)
            leaves_clfhat_train = clfhat.apply(self.x_train_target_snapshot)
            leaves_clf_test = clf.apply(x_test)
            leaves_clfhat_test = clfhat.apply(x_test)

            leaves_clf_train_test = np.concatenate(
                (leaves_clf_train, leaves_clf_test))
            leaves_clfhat_train_test = np.concatenate(
                (leaves_clfhat_train, leaves_clfhat_test))

            indexed_leaves_clf_train_test = map_leaves_to_number(
                leaves_clf_train_test)
            indexed_leaves_clfhat_train_test = map_leaves_to_number(
                leaves_clfhat_train_test)

            # Map test datapoints to correct leaves
            indexed_leaves_clf_test = indexed_leaves_clf_train_test[len(leaves_clf_train):]
            indexed_leaves_clfhat_test = indexed_leaves_clfhat_train_test[len(leaves_clfhat_train):]

            # Retrieve gamma values
            leaf_gamma = self.leaf_gammas_tray[i]
            leaf_gammahat = self.leaf_gammashats_tray[i]

            for index in np.unique(indexed_leaves_clf_test):
                F_pred[indexed_leaves_clf_test == index] += v * leaf_gamma[index] * (1 - alpha)

            for index in np.unique(indexed_leaves_clfhat_test):
                F_pred[indexed_leaves_clfhat_test == index] += v * leaf_gammahat[index] * alpha

        return F_pred
    
    def evaluate(self, x_test, y_test, metric = 'rmse'):
        preds = self.predict(x_test)
        if metric == 'rmse':
            return compute_rmse(preds, y_test)
        if metric == 'mse':
            return compute_mse(preds, y_test)
        if metric == 'mae':
            return compute_mae(preds, y_test)
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def fit(self, x_train_target, y_train_target, x_train_source, y_train_source, 
            show_curves=False, val_x = None, val_y = None, early_stopping_rounds = 5):
        self.x_train_target_snapshot = x_train_target
        all_X = np.concatenate((x_train_target, x_train_source))
        self.initial_guess = np.mean(y_train_target)
        F = np.full(all_X.shape[0], self.initial_guess)

        alpha = self.alpha_0
        self.model_tray_clf = []
        self.model_tray_clfhat = []
        self.leaf_gammas_tray = []
        self.leaf_gammashats_tray = []
        self.alpha_tray = []
        if show_curves:
            losses = []
        if val_x is not None and val_y is not None:
            val_losses = []

        target_indices = np.arange(len(y_train_target))
        source_indices = np.arange(len(y_train_target), len(y_train_target) + len(y_train_source))

        for m in range(self.epochs):
            y_train_target_residuals = y_train_target - F[target_indices]
            y_train_source_residuals = y_train_source - F[source_indices]

            clf = DecisionTreeRegressor(max_depth=self.target_tree_size, min_samples_leaf=self.min_samples_leaf)
            clf.fit(x_train_target, y_train_target_residuals)

            clfhat = DecisionTreeRegressor(max_depth=self.source_tree_size, min_samples_leaf=self.min_samples_leaf)
            clfhat.fit(x_train_source, y_train_source_residuals)

            self.model_tray_clf.append(clf)
            self.model_tray_clfhat.append(clfhat)

            leaves_clf_target = clf.apply(x_train_target)
            leaves_clfhat_target = clfhat.apply(x_train_target)
            leaves_clf_source = clf.apply(x_train_source)
            leaves_clfhat_source = clfhat.apply(x_train_source)

            all_leaves_clf = np.concatenate([leaves_clf_target, leaves_clf_source])
            all_leaves_clfhat = np.concatenate([leaves_clfhat_target, leaves_clfhat_source])

            unique_leaves_clf = np.unique(all_leaves_clf)
            unique_leaves_clfhat = np.unique(all_leaves_clfhat)

            indexed_leaves_clf_train_test = map_leaves_to_number(all_leaves_clf)
            indexed_leaves_clfhat_train_test = map_leaves_to_number(all_leaves_clfhat)

            indexed_leaves_clf = indexed_leaves_clf_train_test[:len(leaves_clf_target)]
            indexed_leaves_clfhat = indexed_leaves_clfhat_train_test[:len(leaves_clfhat_target)]

            leaf_gamma, leaf_gammahat = find_gamma_gammahat(unique_leaves_clf, indexed_leaves_clf, 
                        unique_leaves_clfhat, indexed_leaves_clfhat,
                        y_train_target_residuals)

            self.leaf_gammas_tray.append(leaf_gamma)
            self.leaf_gammashats_tray.append(leaf_gammahat)

            alpha *= self.decay_factor
            self.alpha_tray.append(alpha)

            indexed_leaves_clf = map_leaves_to_number(all_leaves_clf)
            indexed_leaves_clfhat = map_leaves_to_number(all_leaves_clfhat)

            for index in np.unique(indexed_leaves_clf):
                F[indexed_leaves_clf == index] += self.v * leaf_gamma[index] * (1 - alpha)
            for index in np.unique(indexed_leaves_clfhat):
                F[indexed_leaves_clfhat == index] += self.v * leaf_gammahat[index] * alpha

            if show_curves:
                losses.append(compute_mse(F[target_indices], y_train_target))

            if val_x is not None and val_y is not None:
                val_mse = self.evaluate(val_x, val_y, metric = 'mse')
                val_losses.append(val_mse)
                es = early_stopping(early_stopping_rounds, val_losses, tol = 1e-6)
                
                if es:
                    break

        if show_curves:
            x = np.linspace(1, m+1, m+1)
            plt.plot(x, np.array(losses))
            if val_x is not None and val_y is not None:
                plt.plot(x, np.array(val_losses))
                plt.legend(['train_loss', 'val_loss'])

            plt.title('Loss over epochs')
            plt.xlabel('epoch')
            plt.ylabel('MSE')
            plt.show()


        return self.leaf_gammas_tray, self.leaf_gammashats_tray, self.model_tray_clf, self.model_tray_clfhat, self.alpha_tray

class LADTransferTreeBoost():
    def __init__(self, v=0.1, epochs=100, target_tree_size=2, source_tree_size=2,
                 alpha_0=1.0, decay_factor=0.99, min_samples_leaf=25):
        # Save hyperparameters
        self.v = v
        self.epochs = epochs
        self.target_tree_size = target_tree_size
        self.source_tree_size = source_tree_size
        self.alpha_0 = alpha_0
        self.decay_factor = decay_factor
        self.min_samples_leaf = min_samples_leaf

        self.initial_guess = None
        self.x_train_target_snapshot = None #to use in predict


        # Initialize containers to be set during fit
        self.model_tray_clf = []
        self.model_tray_clfhat = []
        self.leaf_gammas_tray = []
        self.leaf_gammashats_tray = []
        self.alpha_tray = []

    def predict(self, x_test):
        F_pred = np.full_like(
            x_test[:, 0], self.initial_guess)  # Initialize with training mean

        # Iterate over each model (clf, clfhat) in the model tray
        for i, clf in enumerate(self.model_tray_clf):
            clfhat = self.model_tray_clfhat[i]
            alpha = self.alpha_tray[i]
            v = self.v

            # Get leaf indices for train and test
            leaves_clf_train = clf.apply(self.x_train_target_snapshot)
            leaves_clfhat_train = clfhat.apply(self.x_train_target_snapshot)
            leaves_clf_test = clf.apply(x_test)
            leaves_clfhat_test = clfhat.apply(x_test)

            leaves_clf_train_test = np.concatenate(
                (leaves_clf_train, leaves_clf_test))
            leaves_clfhat_train_test = np.concatenate(
                (leaves_clfhat_train, leaves_clfhat_test))

            indexed_leaves_clf_train_test = map_leaves_to_number(
                leaves_clf_train_test)
            indexed_leaves_clfhat_train_test = map_leaves_to_number(
                leaves_clfhat_train_test)

            # Map test datapoints to correct leaves
            indexed_leaves_clf_test = indexed_leaves_clf_train_test[len(leaves_clf_train):]
            indexed_leaves_clfhat_test = indexed_leaves_clfhat_train_test[len(leaves_clfhat_train):]

            # Retrieve gamma values
            leaf_gamma = self.leaf_gammas_tray[i]
            leaf_gammahat = self.leaf_gammashats_tray[i]

            for index in np.unique(indexed_leaves_clf_test):
                F_pred[indexed_leaves_clf_test == index] += v * leaf_gamma[index] * (1 - alpha)

            for index in np.unique(indexed_leaves_clfhat_test):
                F_pred[indexed_leaves_clfhat_test == index] += v * leaf_gammahat[index] * alpha

        return F_pred
    
    def evaluate(self, x_test, y_test, metric = 'rmse'):
        preds = self.predict(x_test)
        if metric == 'rmse':
            return compute_rmse(preds, y_test)
        if metric == 'mse':
            return compute_mse(preds, y_test)
        if metric == 'mae':
            return compute_mae(preds, y_test)
        else:
            raise ValueError(f"Unsupported metric: {metric}")


    def fit(self, x_train_target, y_train_target, x_train_source, y_train_source, 
            show_curves=False, val_x = None, val_y = None, early_stopping_rounds = 5):
        self.x_train_target_snapshot = x_train_target
        all_X = np.concatenate((x_train_target, x_train_source))
        self.initial_guess = np.median(y_train_target)
        F = np.full(all_X.shape[0], self.initial_guess)

        alpha = self.alpha_0
        self.model_tray_clf = []
        self.model_tray_clfhat = []
        self.leaf_gammas_tray = []
        self.leaf_gammashats_tray = []
        self.alpha_tray = []
        if show_curves:
            losses = []
        if val_x is not None and val_y is not None:
            val_losses = []

        target_indices = np.arange(len(y_train_target))
        source_indices = np.arange(len(y_train_target), len(y_train_target) + len(y_train_source))

        for m in range(self.epochs):
            y_train_target_residuals = y_train_target - F[target_indices]
            y_train_source_residuals = y_train_source - F[source_indices]

            clf = DecisionTreeRegressor(max_depth=self.target_tree_size, min_samples_leaf=self.min_samples_leaf)
            clf.fit(x_train_target, y_train_target_residuals)

            clfhat = DecisionTreeRegressor(max_depth=self.source_tree_size, min_samples_leaf=self.min_samples_leaf)
            clfhat.fit(x_train_source, y_train_source_residuals)

            self.model_tray_clf.append(clf)
            self.model_tray_clfhat.append(clfhat)

            leaves_clf_target = clf.apply(x_train_target)
            leaves_clfhat_target = clfhat.apply(x_train_target)
            leaves_clf_source = clf.apply(x_train_source)
            leaves_clfhat_source = clfhat.apply(x_train_source)

            all_leaves_clf = np.concatenate([leaves_clf_target, leaves_clf_source])
            all_leaves_clfhat = np.concatenate([leaves_clfhat_target, leaves_clfhat_source])

            unique_leaves_clf = np.unique(all_leaves_clf)
            unique_leaves_clfhat = np.unique(all_leaves_clfhat)

            indexed_leaves_clf_train_test = map_leaves_to_number(all_leaves_clf)
            indexed_leaves_clfhat_train_test = map_leaves_to_number(all_leaves_clfhat)

            indexed_leaves_clf = indexed_leaves_clf_train_test[:len(leaves_clf_target)]
            indexed_leaves_clfhat = indexed_leaves_clfhat_train_test[:len(leaves_clfhat_target)]

            leaf_gamma, leaf_gammahat = find_gamma_gammahat_LAD(unique_leaves_clf, indexed_leaves_clf, 
                        unique_leaves_clfhat, indexed_leaves_clfhat,
                        y_train_target_residuals)

            self.leaf_gammas_tray.append(leaf_gamma)
            self.leaf_gammashats_tray.append(leaf_gammahat)

            alpha *= self.decay_factor
            self.alpha_tray.append(alpha)

            indexed_leaves_clf = map_leaves_to_number(all_leaves_clf)
            indexed_leaves_clfhat = map_leaves_to_number(all_leaves_clfhat)

            for index in np.unique(indexed_leaves_clf):
                F[indexed_leaves_clf == index] += self.v * leaf_gamma[index] * (1 - alpha)
            for index in np.unique(indexed_leaves_clfhat):
                F[indexed_leaves_clfhat == index] += self.v * leaf_gammahat[index] * alpha

            if show_curves:
                losses.append(compute_mae(F[target_indices], y_train_target))

            if val_x is not None and val_y is not None:
                val_lad = self.evaluate(val_x, val_y, metric = 'mae')
                val_losses.append(val_lad)
                es = early_stopping(early_stopping_rounds, val_losses, tol = 1e-6)
                
                if es:
                    break

        if show_curves:
            x = np.linspace(1, m+1, m+1)
            plt.plot(x, np.array(losses))
            if val_x is not None and val_y is not None:
                plt.plot(x, np.array(val_losses))
                plt.legend(['train_loss', 'val_loss'])

            plt.title('Loss over epochs')
            plt.xlabel('epoch')
            plt.ylabel('MAE')
            plt.show()


            



        return self.leaf_gammas_tray, self.leaf_gammashats_tray, self.model_tray_clf, self.model_tray_clfhat, self.alpha_tray

