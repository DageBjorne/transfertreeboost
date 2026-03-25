import numpy as np

def gaussian_noise(n_samples, signal_y, snr=3.0, random_seed = None):
    """
    Generate Gaussian noise with variance scaled to achieve the desired SNR
    relative to the variance of signal_y.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    signal_var = np.var(signal_y)
    target_noise_var = signal_var / snr
    noise = np.random.normal(0, 1, size=n_samples)
    noise_std = np.std(noise)
    noise *= np.sqrt(target_noise_var) / noise_std
    return noise


def slash_noise(n_samples, signal_y, snr=3.0, random_seed = None):
    """
    Generate Slash-distributed noise (Z/U) scaled to achieve desired SNR.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    signal_var = np.var(signal_y)
    target_noise_var = signal_var / snr

    u = np.random.normal(0, 1, size=n_samples)
    v = np.random.uniform(0.001, 1, size=n_samples)
    raw_noise = u / v

    # Scale to achieve target variance
    noise_std = np.std(raw_noise)
    scaled_noise = raw_noise * np.sqrt(target_noise_var) / noise_std
    return scaled_noise

def friedman1(n_samples, add_noise=False, noise_distribution='gaussian', n_features=5, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    x0 = np.random.uniform(0,1,size=n_samples)
    x1 = np.random.uniform(0,1,size=n_samples)
    x2 = np.random.uniform(0,1,size=n_samples)
    x3 = np.random.uniform(0,1,size=n_samples)
    x4 = np.random.uniform(0,1,size=n_samples)
    X = np.column_stack((x0, x1, x2, x3, x4)) 
    y = 10*np.sin(x0*x1) + 20*(x2 - 0.5)**2 + 10*x3 + 5*x4

    if add_noise:
        if noise_distribution == 'gaussian':
            if random_seed is not None:
                eps = gaussian_noise(n_samples, y, random_seed = random_seed)
            else:
                eps = gaussian_noise(n_samples, y)
            y += eps
        elif noise_distribution == 'slash':
            if random_seed is not None:
                eps = slash_noise(n_samples, y, random_seed = random_seed)
            else:
                eps = slash_noise(n_samples, y)
            
            y += eps
        else:
            raise Exception("No valid distribution, only gaussian or slash are accepted")

    if n_features > 5:
        noise_features = np.random.uniform(0, 1, size=(n_samples, n_features - 5))
        X = np.hstack((X, noise_features))

    return X, y


def friedman1_altered(n_samples, add_noise=False, noise_distribution='gaussian', n_features=5,
                      d = 1.0,
                      shift_seed=None, random_seed=None):

    if random_seed is not None:
        np.random.seed(random_seed)
    
    #generate distrubance factors
    a = np.random.normal(1, 0.1*d, size = 4)
    a0, a1, a2, a3 = a
    b = np.random.normal(1, 0.1*d, size = 5)
    b0, b1, b2, b3, b4 = b
    c = np.random.normal(0, 0.05*d, size = 5)
    c0, c1, c2, c3, c4 = c
    #

    #feature scaling
    x0 = np.random.uniform(0,1,size=n_samples) * b0 + c0
    x1 = np.random.uniform(0,1,size=n_samples) * b1 + c1
    x2 = np.random.uniform(0,1,size=n_samples) * b2 + c2
    x3 = np.random.uniform(0,1,size=n_samples) * b3 + c3
    x4 = np.random.uniform(0,1,size=n_samples) * b4 + c4
    

    X = np.column_stack((x0, x1, x2, x3, x4))
    #label scaling
    y = a0*10*np.sin(x0*x1) + a1*20*(x2 - 0.5)**2 + a2*10*x3 + a3*5*x4

    if add_noise:
        if noise_distribution == 'gaussian':
            eps = gaussian_noise(n_samples, y)
            y += eps
        elif noise_distribution == 'slash':
            eps = slash_noise(n_samples, y)
            y += eps
        else:
            raise Exception("No valid distribution, only gaussian or slash are accepted")

    if n_features > 5:
        noise_features = np.random.uniform(0, 1, size=(n_samples, n_features - 5))
        X = np.hstack((X, noise_features))

    return X, y




