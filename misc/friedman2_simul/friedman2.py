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

def friedman2(n_samples, add_noise=False, noise_distribution='gaussian', random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    x0 = np.random.uniform(0,100,size=n_samples)
    x1 = np.random.uniform(40*np.pi,560*np.pi,size=n_samples)
    x2 = np.random.uniform(0,1,size=n_samples)
    x3 = np.random.uniform(1,11,size=n_samples)
    X = np.column_stack((x0, x1, x2, x3)) 
    y = (x0**2 + (x1*x2 - (1/(x1*x3)))**2)**(1/2) 

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


    return X, y

def friedman2_altered(n_samples, add_noise=False, noise_distribution='gaussian',
                      d = 1.0, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)


    #generate distrubance factors
    a = np.random.normal(1, 0.1*d, size = 1)
    b = np.random.normal(1, 0.1*d, size = 4)
    b0, b1, b2, b3 = b
    c = np.random.normal(0, 0.05*d, size = 4)
    c0, c1, c2, c3 = c
    #

    #feature scaling
    x0 = np.random.uniform(0,100,size=n_samples)* b0 + c0
    x1 = np.random.uniform(40*np.pi,560*np.pi,size=n_samples) * b1 + c1
    x2 = np.random.uniform(0,1,size=n_samples) * b2 + c2
    x3 = np.random.uniform(1,11,size=n_samples) * b3 + c3
    
    X = np.column_stack((x0, x1, x2, x3))
    #label scaling
    y = a*(x0**2 + (x1*x2 - (1/(x1*x3)))**2)**(1/2) 

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


    return X, y




