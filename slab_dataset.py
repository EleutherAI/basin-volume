import numpy as np


def get_linear_data(samples):
    X2 = np.linspace(1, -1, samples)
    X1 = np.zeros(len(X2))

    shift = 0.1
    X_linear_c0 = np.concatenate([X1[:, None]+shift, X2[:, None]], axis=1)
    X_linear_c1 = np.concatenate([X1[:, None]-shift, X2[:, None]], axis=1)
    X_linear = np.concatenate([X_linear_c0, X_linear_c1], axis=0)
    Y_linear = np.concatenate([np.zeros(len(X1)), np.ones(len(X1))])
    return X_linear, Y_linear


def get_nonlinear_data(samples, num_slabs=4):
    # 4 slab dataset
    complex_margin = 0.1
    linear_margin = 0.1
    slab_thickness = (2-(num_slabs-1)*complex_margin*2)/num_slabs
    y_intersections = []
    y_intersection = -1+slab_thickness+complex_margin
    y_start = -1 + slab_thickness/2
    a_sign = (-1) ** num_slabs
    X = []
    for i in range(num_slabs-1):
        y_intersections.append(y_intersection)
        y_end = y_start + complex_margin * 2 + slab_thickness
        y_cur = np.linspace(y_start, y_end, 10)
        x_cur = (y_cur-y_intersection)*a_sign
        X_cur = np.concatenate([x_cur[:, None], y_cur[:, None]], axis=1)
        X.append(X_cur)
        y_intersection += complex_margin*2 + slab_thickness
        y_start = y_end
        a_sign *= -1
    X = np.concatenate(X, axis=0)
    shift = complex_margin * (2**0.5)
    X_nonlinear = np.concatenate([
        X + np.array([[shift, 0]]),
        X - np.array([[shift, 0]])], axis=0)
    Y_nonlinear = np.concatenate([
        np.zeros(len(X)),
        np.ones(len(X))
    ])

    return X_nonlinear, Y_nonlinear


def get_slab_data(num_slabs=4, seed: int = 0):
    # 4 slab dataset
    complex_margin = 0.1
    linear_margin = 0.1
    slab_thickness = (2-(num_slabs-1)*complex_margin*2)/num_slabs
    slab_end = 1
    slab_begin = 1-slab_thickness
    slab_sign = -1
    slab_data = []
    slab_labels = []
    
    rng = np.random.default_rng(seed)  # Initialize default RNG
    for i in range(num_slabs):
        if slab_sign == -1:
            slab_current = np.concatenate([
                rng.uniform(-1, 0-linear_margin, size=(400, 1)), 
                rng.uniform(slab_begin, slab_end, size=(400, 1))], axis=1)
            slab_labels.append(np.ones(len(slab_current)))
        else:
            slab_current = np.concatenate([
                rng.uniform(0+linear_margin, 1, size=(400, 1)), 
                rng.uniform(slab_begin, slab_end, size=(400, 1))], axis=1)
            slab_labels.append(np.zeros(len(slab_current)))
        slab_data.append(slab_current)
        slab_begin -= slab_thickness + complex_margin * 2
        slab_end -= slab_thickness + complex_margin * 2
        slab_sign *= -1

    slab_data = np.concatenate(slab_data, axis=0)
    slab_labels = np.concatenate(slab_labels)
    return slab_data, slab_labels
