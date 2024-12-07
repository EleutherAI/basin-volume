from copy import deepcopy
from scipy.optimize import linear_sum_assignment
import jax.numpy as jnp


def block_diag(matrices):
    """
    Creates a block diagonal matrix from a list of matrices.

    Parameters:
    - matrices: list of 2D arrays. Each element is a matrix (can be different sizes).

    Returns:
    - A block diagonal matrix combining all the input matrices.
    """
    # Determine the total size of the block diagonal matrix
    shapes = [m.shape for m in matrices]
    total_rows = sum(shape[0] for shape in shapes)
    total_cols = sum(shape[1] for shape in shapes)
    
    # Initialize the block diagonal matrix with zeros
    block_diag_matrix = jnp.zeros((total_rows, total_cols))
    
    # Populate the block diagonal matrix with the input matrices
    current_row, current_col = 0, 0
    for matrix in matrices:
        rows, cols = matrix.shape
        block_diag_matrix = block_diag_matrix.at[current_row:current_row + rows, current_col:current_col + cols].set(matrix)
        current_row += rows
        current_col += cols
    
    return block_diag_matrix


def canonicalize_dense(kernel1, kernel2, bias1):
    """Canonicalize adjacent layers of a ReLU network"""
    # Compute norms of the first layer weights
    norms = jnp.linalg.norm(kernel1, axis=0)
    kernel1 = kernel1 / norms

    # Adjust outgoing weights of the next layer
    kernel2 = kernel2 * norms[:, None]
    return kernel1, kernel2, bias1


def align_networks(
    src,
    target,
    *,
    canonicalize: bool = False,
    include_bias: bool = False,
    misalign: bool = False,
    symmetric: bool = False,
):
    # Don't modify the original parameters
    src = deepcopy(src)
    target = deepcopy(target)

    tgt_bias0 = target['params']['Dense_0']['bias']
    tgt_kernel0 = target['params']['Dense_0']['kernel']
    tgt_kernel1 = target['params']['Dense_1']['kernel']

    src_bias0 = src['params']['Dense_0']['bias']
    src_kernel0 = src['params']['Dense_0']['kernel']
    src_kernel1 = src['params']['Dense_1']['kernel']

    # Account for scaling symmetry in ReLU networks
    if canonicalize:
        src_kernel0, src_kernel1, src_bias0 = canonicalize_dense(
            src_kernel0, src_kernel1, src_bias0
        )
        tgt_kernel0, tgt_kernel1, tgt_bias0 = canonicalize_dense(
            tgt_kernel0, tgt_kernel1, tgt_bias0
        )

        # Modify target here since we won't permute it
        target['params']['Dense_0']['bias'] = tgt_bias0
        target['params']['Dense_0']['kernel'] = tgt_kernel0
        target['params']['Dense_1']['kernel'] = tgt_kernel1

    m, n = tgt_kernel0.shape
    assert tgt_kernel0.shape == src_kernel0.shape

    # Use the Hungarian algorithm to find the permutation that
    # maximizes the sum of the dot products
    cost = src_kernel0.T @ tgt_kernel0
    if include_bias:
        cost += src_bias0[:, None] @ tgt_bias0[None, :]
    if symmetric:
        cost += src_kernel1 @ tgt_kernel1.T

    rows, cols = linear_sum_assignment(cost, maximize=not misalign)

    # convert to permutation matrix
    P = jnp.zeros((n, n)).at[rows, cols].set(1)

    # Modify the source parameters
    src['params']['Dense_0']['kernel'] = src_kernel0 @ P
    src['params']['Dense_0']['bias'] = src_bias0 @ P

    # Permute the weights of the second layer in the inverse order
    src['params']['Dense_1']['kernel'] = P.T @ src_kernel1

    m, n = 64, 10
    I_m, I_n = jnp.eye(m), jnp.eye(n)

    P_k0 = jnp.kron(I_m, P)
    P_k1 = jnp.kron(P, I_n)
    P_ravel = block_diag([P, P_k0, jnp.eye(10), P_k1])
    return src, target, P_ravel