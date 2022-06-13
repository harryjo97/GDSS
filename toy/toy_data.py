import numpy as np


# Mixture of 4 multivariate normal distribution
def data_distribution(mus, covs, batch_size):

    k = np.random.randint(0, len(mus), size=batch_size)
    for i in range(len(mus)):
        if i==0:
            z = np.random.multivariate_normal(mus[i], covs[i], size=batch_size)
        else:
            z = np.stack([z, np.random.multivariate_normal(mus[i], covs[i], size=batch_size)])

    return z[k, np.arange(batch_size)]


if __name__=='__main__':
    from toy_loader import plot, load_seed

    load_seed(11)

    com1 = 0.5
    com2 = 0.5
    rho = 0.9
    mus = [[com1,com2],[-com1,-com2]]
    covs = np.array([[[1,rho],[rho,1]],
                    [[1,rho],[rho,1]],])*0.01
    batch_size = 2048 * 4
    p = data_distribution(mus, covs, batch_size)

    plot(p, 'data', 24)
