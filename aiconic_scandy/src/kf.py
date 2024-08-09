import torch


def update(mu, Sigma, z, H, R):
    innovation = z - H.mv(mu)

    S = torch.mm(torch.mm(H, Sigma), H.transpose(0, 1)) + R(mu)
    K = torch.mm(torch.mm(Sigma, H.transpose(0, 1)), torch.inverse(S))

    change = torch.mv(K, innovation)

    mu_new = mu + change
    Sigma_new = Sigma - torch.mm(K, torch.mm(H, Sigma))

    return mu_new, Sigma_new


def update_batch(mu, Sigma, z, H, R):

    innovation = z - H.bmm(mu)

    S = torch.bmm(torch.bmm(H, Sigma), H.transpose(1, 2)) + R(mu)
    K = torch.bmm(torch.bmm(Sigma, H.transpose(1, 2)), torch.inverse(S))

    # batch matrix vector multiplication
    change = torch.einsum('bij,bj->bi', K, innovation)

    mu_new = mu + change

    Sigma_new = Sigma - torch.bmm(K, torch.bmm(H, Sigma))

    return mu_new, Sigma_new


def predict(mu, Sigma, u, F_mu, F_u, Q):
    mu_new = F_mu.mv(mu) + F_u.mv(u)

    Sigma_new = F_mu.mm(Sigma).mm(F_mu.t()) + Q(mu, u)

    return mu_new, Sigma_new


def predict_batch(mu, Sigma, u, F_mu, F_u, Q):

    mu_new = F_mu.bmm(mu) + F_u.bmm(u)
    Sigma_new = F_mu.bmm(Sigma).bmm(F_mu.transpose(1,2)) + Q(mu, u)

    return mu_new, Sigma_new