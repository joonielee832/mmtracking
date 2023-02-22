import torch
import torch.nn as nn

class JSD(nn.Module):
    def __init__(self, num_samples=10 ** 3) -> None:
        super(JSD, self).__init__()
        self.num_samples = num_samples
    
    def sample_normal(self, mu, sigma):
        if not hasattr(self, 'mvn'):
            self.mvn = torch.distributions.MultivariateNormal(torch.zeros_like(mu, device=mu.device, requires_grad=False), 
                                                              torch.eye(mu.shape[-1], device=mu.device, requires_grad=False))
        standard_norm = self.mvn.rsample((self.num_samples,))
        samples = mu + sigma * standard_norm
        return samples
    
    def forward(self, mu_1: torch.tensor, sigma_1: torch.tensor, mu_2: torch.tensor, sigma_2: torch.tensor):
        if sigma_1.shape != sigma_2.shape:
            raise ValueError('Sigma shapes are not equal')

        P_samples = self.sample_normal(mu_1, sigma_1)
        Q_samples = self.sample_normal(mu_2, sigma_2)

        if len(sigma_1.shape) == 1:
            sigma_1 = torch.diag(sigma_1)
            sigma_2 = torch.diag(sigma_2)

        P = torch.distributions.MultivariateNormal(mu_1, sigma_1)
        Q = torch.distributions.MultivariateNormal(mu_2, sigma_2)

        P_log_prob_P = P.log_prob(P_samples)
        P_log_prob_Q = P.log_prob(Q_samples)
        Q_log_prob_P = Q.log_prob(P_samples)
        Q_log_prob_Q = Q.log_prob(Q_samples)

        log_mix_X = torch.logaddexp(P_log_prob_P, Q_log_prob_P)
        log_mix_Y = torch.logaddexp(P_log_prob_Q, Q_log_prob_Q)

        # Optimize the mean computation using `einsum`
        num_samples = P_samples.shape[0]
        jsd_einsum = ((torch.einsum('i...->', P_log_prob_P)/num_samples) - ((torch.einsum('i...->', log_mix_X)/num_samples) - torch.log(torch.tensor(2.0)))
               + (torch.einsum('i...->', Q_log_prob_Q)/num_samples) - ((torch.einsum('i...->', log_mix_Y)/num_samples) - torch.log(torch.tensor(2.0)))) / 2

        # Optimize the return statement using `clamp`
        return torch.sqrt(torch.clamp(jsd_einsum, min=1E-5))