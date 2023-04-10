import numpy as np
from scipy.linalg import logm, sqrtm
import torch
import torch.nn as nn

class JSD(nn.Module):
    def __init__(self, num_samples=10 ** 3) -> None:
        super(JSD, self).__init__()
        self.num_samples = num_samples
    
    def sample_normal(self, mu, sigma):
        # if not hasattr(self, 'mvn'):
        #     self.mvn = torch.distributions.MultivariateNormal(torch.zeros_like(mu, device=mu.device, requires_grad=False), 
        #                                                       torch.eye(mu.shape[-1], device=mu.device, requires_grad=False))
        # standard_norm = self.mvn.rsample((self.num_samples,))
        # samples = mu + sigma * standard_norm
        univ_norm_dists = torch.distributions.normal.Normal(mu, scale=sigma)
        pred_stochastic = univ_norm_dists.rsample((self.num_samples,))
        return pred_stochastic
    
    def forward(self, mu_1: torch.tensor, sigma_1: torch.tensor, mu_2: torch.tensor, sigma_2: torch.tensor):
        if sigma_1.shape != sigma_2.shape:
            raise ValueError('Sigma shapes are not equal')

        # P_samples = self.sample_normal(mu_1, sigma_1)
        # Q_samples = self.sample_normal(mu_2, sigma_2)
        # breakpoint()
        P = torch.distributions.normal.Normal(mu_1, scale=sigma_1)
        Q = torch.distributions.normal.Normal(mu_2, scale=sigma_2)
        P_samples = P.rsample((self.num_samples,))
        Q_samples = Q.rsample((self.num_samples,))

        # if len(sigma_1.shape) == 1:
        #     sigma_1 = torch.diag(sigma_1)
        #     sigma_2 = torch.diag(sigma_2)

        # P = torch.distributions.MultivariateNormal(mu_1, sigma_1)
        # Q = torch.distributions.MultivariateNormal(mu_2, sigma_2)

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
    
class KLDiv(nn.Module):
    """Calculate KL divergence between two Gaussian distributions with diagonal covariance matrices.
    """
    def __init__(self, mode="diagonal") -> None:
        super(KLDiv, self).__init__()
        self.mode = mode
        
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        """Calculate KL divergence between two Gaussian distributions with full covariance matrices.
        Args:
            mu_p (torch.tensor): mean vector of distribution P
            sigma_p (torch.tensor): log of covariance matrix of distribution P
            mu_q (torch.tensor): mean vector of distribution Q
            sigma_q (torch.tensor): log of covariance matrix of distribution Q

        Returns:
            float: KL Divergence scalar
        """
        if sigma_p.ndim != 3:
            raise ValueError("Covariance matrix must be 3-dimensional in full mode (batch, dim, dim)")

        k = mu_p.shape[-1]

        #? Calculate the inverse of the measurement covariance matrix
        inv_sigma_q = np.linalg.inv(sigma_q)
        
        #? Compute the residual
        residual = np.expand_dims((mu_p - mu_q),axis=-1)
        
        #? Compute KL divergence
        trace = np.trace(inv_sigma_q @ sigma_p, axis1=1, axis2=2)
        maha = (np.transpose(residual, (0,2,1)) @ inv_sigma_q @ residual).squeeze()
        _, logdet_p = np.linalg.slogdet(sigma_p)
        _, logdet_q = np.linalg.slogdet(sigma_q)
        kl = 0.5 * (trace + maha - k + logdet_q - logdet_p)
        
        return kl
    
    def forward_diag(self, mu_p: torch.tensor, 
                    log_sigma_p: torch.tensor, 
                    mu_q: torch.tensor, 
                    log_sigma_q: torch.tensor):
        """Calculate KL divergence between two Gaussian distributions with diagonal covariance matrices.
        Args:
            mu_p (torch.tensor): mean vector of distribution P
            log_sigma_p (torch.tensor): log of diagonal covariance matrix of distribution P
            mu_q (torch.tensor): mean vector of distribution Q
            log_sigma_q (torch.tensor): log of diagonal covariance matrix of distribution Q

        Returns:
            float: KL Divergence scalar
        """
        #? Check if covariance diagonal
        if len(log_sigma_p.shape) != 1:
            raise NotImplementedError('KL divergence for non-diagonal covariance matrices is not implemented')

        log_p_q = (log_sigma_q - log_sigma_p).sum()
        k = mu_p.shape[-1]
        sigma_q_inv = torch.exp(-log_sigma_q)
        maha_dist = ((mu_p - mu_q) * sigma_q_inv) @ (mu_p - mu_q)
        trace = torch.exp(-log_sigma_q + log_sigma_p).sum()
        kldiv = 0.5 * (log_p_q + maha_dist - k + trace)
        return kldiv
    
    def forward(self, mu_p: torch.tensor, 
                sigma_p: torch.tensor, 
                mu_q: torch.tensor, 
                sigma_q: torch.tensor):
        shape_conditions = [sigma_p.shape == sigma_q.shape, mu_p.shape == mu_q.shape]
        if not all(shape_conditions):
            raise ValueError('Input shapes are not equal')
        if self.mode == "diagonal":
            return self.forward_diag(mu_p, sigma_p, mu_q, sigma_q)
        elif self.mode == "full":
            return self.forward_full(mu_p, sigma_p, mu_q, sigma_q)
        else:
            raise ValueError('Mode must be "diagonal" or "full"', self.mode)
    
class JRDiv(nn.Module):
    """Calculate Jeffrey-Riemannian divergence between two Gaussian distributions with covariance matrices.
    """
    def __init__(self, beta=0.85, mode="diagonal") -> None:
        super(JRDiv, self).__init__()
        self.beta = beta
        self.mode = mode
        
    def forward_full(self, mu_p, sigma_p, mu_q, sigma_q):
        """Calculate Jeffrey-Riemannian divergence between two Gaussian distributions with full covariance matrices.
        Args:
            mu_p (torch.tensor): mean vector of distribution P
            log_sigma_p (torch.tensor): covariance matrix of distribution P
            mu_q (torch.tensor): mean vector of distribution Q
            log_sigma_q (torch.tensor): covariance matrix of distribution Q

        Returns:
            float: JR Divergence scalar
        """
        if sigma_p.ndim != 3:
            raise ValueError("Covariance matrix must be 3-dimensional in full mode (batch, dim, dim)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        mu_p = torch.from_numpy(mu_p).to(device)
        mu_q = torch.from_numpy(mu_q).to(device)
        sigma_p = torch.from_numpy(sigma_p).to(device)
        sigma_q = torch.from_numpy(sigma_q).to(device)
        
        log_sigma_p = torch.log(torch.diagonal(sigma_p, dim1=1, dim2=2))
        log_sigma_q = torch.log(torch.diagonal(sigma_q, dim1=1, dim2=2))
        sigma_p_inv = torch.exp(-log_sigma_p)
        sigma_q_inv = torch.exp(-log_sigma_q)
        
        #? Compute the residual
        residual = mu_q - mu_p
        
        #? Compute the first term in JR divergence
        maha_dist = torch.einsum('bi, bi -> b', residual * (sigma_p_inv + sigma_q_inv), residual)
        maha_dist = maha_dist.clip(min=1e-12).sqrt().cpu().numpy()
        
        #? Compute the second term in JR divergence
        riemann_dist = torch.norm(-log_sigma_p + log_sigma_q).cpu().numpy()
        
        #? Compute JR divergence
        jr = (1 - self.beta) * maha_dist + self.beta * riemann_dist
        return jr
    
    def forward_diag(self, mu_p: torch.tensor,
                    log_sigma_p: torch.tensor, 
                    mu_q: torch.tensor, 
                    log_sigma_q: torch.tensor):
        """Calculate Jeffrey-Riemannian divergence between two Gaussian distributions with diagonal covariance matrices.
        Args:
            mu_p (torch.tensor): mean vector of distribution P
            log_sigma_p (torch.tensor): log of diagonal covariance matrix of distribution P
            mu_q (torch.tensor): mean vector of distribution Q
            log_sigma_q (torch.tensor): log of diagonal covariance matrix of distribution Q

        Returns:
            float: JR Divergence scalar
        """
        #? Check if covariance diagonal
        if len(log_sigma_p.shape) != 1:
            raise NotImplementedError('JR divergence for non-diagonal covariance matrices is not implemented')

        sigma_p_inv = torch.exp(-log_sigma_p)
        sigma_q_inv = torch.exp(-log_sigma_q)
        maha_dist = ((mu_q - mu_p) * (sigma_p_inv + sigma_q_inv)) @ (mu_q - mu_p)
        maha_dist = maha_dist.clamp(min=1e-12).sqrt()
        riemann_dist = torch.norm(-log_sigma_p + log_sigma_q)
        
        return (1 - self.beta) * maha_dist + self.beta * riemann_dist
    
    def forward(self, mu_p: torch.tensor, 
                sigma_p: torch.tensor, 
                mu_q: torch.tensor, 
                sigma_q: torch.tensor):
        shape_conditions = [sigma_p.shape == sigma_q.shape, mu_p.shape == mu_q.shape]
        if not all(shape_conditions):
            raise ValueError('Input shapes are not equal')
        
        if self.mode == "diagonal":
            return self.forward_diag(mu_p, sigma_p, mu_q, sigma_q)
        elif self.mode == "full":
            return self.forward_full(mu_p, sigma_p, mu_q, sigma_q)
        else:
            raise ValueError('Mode must be "diagonal" or "full"', self.mode)
        
class Mahalanobis(nn.Module):
    """
    Calculates the squared Malahanobis distance between a distribution and a set of measurements.
    Note: implemented in numpy for now, because of the cholesky decomposition.
    """
    def __init__(self) -> None:
        super(Mahalanobis, self).__init__()
    
    def forward(self,
                mean,
                covariance,
                measurements,
                measurement_cov=None):
        
        cholesky_factors = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = np.linalg.solve(cholesky_factors.transpose(0, 2, 1), d[..., np.newaxis]).squeeze(-1)
        squared_mahas = np.sum(z * z, axis=-1)
        return squared_mahas