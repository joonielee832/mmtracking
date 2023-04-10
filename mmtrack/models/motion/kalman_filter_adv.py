# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
import scipy.linalg

from mmtrack.core.utils import KLDiv, JRDiv, Mahalanobis
from ..builder import MOTION
from .kalman_filter import KalmanFilter

def full2diagonal(cov):
    """Convert full covariance matrix to diagonal covariance matrix.

    Args:
        cov (ndarray): The full covariance matrix (kxk dimensional).

    Returns:
        (ndarray): The diagonal covariance matrix (kxk dimensional).
    """
    cov = np.diag(np.diag(cov))
    return cov
@MOTION.register_module()
class KalmanFilterAdvanced(KalmanFilter):
    """A Kalman filter with measurement noise for tracking bounding boxes in image space.

    The implementation is referred to https://github.com/nwojke/deep_sort.
    """
    chi2inv95 = {
        1: 3.8415,
        2: 5.9915,
        3: 7.8147,
        4: 9.4877,
        5: 11.070,
        6: 12.592,
        7: 14.067,
        8: 15.507,
        9: 16.919
    }

    def __init__(self, mode="full", center_only=False, distance="mahalanobis", threshold=None, scale_gating=None):
        self.center_only = center_only
        if self.center_only:
            self.gating_threshold = self.chi2inv95[2]
        else:
            self.gating_threshold = self.chi2inv95[4]

        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        
        #* Use diagonal vs full covariance matrix for incoming measurements
        self.mode = mode

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160
        
        self.scale_gating = scale_gating
        self.distance = distance
        if distance == "KL":
            self.distance_fn = KLDiv(mode="full")
            if threshold is not None:
                self.gating_threshold = threshold
        elif distance == "JR":
            self.distance_fn = JRDiv(mode="full")
            if threshold is not None:
                self.gating_threshold = threshold
        else:
            self.distance_fn = Mahalanobis()

    def initiate(self, measurement, bbox_cov):
        """Create track from unassociated measurement.

        Args:
            measurement (ndarray):  Bounding box coordinates (x, y, a, h) with
            center position (x, y), aspect ratio a, and height h.
            measurement_covariance (ndarray): The 4x4 dimensional covariance

        Returns:
             (ndarray, ndarray): Returns the mean vector (8 dimensional) and
                covariance matrix (8x8 dimensional) of the new track.
                Unobserved velocities are initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std_vel = [
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3], 1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        vel_covariance = np.diag(np.square(std_vel))
        
        #? Convert bbox_cov to diagonal covariance matrix if mode is diagonal
        bbox_cov = full2diagonal(bbox_cov) if self.mode == "diagonal" else bbox_cov
        
        covariance = np.block([[bbox_cov, np.zeros_like(bbox_cov)],
                               [np.zeros_like(bbox_cov), vel_covariance]])
        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Args:
            mean (ndarray): The 8 dimensional mean vector of the object
                state at the previous time step.

            covariance (ndarray): The 8x8 dimensional covariance matrix
                of the object state at the previous time step.

        Returns:
            (ndarray, ndarray): Returns the mean vector and covariance
                matrix of the predicted state. Unobserved velocities are
                initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3], 1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3], 1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance, measurement_cov, gating=False):
        """Project state distribution to measurement space.

        Args:
            mean (ndarray): The state's mean vector (8 dimensional array).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            measurement_cov (ndarray): The measurement's covariance matrix (N, 4, 4)
            gating (bool): Whether function is used for gating or not. (default: False)

        Returns:
            (ndarray, ndarray):  Returns the projected mean and covariance
            matrix of the given state estimate.
        """
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T))
        
        if gating:
            mean = np.tile(mean, (measurement_cov.shape[0], 1))
            covariance = np.tile(covariance, (measurement_cov.shape[0], 1, 1))

        covariance += measurement_cov
        return mean, covariance

    def update(self, mean, covariance, measurement, measurement_cov):
        """Run Kalman filter correction step.

        Args:
            mean (ndarray): The predicted state's mean vector (8 dimensional).
            covariance (ndarray): The state's covariance matrix (8x8
                dimensional).
            measurement (ndarray): The 4 dimensional measurement vector
                (x, y, a, h), where (x, y) is the center position, a the
                aspect ratio, and h the height of the bounding box.


        Returns:
             (ndarray, ndarray): Returns the measurement-corrected state
             distribution.
        """
        #? Convert measurement_cov to diagonal covariance matrix if mode is diagonal
        measurement_cov = full2diagonal(measurement_cov) if self.mode == "diagonal" else measurement_cov
        projected_mean, projected_cov = self.project(mean, covariance, measurement_cov)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             np.dot(covariance,
                                                    self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self,
                        mean,
                        covariance,
                        measurements,
                        measurement_cov,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Args:
            mean (ndarray): Mean vector over the state distribution (8
                dimensional).
            covariance (ndarray): Covariance of the state distribution (8x8
                dimensional).
            measurements (ndarray): An Nx4 dimensional matrix of N
                measurements, each in format (x, y, a, h) where (x, y) is the
                bounding box center position, a the aspect ratio, and h the
                height.
            only_position (bool, optional): If True, distance computation is
                done with respect to the bounding box center position only.
                Defaults to False.

        Returns:
            ndarray: Returns an array of length N, where the i-th element
            contains the distance between (mean, covariance) and `measurements[i]`.
        """
        projected_means, projected_covs = self.project(mean, covariance, measurement_cov, gating=True)

        if only_position:
            # projected_means = projected_means[:, :2]
            # projected_covs = projected_covs[:, :2, :2]
            # measurements = measurements[:, :2]
            raise NotImplementedError

        distance = self.distance_fn(projected_means,
                                 projected_covs,
                                 measurements,
                                 measurement_cov)

        # if self.scale_gating is not None:
        #     if not isinstance(self.scale_gating, dict):
        #         raise TypeError("normalize must be a dict")
        #     if not hasattr(self.scale_gating, "type"):
        #         raise ValueError("normalize must have a type key")
        #     if self.scale_gating["type"] == "trace":
        #         # Normalize the covariance matrices by their trace
        #         normalized_projected_covs = projected_covs / np.trace(projected_covs, axis1=1, axis2=2)[:, np.newaxis, np.newaxis]

        #         # Calculate the average normalized covariance
        #         average_normalized_covariance = np.mean(normalized_projected_covs, axis=0)

        #         # Calculate the trace of the average normalized covariance
        #         trace_avg_normalized_cov = np.trace(average_normalized_covariance)
        #         normalization_factor = 1/(1+trace_avg_normalized_cov)
        #         squared_mahas = squared_mahas * normalization_factor
        #     elif self.scale_gating["type"] == "det":
        #         power = self.scale_gating.get("power", 0.10)
        #         projected_cov_det = np.linalg.det(projected_covs)
        #         squared_mahas = squared_mahas * np.power(projected_cov_det, power)
        return distance

    def track(self, tracks, bboxes, bbox_covs):
        """Track forward.

        Args:
            tracks (dict[int:dict]): Track buffer.
            bboxes (Tensor): Detected bounding boxes.

        Returns:
            (dict[int:dict], Tensor): Updated tracks and bboxes.
        """
        costs = []
        #? Convert bbox_covs to diagonal covariance matrices if mode is diagonal
        bbox_covs = torch.diag_embed(torch.diagonal(bbox_covs,dim1=1,dim2=2)) if self.mode == "diagonal" else bbox_covs
        
        for id, track in tracks.items():
            track.mean, track.covariance = self.predict(
                track.mean, track.covariance)
            gating_distance = self.gating_distance(track.mean,
                                                    track.covariance,
                                                    bboxes.cpu().numpy(),
                                                    bbox_covs.cpu().numpy(),
                                                    only_position=self.center_only)
            costs.append(gating_distance)

        costs = np.stack(costs, 0)
        costs[costs > self.gating_threshold] = np.nan
        return tracks, costs
