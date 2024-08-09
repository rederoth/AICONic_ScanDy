import matplotlib.pyplot as plt

from kf import predict, update
import torch
import numpy as np


def gaussian_2d_torch(x0, y0, xmax, ymax, xsig, ysig=None):
    """
    TODO: replace with torch.distributions.multivariate_normal.MultivariateNormal
    Function draws a 2D Gaussian on a frame with the dimensions of the video (xmax, ymax).
    x0, y0: center point of the Gaussian
    xsig: standard deviation of the Gaussian,
    ysig: default assumes to be the same as xsig
    """
    if not ysig:
        ysig = xsig
    X, Y = torch.meshgrid(torch.arange(0, xmax, dtype=torch.get_default_dtype()),
                          torch.arange(0, ymax, dtype=torch.get_default_dtype()), indexing="ij")
    G = torch.exp(-0.5 * (((X - x0) / xsig) ** 2 + ((Y - y0) / ysig) ** 2))
    return G


def rotated_atan2_2d_torch(x0, y0, xmax, ymax, angle_degrees, min_val=0, max_val=2, restricted_range=180):
    X, Y = torch.meshgrid(torch.arange(0, xmax, dtype=torch.get_default_dtype()),
                          torch.arange(0, ymax, dtype=torch.get_default_dtype()), indexing="ij")
    # Center to the current gaye position
    X = X - x0
    Y = Y - y0
    # Rotate the coordinates by the previous saccade angle
    # introduce minus since y-axis origin is in the top left corner
    angle_radians = - torch.deg2rad(torch.tensor(angle_degrees, dtype=torch.get_default_dtype()))
    # X_rotated = X * torch.cos(angle_radians) + Y * torch.sin(angle_radians)
    # Y_rotated = X * torch.sin(angle_radians) + Y * torch.cos(angle_radians)
    X_rotated = X * torch.sin(angle_radians) + Y * torch.cos(angle_radians)
    Y_rotated = X * torch.cos(angle_radians) - Y * torch.sin(angle_radians)
    angle = torch.arctan2(Y_rotated, X_rotated)
    angle = torch.clip(angle, -np.deg2rad(restricted_range), np.deg2rad(restricted_range))

    # Calculate the preference map assuming saccadic momentum
    angle_preference = 1 - torch.abs(torch.rad2deg(angle) / restricted_range)
    # the min_val parameter influences the strength of this effect
    # return angle_preference * (2 - 2 * min_val) + min_val
    # TODO find good way to make max val dependent on the angle restriction
    return angle_preference * (max_val - min_val) + min_val


class GazeState:
    def __init__(self):
        self.mu = torch.zeros(2)
        self.Sigma = torch.eye(2)
        self.current_time = 0
        self.gaze_object = None
        # self.object_center = None
        # self.object_shift = torch.zeros(2)
        self.sensitivity_map = torch.zeros(200, 200)

    def init_state(self, mu, Sigma, sensitivity_map):
        self.mu = mu
        self.Sigma = Sigma
        self.current_time = 0
        self.gaze_object = 0
        # self.object_center = torch.zeros(2)  # TODO: set to object center
        # self.object_shift = torch.zeros(2)

        self.sensitivity_map = sensitivity_map

    def update_state(self, mu, Sigma, sensitivity_map, gaze_object):  # object_center, object_shift
        self.mu = mu
        self.Sigma = Sigma
        self.current_time = self.current_time + 1
        self.gaze_object = gaze_object
        # self.object_center = object_center
        # self.object_shift = object_shift
        self.sensitivity_map = sensitivity_map

    def create_visualization(self, image_size):
        # G is a (verz small!) gaussian around the gaze position, from which onlz the position is used
        # TODO: return pos (instead of G) and seneitivity?!
        X, Y = torch.meshgrid(
            torch.arange(0, image_size[0]), torch.arange(0, image_size[1]), indexing="ij"
        )
        G = torch.exp(
            -0.5
            * (
                    ((X - self.mu[0]) / self.Sigma[0, 0]) ** 2
                    + ((Y - self.mu[1]) / self.Sigma[1, 1]) ** 2
            )
        )
        return G, self.sensitivity_map


class GazeFilter:
    def __init__(self, config):
        self.sens_spread = config.sensitivity_dva_sigma  # set in [DVA] (~7-10)
        self.px2dva = 0.0
        self.saccadic_momentum = config.saccadic_momentum
        self.sac_momentum_min = config.sac_momentum_min  # set in [0, 1], where 1 means no saccadic momentum
        self.sac_momentum_max = config.sac_momentum_max  # was previously implicitly 1 or 2, now chosen explicitly
        self.sac_momentum_restricted_angle = config.sac_momentum_restricted_angle  # 180 means whole range, smaller leads to cone
        self.sac_momentum_on_obj = config.sac_momentum_on_obj  # experimental...
        self.presaccadic_prompting = config.presaccadic_prompting
        self.presaccadic_sensitivity = config.presaccadic_sensitivity

    def predict(self, state, action, dt):
        mu, Sigma = state.mu, state.Sigma
        F_mu, F_u = torch.eye(2), torch.eye(
            2
        )  # forward model for gaze location (currently linear)
        mu_new, Sigma_new = predict(
            mu, Sigma, action, F_mu, F_u, lambda mu, u: self.Q(mu, u, dt)
        )
        return mu_new, Sigma_new

    def correct(self, state, action, measurement, meas_time, segmentation_map, presac_obj, last_foveation):
        dt = meas_time - state.current_time
        mu, Sigma = self.predict(state, action, dt)
        H = torch.eye(2)  # Measurement model for gaze location (currently linear)
        mu_new, Sigma_new = update(mu, Sigma, measurement, H, self.R)
        xmax, ymax = segmentation_map.shape
        mu_new[0] = torch.clip(mu_new[0], 0, xmax - 1)
        mu_new[1] = torch.clip(mu_new[1], 0, ymax - 1)
        sensitivity_map = gaussian_2d_torch(
            int(mu[0]), int(mu[1]), xmax, ymax, self.sens_spread / self.px2dva
        )
        # potentially apply saccadic momentum, i.e. angle preference based on previous saccade direction, here:
        if self.saccadic_momentum & (self.sac_momentum_on_obj is False):
            sac_ang_h = last_foveation[10]
            if not np.isnan(sac_ang_h):
                sac_momentum_map = rotated_atan2_2d_torch(int(mu[0]), int(mu[1]), xmax, ymax, sac_ang_h, self.sac_momentum_min, self.sac_momentum_max, self.sac_momentum_restricted_angle)
                sensitivity_map = sensitivity_map * sac_momentum_map

        gaze_obj = segmentation_map[int(mu[0]), int(mu[1])].item()
        # set values in gaze_gaussian to 1 if they are in the foveated object
        # --> Object Based Attention; but not for the background!
        obj_mask = segmentation_map == gaze_obj
        if gaze_obj != 0:
            sensitivity_map[obj_mask] = 1

        # if sac_momentum_on_obj apply this afterwards!
        if self.saccadic_momentum & self.sac_momentum_on_obj:
            sac_ang_h = last_foveation[10]
            if not np.isnan(sac_ang_h):
                sac_momentum_map = rotated_atan2_2d_torch(int(mu[0]), int(mu[1]), xmax, ymax, sac_ang_h, self.sac_momentum_min, self.sac_momentum_max, self.sac_momentum_restricted_angle)
                sensitivity_map = sensitivity_map * sac_momentum_map


        presac_obj = presac_obj > 0
        if self.presaccadic_prompting:
            if presac_obj.any():
                sensitivity_map[presac_obj] = self.presaccadic_sensitivity

        # object_center_old = state.object_center
        # cm = np.array(center_of_mass(obj_mask.detach().cpu().numpy()))
        # object_center_new = torch.from_numpy(cm).type(torch.get_default_dtype()).to(torch.cuda.current_device()) # y, x
        # object_shift_new = object_center_new - object_center_old

        state.update_state(mu_new, Sigma_new, sensitivity_map, gaze_obj)  # object_center_new, object_shift_new

    def Q(self, mu, u, dt):
        return torch.eye(2)  # TODO define an actual forward noise

    def R(self, mu):
        return torch.eye(2)  # TODO define an actual measuremnt noise

if __name__ == "__main__":
    momentum = rotated_atan2_2d_torch(100, 100, 200, 300, 150, 0.85, 3, 30)
    print(torch.mean(momentum), torch.min(momentum), torch.max(momentum))

    sensitivity = gaussian_2d_torch(100, 100, 200, 300, 90)
    print(torch.mean(sensitivity))
    plt.imshow((momentum*sensitivity).cpu().numpy()); plt.colorbar(); plt.show()

    # plt.imshow(sensitivity.cpu().numpy()); plt.colorbar(); plt.show()
