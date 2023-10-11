from typing import overload
from typing import Tuple
from typing import Optional
import numpy as np
import numpy.typing as npt
from erl_common.yaml import YamlableBase
from erl_gaussian_process import NoisyInputGaussianProcess
from . import gpis

__all__ = [
    "LogSdfGaussianProcess",
    "AbstractSurfaceMapping2D",
    "GpOccSurfaceMapping2D",
    "GpSdfMapping2D",
]

class LogSdfGaussianProcess(NoisyInputGaussianProcess):
    class Setting(NoisyInputGaussianProcess.Setting):
        log_lambda: float

        def __init__(self: LogSdfGaussianProcess.Setting): ...

    def __init__(self: LogSdfGaussianProcess, setting: Setting): ...
    def reset(self: LogSdfGaussianProcess) -> None: ...
    def train(
        self: LogSdfGaussianProcess,
        mat_x_train: npt.NDArray[np.float64],
        vec_grad_flag: npt.NDArray[np.bool_],
        vec_y: npt.NDArray[np.float64],
        vec_sigma_x: npt.NDArray[np.float64],
        vec_sigma_y: npt.NDArray[np.float64],
        vec_sigma_grad: npt.NDArray[np.float64],
    ) -> None: ...
    def test(
        self: LogSdfGaussianProcess, mat_x_test: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: ...

class AbstractSurfaceMapping2D: ...

class GpOccSurfaceMapping2D(AbstractSurfaceMapping2D):
    class Setting(YamlableBase):
        class ComputeVariance(YamlableBase):
            zero_gradient_position_var: float
            zero_gradient_gradient_var: float
            min_distance_var: float
            max_distance_var: float
            position_var_alpha: float
            min_gradient_var: float
            max_gradient_var: float

        class UpdateMapPoints(YamlableBase):
            min_observable_occ: float
            max_surface_abs_occ: float
            max_valid_gradient_var: float
            max_adjust_tries: int
            max_bayes_position_var: float
            max_bayes_gradient_var: float
        gp_theta: LogSdfGaussianProcess.Setting
        compute_variance: ComputeVariance
        update_map_points: UpdateMapPoints
        quadtree_resolution: float
        cluster_level: int
        perturb_delta: float
        update_occupancy: bool
    @overload
    def __init__(self): ...
    @overload
    def __init__(self, setting: Setting): ...
    @property
    def setting(self) -> Setting: ...

class GpSdfMapping2D:
    class Setting(YamlableBase):
        class TestQuery(YamlableBase):
            max_test_valid_distance_var: float
            search_area_half_size: float
            use_nearest_only: bool
        num_threads: int
        update_hz: float
        gp_sdf_area_scale: float
        offset_distance: float
        zero_gradient_threshold: float
        max_valid_gradient_var: float
        invalid_position_var: float
        train_gp_immediately: bool
        gp_sdf: GpSdfMapping2D.Setting
        test_query: TestQuery
    def __init__(self, surface_mapping: AbstractSurfaceMapping2D, setting: Setting): ...
    @property
    def setting(self) -> Setting: ...
    @property
    def surface_mapping(self) -> AbstractSurfaceMapping2D: ...
    def update(
        self: GpSdfMapping2D,
        angles: npt.NDArray[np.float64],
        distances: npt.NDArray[np.float64],
        pose: npt.NDArray[np.float64],
    ) -> bool: ...
    def test(
        self: GpSdfMapping2D, xy: npt.NDArray[np.float64]
    ) -> Tuple[
        Optional[npt.NDArray[np.float64]],  # sdf
        Optional[npt.NDArray[np.float64]],  # sdf gradient
        Optional[npt.NDArray[np.float64]],  # variance
        Optional[npt.NDArray[np.float64]],  # covariance
    ]: ...
