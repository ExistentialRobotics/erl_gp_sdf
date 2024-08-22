from typing import Optional
from typing import Tuple

import numpy as np
import numpy.typing as npt
from erl_common.yaml import YamlableBase

from erl_covariance import Covariance
from erl_gaussian_process import LidarGaussianProcess2D
from erl_gaussian_process import NoisyInputGaussianProcess
from erl_gaussian_process import RangeSensorGaussianProcess3D
from erl_geometry import AbstractSurfaceMapping
from erl_geometry import AbstractSurfaceMapping2D
from erl_geometry import AbstractSurfaceMapping3D
from erl_geometry import QuadtreeKey
from erl_geometry import SurfaceMappingOctree
from erl_geometry import SurfaceMappingQuadtree

__all__ = [
    "LogSdfGaussianProcess",
    "GpOccSurfaceMapping2D",
    "GpOccSurfaceMapping3D",
    "GpSdfMapping2D",
    "GpSdfMapping3D",
]

class LogSdfGaussianProcess(NoisyInputGaussianProcess):
    class Setting(NoisyInputGaussianProcess.Setting):
        log_lambda: float
        edf_threshold: float
        unify_scale: bool

        def __init__(self: LogSdfGaussianProcess.Setting): ...

    def __init__(self: LogSdfGaussianProcess, setting: Setting): ...
    def reset(self: LogSdfGaussianProcess, max_num_samples: int, x_dim: int) -> None: ...
    @property
    def log_kernel(self: LogSdfGaussianProcess) -> Covariance: ...
    @property
    def log_k_train(self: LogSdfGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def log_alpha(self: LogSdfGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def log_cholesky_k_train(self: LogSdfGaussianProcess) -> npt.NDArray[np.float64]: ...
    @property
    def memory_usage(self: LogSdfGaussianProcess) -> int: ...
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

class GpOccSurfaceMappingBaseSetting(AbstractSurfaceMapping.Setting):
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
        min_position_var: float
        min_gradient_var: float

    compute_variance: ComputeVariance
    update_map_points: UpdateMapPoints
    cluster_level: int
    perturb_delta: float
    zero_gradient_threshold: float
    update_occupancy: bool

class GpOccSurfaceMapping2D(AbstractSurfaceMapping2D):
    class Setting(GpOccSurfaceMappingBaseSetting):
        sensor_gp: LidarGaussianProcess2D.Setting
        quadtree: SurfaceMappingQuadtree.Setting

    def __init__(self, setting: Setting): ...
    @property
    def setting(self) -> Setting: ...

class GpOccSurfaceMapping3D(AbstractSurfaceMapping3D):
    class Setting(GpOccSurfaceMappingBaseSetting):
        sensor_gp: RangeSensorGaussianProcess3D.Setting
        octree: SurfaceMappingOctree.Setting

    def __init__(self, setting: Setting): ...
    @property
    def setting(self) -> Setting: ...

class GpSdfMappingBaseSetting(YamlableBase):
    class TestQuery(YamlableBase):
        max_test_valid_distance_var: float
        search_area_half_size: float
        num_neighbor_gps: int
        use_smallest: bool
        compute_covariance: bool
        recompute_variance: bool
        softmax_temperature: float

    num_threads: int
    update_hz: float
    gp_sdf_area_scale: float
    offset_distance: float
    max_valid_gradient_var: float
    invalid_position_var: float
    train_gp_immediately: bool
    gp_sdf: GpSdfMapping2D.Setting
    test_query: TestQuery
    log_timing: bool

class GpSdfMapping2D:
    class Gp:
        @property
        def active(self) -> bool: ...
        @property
        def locked_for_test(self) -> bool: ...
        @property
        def num_train_samples(self) -> int: ...
        @property
        def position(self) -> npt.NDArray[np.float64]: ...
        @property
        def half_size(self) -> float: ...
        @property
        def gp(self) -> LogSdfGaussianProcess: ...
        def train(self) -> None: ...

    class Setting(GpSdfMappingBaseSetting):
        surface_mapping_type: str
        surface_mapping: AbstractSurfaceMapping2D.Setting

    def __init__(self, setting: Setting): ...
    @property
    def setting(self) -> Setting: ...
    @property
    def surface_mapping(self) -> AbstractSurfaceMapping2D: ...
    def update(
        self: GpSdfMapping2D,
        rotation: npt.NDArray[np.float64],
        translation: npt.NDArray[np.float64],
        ranges: npt.NDArray[np.float64],
    ) -> bool: ...
    def test(self: GpSdfMapping2D, xy: npt.NDArray[np.float64]) -> Tuple[
        Optional[npt.NDArray[np.float64]],  # sdf
        Optional[npt.NDArray[np.float64]],  # sdf gradient
        Optional[npt.NDArray[np.float64]],  # variance
        Optional[npt.NDArray[np.float64]],  # covariance
    ]: ...
    @property
    def used_gps(self) -> list[tuple[GpSdfMapping2D.Gp, GpSdfMapping2D.Gp]]: ...
    @property
    def gps(self) -> dict[QuadtreeKey, GpSdfMapping2D.Gp]: ...
    @property
    def num_update_calls(self) -> int: ...
    @property
    def num_test_calls(self) -> int: ...
    @property
    def num_test_positions(self) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def write(self, filename: str) -> bool: ...
    def read(self, filename: str) -> bool: ...

class GpSdfMapping3D:
    class Gp:
        @property
        def active(self) -> bool: ...
        @property
        def locked_for_test(self) -> bool: ...
        @property
        def num_train_samples(self) -> int: ...
        @property
        def position(self) -> npt.NDArray[np.float64]: ...
        @property
        def half_size(self) -> float: ...
        @property
        def gp(self) -> LogSdfGaussianProcess: ...
        def train(self) -> None: ...

    class Setting(GpSdfMappingBaseSetting):
        surface_mapping_type: str
        surface_mapping: AbstractSurfaceMapping3D.Setting

    def __init__(self, setting: Setting): ...
    @property
    def setting(self) -> Setting: ...
    @property
    def surface_mapping(self) -> AbstractSurfaceMapping3D: ...
    def update(
        self: GpSdfMapping3D,
        rotation: npt.NDArray[np.float64],
        translation: npt.NDArray[np.float64],
        ranges: npt.NDArray[np.float64],
    ) -> bool: ...
    def test(self: GpSdfMapping3D, xyz: npt.NDArray[np.float64]) -> Tuple[
        Optional[npt.NDArray[np.float64]],  # sdf
        Optional[npt.NDArray[np.float64]],  # sdf gradient
        Optional[npt.NDArray[np.float64]],  # variance
        Optional[npt.NDArray[np.float64]],  # covariance
    ]: ...
    @property
    def used_gps(self) -> list[tuple[GpSdfMapping3D.Gp, GpSdfMapping3D.Gp]]: ...
    @property
    def gps(self) -> dict[QuadtreeKey, GpSdfMapping3D.Gp]: ...
    @property
    def num_update_calls(self) -> int: ...
    @property
    def num_test_calls(self) -> int: ...
    @property
    def num_test_positions(self) -> int: ...
    def __eq__(self, other) -> bool: ...
    def __ne__(self, other) -> bool: ...
    def write(self, filename: str) -> bool: ...
    def read(self, filename: str) -> bool: ...
