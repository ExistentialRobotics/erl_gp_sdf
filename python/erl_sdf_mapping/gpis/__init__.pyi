from typing import Optional
from typing import Tuple
from typing import overload

import numpy as np
import numpy.typing as npt

from erl_common.yaml import YamlableBase
from erl_gaussian_process import LidarGaussianProcess1D
from erl_sdf_mapping import LogSdfGaussianProcess
from erl_gaussian_process import NoisyInputGaussianProcess
from erl_geometry import NodeData
from erl_geometry import IncrementalQuadtree
from erl_geometry import Node
from erl_geometry import NodeContainer

__all__ = [
    "GpisData2D",
    "GpisNode2D",
    "GpisNodeContainer2D",
    "GpisMapBase2D",
    "GpisMap2D",
    "LogGpisMap2D",
    "GpisData3D",
    "GpisNode3D",
    "GpisNodeContainer3D",
]

class GpisData2D(NodeData):
    distance: float
    gradient: npt.NDArray[np.float64]
    var_position: float
    var_gradient: float

    def update_data(
        self: GpisData2D,
        new_distance: float,
        new_gradient: npt.NDArray[np.float64],
        new_var_position: float,
        new_var_gradient: float,
    ): ...
    def __str__(self: GpisData2D) -> str: ...

class GpisNode2D(Node):
    def __init__(self: GpisNode2D, position: npt.NDArray[np.float64]) -> None: ...

class GpisNodeContainer2D(NodeContainer):
    class Setting(YamlableBase):
        capacity: int
        min_squared_distance: float
    setting: Setting

    def __len__(self: GpisNodeContainer2D) -> int: ...

class GpisMapBase2D:
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
            occ_test_temperature: float
            min_observable_occ: float
            max_surface_abs_occ: float
            max_valid_gradient_var: float
            max_adjust_tries: int
            max_bayes_position_var: float
            max_bayes_gradient_var: float

        class UpdateGpSdf(YamlableBase):
            add_offset_points: bool
            offset_distance: float
            search_area_scale: float
            zero_gradient_threshold: float
            max_valid_gradient_var: float
            invalid_position_var: float

        class TestQuery(YamlableBase):
            max_test_valid_distance_var: float
            search_area_half_size: float
            use_nearest_only: bool
        init_tree_half_size: float
        perturb_delta: float
        compute_variance: ComputeVariance
        update_map_points: UpdateMapPoints
        update_gp_sdf: UpdateGpSdf
        gp_theta: LidarGaussianProcess1D.Setting
        gp_sdf: NoisyInputGaussianProcess.Setting
        node_container: GpisNodeContainer2D.Setting
        quadtree: IncrementalQuadtree.Setting
        test_query: TestQuery
    @property
    def quadtree(self: GpisMapBase2D) -> IncrementalQuadtree: ...
    def update(
        self: GpisMapBase2D,
        angles: npt.NDArray[np.float64],
        distances: npt.NDArray[np.float64],
        pose: npt.NDArray[np.float64],
    ) -> bool: ...
    def test(
        self: GpisMapBase2D, xy: npt.NDArray[np.float64]
    ) -> Tuple[
        Optional[npt.NDArray[np.float64]],
        Optional[npt.NDArray[np.float64]],
        Optional[npt.NDArray[np.float64]],
        Optional[npt.NDArray[np.float64]],
    ]: ...
    def compute_sddf_v2(
        self: GpisMapBase2D,
        positions: npt.NDArray[np.float64],
        angles: npt.NDArray[np.float64],
        threshold: float,
        max_distance: float,
        max_marching_steps: int,
    ) -> npt.NDArray[np.float64]: ...
    def dump_quadtree_structure(self: GpisMapBase2D) -> str: ...
    def dump_surface_points(self: GpisMapBase2D) -> npt.NDArray[np.float64]: ...
    def dump_surface_normals(self: GpisMapBase2D) -> npt.NDArray[np.float64]: ...
    def dump_surface_data(
        self: GpisMapBase2D,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64],]: ...

class GpisMap2D(GpisMapBase2D):
    @overload
    def __init__(self: GpisMap2D): ...
    @overload
    def __init__(self: GpisMap2D, setting: GpisMap2D.Setting): ...
    @property
    def setting(self: GpisMap2D) -> GpisMap2D.Setting: ...

class LogGpisMap2D(GpisMapBase2D):
    class Setting(GpisMapBase2D.Setting):
        gp_sdf: LogSdfGaussianProcess.Setting

        def __init__(self: LogGpisMap2D.Setting): ...

    @overload
    def __init__(self: LogGpisMap2D): ...
    @overload
    def __init__(self: LogGpisMap2D, setting: Setting): ...
    @property
    def setting(self: LogGpisMap2D) -> Setting: ...

class GpisData3D(NodeData):
    distance: float
    gradient: npt.NDArray[np.float64]
    var_position: float
    var_gradient: float

    def update_data(
        self: GpisData3D,
        new_distance: float,
        new_gradient: npt.NDArray[np.float64],
        new_var_position: float,
        new_var_gradient: float,
    ): ...
    def __str__(self: GpisData3D) -> str: ...

class GpisNode3D(Node):
    def __init__(self: GpisNode3D, position: npt.NDArray[np.float64]) -> None: ...

class GpisNodeContainer3D(NodeContainer):
    class Setting(YamlableBase):
        capacity: int
        min_squared_distance: float
    setting: Setting

    def __len__(self: GpisNodeContainer3D) -> int: ...
