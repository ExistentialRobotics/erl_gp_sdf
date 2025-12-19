Configuration Guidance
======================

This guidance shows you how to create the YAML file for running the surface mapping and SDF mapping.
The YAML file is loaded using an implementation from `erl_common`, which is based on `yaml-cpp` and
support nested YAML file loading. For each map/dict structure in the YAML file, another file will be
loaded at first if `__base__: <relative_file_path>` is provided. This enables cleaner configuration
management that commonly used parameters can be put in one place and shared across different config
files. For commonly used parameters in `erl_gp_sdf`, they are put in `config/common/xxxx.yaml`.

Surface Mapping Configuration
-----------------------------

In this section, we describe the configuration parameters for Bayesian-Hilbert Mapping (BHM) based
surface mapping.

For your interest, the corresponding C++ struct is defined
in [include/erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp#L81-L188](../include/erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp).

The commonly used parameters are put in [
config/common/bayesian_hilbert_surf_mapping_float.yaml](../config/common/bayesian_hilbert_surf_mapping_float.yaml).

Examples of configuration files are provided:

- [3D Depth Camera, Cow And Lady](../config/cow_and_lady/bayesian_hilbert_surf_mapping_float.yaml)
- [3D Lidar, Newer College](../config/newer_college/bayesian_hilbert_surf_mapping_float.yaml)
- [2D Lidar, Gazebo Room](../config/gazebo/bayesian_hilbert_surf_mapping_float.yaml)

The basic structure of the configuration file is as follows:

```yaml
__base__: <relative_path_to_common_config_file>
local_bhm: # parameters for local BHM
    ...
ray_selector: # parameters for ray selection
    ...
tree: # parameters for the occupancy quadtree/octree
    ...
update_tree: # parameters for updating the tree
    ...
update_map: # parameters for updating the map
    ...
scaling: 1.0  # global scaling factor for the input points
bhm_depth: 15  # depth of the tree node to store the local BHM
weight_sync: true  # whether to synchronize the overlapped weights between neighboring local BHMs
sync_method: "copy" # method for weight synchronization, options: "copy", "mean" or "bayesian"
hinged_grid_size: 7  # size of the hinged grid
bhm_overlap: 0.3   # overlap size between neighboring BHMs (in meters), used when `update_map.method` is 1.
bhm_overlap_sync: 1  # overlap size between neighboring BHMs (in grid), used when `update_map.method` is 2.
build_bhm_on_hit: true  # whether to build local BHM as long as there is at least one hit point
unknown_log_odds: 100.0 # log-odds value for unknown space (no tree nodes, no local BHMs)
test_knn: 1  # number of nearest neighbors to use during inference
test_batch_size: 1000 # batch size for inference
```

The above comments give a brief description of each parameter. Below is a more detailed explanation:

| Parameter          | Description                                                                                                                                                                                                                                                                                                        |
|--------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `local_bhm`        | Parameters for the local Bayesian-Hilbert Map. See [Local BHM Parameters](#local-bhm-parameters) for more details.                                                                                                                                                                                                 |
| `ray_selector`     | Parameters for ray selection. See [Ray Selector Parameters](#ray-selector-parameters) for more details.                                                                                                                                                                                                            |
| `tree`             | Parameters for the occupancy quadtree/octree. See [Tree Parameters](#tree-parameters) for more details.                                                                                                                                                                                                            |
| `update_tree`      | Parameters for updating the tree. See [Tree Parameters](#tree-parameters) for more details.                                                                                                                                                                                                                        |
| `update_map`       | Parameters for updating the map. See [Update Map Parameters](#update-map-parameters) for more details.                                                                                                                                                                                                             |
| `scaling`          | Global scaling factor for the input points. We use single-precision to achieve higher FPS. However, when the scene is too large, the SDF mapping has numerical issue with large `log_lambda`. Setting `scaling < 1` makes the surface mapping run at a smaller scale so that large `log_lambda` is still possible. |
| `bhm_depth`        | Depth of the tree node to store the local BHM. We suggest keeping it as `tree.tree_depth - 1`. So, $\text{bhm_node_size} = \text{tree_resolution} \times 2^{\text{tree_depth} - \text{bhm_depth}}$, $\text{bhm_node_size} = \text{bhm_node_size} + 2 \times \text{bhm_overlap}$.                                   |
| `weight_sync`      | Whether to synchronize the overlapped weights between neighboring local BHMs. We suggest keeping it as true. Weight sync speeds up the convergence of BHMs and improves the consistency between neighboring BHMs.                                                                                                  |
| `sync_method`      | Method for weight synchronization, options: "copy", "mean" or "bayesian".                                                                                                                                                                                                                                          |
| `hinged_grid_size` | Size of the hinged grid. $\text{hinged_grid_size}^{\text{D}}$ grid points are used for D-dim space. $\text{hinged_grid_resolution}=\text{bhm_node_size} / (\text{hinged_grid_size}-2\times \text{bhm_overlap_sync}).$                                                                                              |
| `bhm_overlap`      | Overlap size between neighboring BHMs (in meters), used when `update_map.method` is 1. (Depreacted because method=1 is not recommended.)                                                                                                                                                                           |
| `bhm_overlap_sync` | Overlap size between neighboring BHMs (in grid), used when `update_map.method` is 2. This will set $\text{bhm_overlap} = \text{bhm_overlap_sync} \times \text{hinged_grid_resolution}$.                                                                                                                            |
| `build_bhm_on_hit` | If true, build local BHM as long as there is at least one hit point. Otherwise, a local BHM is built until a tree node is classified as occupied.                                                                                                                                                                  |
| `bhm_test_margin`  | This defines the test boundary of a BHM. $\text{test_bhm_size} = \text{bhm_node_size} + 2 \times \text{bhm_test_margin}$. We suggest set it as approximately the $\text{hinged_grid_resolution}$.                                                                                                                  |
| `unknown_log_odds` | Log-odds value for unknown space (no tree nodes, no local BHMs). Positive log-odds means occupied and negative log-odds means free.                                                                                                                                                                                |
| `test_knn`         | Number of nearest neighbors to use during inference. We suggest keeping it as 1.                                                                                                                                                                                                                                   |
| `test_batch_size`  | Batch size for inference. Multi-threading is used during test if the number of queries is larger than `test_batch_size`. Otherwise, single-thread.                                                                                                                                                                 |

### Local BHM Parameters

The `local_bhm` section contains parameters for the local Bayesian-Hilbert Map. Here is an example:

```yaml
local_bhm:
    bhm:
        min_distance: 0.1
        max_distance: 20.0
        diagonal_sigma: true
        sampling_area_scale: 3.0
        free_points_per_meter: 3.0
        free_sampling_margin: 0.001
        init_mu: 0.0
        init_sigma: 10000
        num_em_iterations: 1
        sparse_zero_threshold: 0.001
        use_sparse: true
    kernel_type: erl::covariance::RadialBiasFunction<float, 3>
    kernel_setting_type: erl::covariance::Covariance<float>::Setting
    kernel:
        x_dim: 3
        scale: 0.022
        scale_mix: 1
    min_dataset_size: 10
    min_dataset_hit_size: 5
    hit_point_buffer_size: 100
    ray_buffer_size: 1024
    surface_grid_size: 5
    surface_log_odds: 0.0
    surface_log_odds_init_count: 1
    surface_log_odds_num_points: 100
    surface_log_odds_min: -5.0
    surface_log_odds_max: 20.0
    auto_surface_log_odds: true
    include_neighbor_voxels: true
    faster_prediction: true  # if true, assume the BHM is converged during prediction.
```

Before we dive into the details of each parameter, we first explain the map structure.

The parameters are mainly three parts: BHM parameters, dataset construction parameters, and surface extraction
parameters. Below is a detailed explanation of each parameter (the prefix `local_bhm.` is omitted for brevity):

**BHM Parameters**

| Parameter                   | Description                                                                                                                                                                                                     |
|-----------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bhm.diagonal_sigma`        | If true, assume the covariance matrix is diagonal. This speeds up the update significantly with slight accuracy drop.                                                                                           |
| `bhm.init_mu`               | Initial mean value for the BHM weights. We suggest 0 or a positive value between 0 and 10.                                                                                                                      |
| `bhm.init_sigma`            | Initial variance for the BHM weights. We suggest a large value like 10000 to represent high uncertainty at the beginning.                                                                                       |
| `bhm.num_em_iterations`     | Number of EM iterations during each BHM update. We suggest 1 for real-time mapping.                                                                                                                             |
| `bhm.sparse_zero_threshold` | Threshold to prune small weights in the sparse representation. We suggest 0.001                                                                                                                                 |
| `bhm.use_sparse`            | Whether to use sparse representation for BHM weights. We suggest true to save memory and speed up computation.                                                                                                  |
| `kernel_type`               | Type of the kernel function. `erl::covariance::RadialBiasFunction<float, 3>` for 3D mapping and `erl::covariance::RadialBiasFunction<float, 2>` for 2D mapping. `Matern32` should also work but is much slower. |
| `kernel_setting_type`       | Type of the kernel setting. Keep it as `erl::covariance::Covariance<float>::Setting`. If you want to use double, use `erl::covariance::Covariance<double>::Setting` and update `kernel_type` correspondingly.   |
| `kernel`                    | Settings for the kernel function. `x_dim` is the input dimension (2 for 2D mapping and 3 for 3D mapping). `scale` is the length scale. `scale_mix` is the scale mixture parameter, unused.                      |
| `faster_prediction`         | If true, assume the BHM is converged during prediction so that we can skip calculating the covariance. This speeds up the inference significantly with slight accuracy drop.                                    |

**Dataset Construction Parameters**

| Parameter                   | Description                                                                                                                                                                   |
|-----------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `bhm.min_distance`          | Minimum valid distance for points from the sensor origin. Points closer than this distance are ignored.                                                                       |
| `bhm.max_distance`          | Maximum valid distance for points from the sensor origin. Points farther than this distance are ignored.                                                                      |
| `bhm.sampling_area_scale`   | Scale of the sampling area for free space points. $\text{sampling_area_size} = \text{sampling_area_scale} \times \text{bhm_node_size}$.                                       |
| `bhm.free_points_per_meter` | Number of free space points to sample per meter along each ray. A reasonable value should make $\text{sampling_area_size} \times \text{free_points_per_meter}$ around 1 to 3. |
| `bhm.free_sampling_margin`  | Margin to avoid sampling free points too close to the hit point. 0.001 to 0.01 are reasonable.                                                                                |
| `min_dataset_size`          | Minimum number of total points (hit + free) required to update the BHM. If the number of points is less than this value, the BHM update is skipped.                           |
| `min_dataset_hit_size`      | Minimum number of hit points required to update the BHM. If the number of hit points is less than this value, the BHM update is skipped.                                      |
| `hit_point_buffer_size`     | Size of the buffer to store hit points. Points in the buffer are used to estimate the surface log-odds.                                                                       |
| `ray_buffer_size`           | Size of the buffer to store rays for free space sampling.                                                                                                                     |

**Surface Extraction Parameters**

| Parameter                     | Description                                                                                                                                                                                                                                 | 
|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `surface_grid_size`           | Size of the grid used for surface extraction. $\text{surface_grid_resolution} = \text{bhm_node_size} / \text{surface_grid_size}$.                                                                                                           |
| `surface_log_odds`            | Log-odds value threshold to classify a grid point as surface. If `auto_surface_log_odds` is true, this is used as the initial value.                                                                                                        |
| `surface_log_odds_init_count` | The initial count $N_0$ for estimating the surface log-odds. $l_t \leftarrow (l_{t-1}\times N_{t-1} + \sum_{j=N_{t-1}+1}^{N_t} l(p_j)) / N_t$                                                                                               |
| `surface_log_odds_num_points` | Max number of points $\Delta N$ used to estimate the surface log-odds in each update. $N_t = N_{t-1} + \min(\Delta N, \text{#hit_points in the buffer})$                                                                                    |
| `surface_log_odds_min`        | Minimum valid surface log-odds value. If the estimated surface log-odds is less than this value, the whole BHM is deactivated.                                                                                                              |
| `surface_log_odds_max`        | Maximum valid surface log-odds value. If the estimated surface log-odds is larger than this value, the whole BHM is deactivated. If $l(p) \not \in [-s, s], s = 10 \times \text{surface_log_odds_max}$, $l(p)$ is considered as an outlier. |
| `auto_surface_log_odds`       | If true, automatically estimate the surface log-odds from the incoming data. Otherwise, use the fixed value defined in `surface_log_odds`.                                                                                                  |
| `include_neighbor_voxels`     | If true, during surface extraction, also include the neighboring voxels of the detected surface voxels. This helps to reduce holes.                                                                                                         |

### Ray Selector Parameters

Ray selector is used to select a subset of rays from the input point cloud for updating a local BHM. It should be
configured based on the sensor specifications. Here we show examples for 3D LiDAR, depth camera and 2D LiDAR.

#### 3D LiDAR

For a synthesized 3D LiDAR with 360 horizontal FOV, 90 vertical FOV, 360 horizontal beams and 90 vertical beams, the
configuration can be as follows:

```yaml
ray_selector:
    azimuth_min: -3.1415926535897931
    azimuth_max: 3.1415926535897931
    elevation_min: -1.5707963267948966
    elevation_max: 1.5707963267948966
    num_azimuth_angles: 360
    num_elevation_angles: 90
    transform:
        - [ 1, 0, 0, 0 ]
        - [ 0, 1, 0, 0 ]
        - [ 0, 0, 1, 0 ]
```

`azimuth_min`, `azimuth_max`, `elevation_min`, `elevation_max` define the field of view (FOV) of the sensor in radians.
`num_azimuth_angles` and `num_elevation_angles` define the grid shape to partition the FOV. `transform` is the
transformation matrix applied to the input points (in a local frame) if we want to select rays in a different frame.

The smaller the grid size (i.e., `num_azimuth_angles` x `num_elevation_angles`), the more rays will be selected for
updating the BHM. This speeds up the ray selection but may slow down the BHM update because more rays need to be
processed.

#### Depth Camera

For a depth camera with `640x480` resolution, 60 degree horizontal FOV and 50 degree vertical FOV, the configuration can
be as follows:

```yaml
ray_selector:
    azimuth_min: -0.55
    azimuth_max: 0.55
    elevation_min: -0.43
    elevation_max: 0.43
    num_azimuth_angles: 640
    num_elevation_angles: 480
    transform: # transform from optical frame to camera frame
        - [ 0, 0, 1, 0 ]
        - [ -1, 0, 0, 0 ]
        - [ 0, -1, 0, 0 ]
```

Here the transform matrix converts points from the optical frame (Z forward, X right, Y down) to the camera frame (X
forward, Y left, Z up).

#### 2D LiDAR

Similarly, for a 2D LiDAR with 270 degree horizontal FOV and 270 beams, the configuration can be as follows:

```yaml
ray_selector:
    angle_min: -2.356194496154785
    angle_max: 2.338737726211548
    num_angles: 270
    transform:
        - [ 1, 0, 0 ]
        - [ 0, 1, 0 ]
```

### Tree Parameters

The `tree` section contains parameters for the occupancy quadtree/octree. And the `update_tree` section contains
parameters for updating the tree.
Here is an example:

```yaml
tree:
    tree_depth: 16
    resolution: 0.1
    log_odd_min: -10.0
    log_odd_max: 50.0
    log_odd_hit: 0.95
    log_odd_miss: -0.05
    log_odd_occ_threshold: 0.0
    use_change_detection: false
    use_aabb_limit: false
    aabb:
        center: [ 0.0, 0.0, 0.0 ]
        half_sizes: [ 0.0, 0.0, 0.0 ]
update_tree: # we suggest keeping the following settings for surface mapping
    with_count: false
    parallel: true
    lazy_eval: true
    discrete: true
```

The parameters are explained in the table below:

| Parameter                    | Description                                                                                                                                                  |
|------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `tree.tree_depth`            | Depth of the tree. The max size of the tree is $\text{max_tree_size} = \text{resolution} \times 2^{\text{tree_depth}}$.                                      |
| `tree.resolution`            | Size of the tree node at the deepest level.                                                                                                                  |
| `tree.log_odd_min`           | Minimum log-odds value for a tree node. Nodes with log-odds less than this value will be clipped.                                                            |
| `tree.log_odd_max`           | Maximum log-odds value for a tree node. Nodes with log-odds larger than this value will be clipped.                                                          |
| `tree.log_odd_hit`           | Log-odds increment for a hit point.                                                                                                                          |
| `tree.log_odd_miss`          | Log-odds decrement for a miss point.                                                                                                                         |
| `tree.log_odd_occ_threshold` | Log-odds threshold to classify a node as occupied. Nodes with log-odds larger than this value are considered occupied.                                       |
| `tree.use_change_detection`  | If true, keep track of changed nodes during tree update. Keep false as this is not used in surface mapping.                                                  |
| `tree.use_aabb_limit`        | If true, only update nodes within the defined Aixis-Aligned Bounding Box (AABB).                                                                             |
| `tree.aabb`                  | Definition of the AABB. `center` is the center of the AABB and `half_sizes` are the half sizes along each dimension.                                         |
| `update_tree.with_count`     | If true, keep track of the number of hit and miss points for each node. Not used in surface mapping.                                                         |
| `update_tree.parallel`       | If true, use multi-threading to speed up the tree update.                                                                                                    |
| `update_tree.lazy_eval`      | If true, update intermediate nodes' log-odds after all leaf nodes are updated. This speeds up the update.                                                    |
| `update_tree.discrete`       | If true, merge the hit/miss points into discrete voxels before updating the tree. This avoids duplicated updates to the same voxel and speeds up the update. |

The purpose of the occupancy tree is to:

- control the density of local BHMs;
- determine where to build local BHMs;
- prune unnecessary BHMs in free space;
- efficiently locate BHMs within a region, thus efficiently collect estimated surface points within a bounding box for
  SDF mapping.

### Update Map Parameters

The `update_map` section contains parameters for updating the map. Here is an example:

```yaml
update_map:
    method: 2
    surface_max_abs_logodd: 2
    surface_bad_abs_logodd: 200
    surface_step_size: 0.001
    max_num_points: 50000
    max_adjust_tries: 1
    include_neighbor_bhm: true
    max_num_bhm: 1500
    max_num_voxels: 10000
    var_scale: 0.1
    var_max: 1.5
```

The parameters are explained in the table below:

| Parameter                | Description                                                                                                                                                                                                                                                                                                                                                                                                                      |
|--------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `method`                 | Method for updating the surface estimation. Options: 1 or 2. We suggest using method 2. <br> Method 1: Update the surface point by moving it along the predicted gradient to minimize the absolute log-odds. <br> Method 2: Update the map using marching squares/cubes on a grid of predicted log-odds values with automatically estimated surface log-odds. <br> Method 2 is prefered because it is more efficient and robust. |
| `surface_max_abs_logodd` | Maximum absolute log-odds value for surface points. Used when `update_map.method` is 1. If the absolute log-odds of a surface point is smaller than this value, the point is not updated.                                                                                                                                                                                                                                        |
| `surface_bad_abs_logodd` | Bad absolute log-odds value for surface points. Used when `update_map.method` is 1. If the absolute log-odds of a surface point is larger than this value, the point is considered as an outlier and removed.                                                                                                                                                                                                                    |
| `surface_step_size`      | Step size for updating surface points. Used when `update_map.method` is 1.                                                                                                                                                                                                                                                                                                                                                       |
| `max_num_points`         | Maximum number of surface points to update during each map update. Used when `update_map.method` is 1.                                                                                                                                                                                                                                                                                                                           |
| `max_adjust_tries`       | Maximum number of adjustment tries for each surface point during update. Used when `update_map.method` is 1.                                                                                                                                                                                                                                                                                                                     |
| `include_neighbor_bhm`   | If true, during map update, also include neighboring BHMs of the BHMs that contain hit points. This helps to improve the consistency between neighboring BHMs.                                                                                                                                                                                                                                                                   |
| `max_num_bhm`            | Maximum number of BHMs to update if the number of BHMs to update is less than this value. If the number of BHMs to update is larger than this value, no more BHMs are added to the update list. Otherwise, add more BHMs that are not in the current observation but have sample rays to train.                                                                                                                                  |
| `max_num_voxels`         | Maximum number of voxels to extract surface points from during each map update. Used when `update_map.method` is 2.                                                                                                                                                                                                                                                                                                              |
| `var_scale`              | Scale factor for the surface point's position variance.                                                                                                                                                                                                                                                                                                                                                                          |
| `var_max`                | Maximum variance for the surface point's position. <br> $\text{var_position} = \text{var_gradient} = \min(\text{var_max}, \text{var_scale} \times \operatorname{abs}\left(l(p) - \text{surface_log_odds}\right))$.                                                                                                                                                                                                               |

SDF Mapping Configuration
-------------------------

In this section, we describe the configuration parameters for SDF mapping based on the surface map built from BHM.
Here is an example configuration file:
<details>

<summary>example SDF mapping config file</summary>

```yaml
test_query:
    default_invalid_sdf: -1
    max_test_valid_distance_var: 0.02
    search_area_half_size: 4
    num_neighbor_gps: 8
    use_smallest: true
    compute_gradient: true # whether to compute gradient
    compute_gradient_variance: false # whether to compute gradient variance
    compute_covariance: false # whether to compute covariance
    use_gp_covariance: false
    retrain_outdated: true  # whether to retrain the trained but outdated GPs
    max_num_retrain_gps: 500  # max number of retrained GPs before a test query
    use_global_buffer: true  # whether to use the global buffer for test queries
queue_priority:
    max_buf_outdated_count: 10000
    distance_weight: 0.1
    query_weight_for_loading: 0.1
    query_weight_for_retrain: 0.5
num_threads: 64
min_num_gps_to_update: 1024
update_hz: 10  # minimum update frequency to maintain, it may be slower
sensor_noise: 0.001
gp_sdf_area_scale: 8  # should be large enough
max_valid_gradient_var: 0.1
invalid_position_var: 2.0
sdf_gp:
    sign_method: "external" # "sign_gp", "normal_gp", "external", "hybrid", "none"
    hybrid_sign_methods: [ "sign_gp", "external" ]
    hybrid_sign_threshold: 0.18 # > sqrt(3) * tree_resolution
    normal_scale: 1000.0
    softmin_temperature: 10.0
    sign_gp_offset_distance: -0.2  # <=0, when it is 0.0, the sign GP relies on the gradients only
    edf_gp_offset_distance: 0.0    # >=0, e.g. 0.015 will cause a shell of thickness 0.03 behind the surface
    sign_gp:
        kernel_type: erl::covariance::Matern32<float, 3>
        kernel_setting_type: erl::covariance::Covariance<float>::Setting
        kernel:
            x_dim: 3
            scale: 0.1
            scale_mix: 1
        max_num_samples: 48 # max dataset size
        no_gradient_observation: false
    edf_gp:
        kernel_type: erl::covariance::RadialBiasFunction<float, 3>
        kernel_setting_type: erl::covariance::Covariance<float>::Setting
        kernel:
            x_dim: 3
            scale: 0.043 # will be overridden by log_lambda
            scale_mix: 1
        log_lambda: 25  # larger more accurate but less numerically stable
        duplicate_epsilon: 0.0
        max_num_samples: 128 # max dataset size
        no_gradient_observation: true
```

</details>

Most parameters are common between 2D and 3D SDF mapping with different datasets. Common parameters are stored
in [config/common/sdf_mapping_float.yaml](../config/common/sdf_mapping_float.yaml). Then, for different datasets and
sensors, the configuration files only need to override a few parameters like `sdf_gp.edf_gp.log_lambda`. Example
configuration files are available in

- [3D Depth Camera, Cow And Lady](../config/cow_and_lady/bayesian_hilbert_sdf_mapping_float.yaml)
- [3D LiDAR, Newer College](../config/newer_college/bayesian_hilbert_sdf_mapping_float.yaml)
- [2D LiDAR, Gazebo Room](../config/gazebo/bayesian_hilbert_sdf_mapping_float.yaml)

We explain the parameters in detail below, with important parameters highlighted.

| Parameter                                 | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
|-------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `test_query.default_invalid_sdf`          | Default SDF value for invalid queries. If a query point is too far from any SDF gp (i.e. in unknown region), it is considered invalid and assigned this SDF value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `test.query.max_test_valid_distance_var`  | If the GP prediction has a variance smaller than this value, the prediction is considered good enough as the final prediction. Used when `use_smallest` is false.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| **test_query.search_area_half_size**      | Half size of the search area (in meters) to look for neighboring SDF GPs. If a query position cannot find a GP within the region, this value is automatically doubled.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `test_query.num_neighbor_gps`             | Number of neighboring SDF GPs to use for prediction.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `test_query.use_smallest`                 | If true, pick the smallest predicted SDF value among all neighboring GPs as the final prediction. Otherwise, use variance-based fusion.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `test_query.compute_gradient`             | Whether to predict the SDF gradient along with the SDF value.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `test_query.compute_gradient_variance`    | Whether to compute the variance of the predicted SDF gradient.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `test_query.compute_covariance`           | Whether to compute the covariance between the predicted SDF value and the gradient.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `test_query.use_gp_covariance`            | Whether to use the GP covariance as the prediction variance. False is suggested because GP's covariance does not reflect the prediction uncertainty correctly with log mapping.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| `test_query.retrain_outdated`             | Whether to retrain the trained but outdated GPs before a test query. This helps to improve the prediction accuracy at the cost of computation time.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **test_query.max_num_retrain_gps**        | Maximum number of retrained GPs before a test query. Larger value makes the SDF mapping update to the latest faster but may harm the real-time performance. When the area is well observed, the SDF prediction does not change much. So, this value should be adjusted to balance the speed and the in-time accuracy in your application.                                                                                                                                                                                                                                                                                                                                                      |
| `test_query.use_global_buffer`            | Whether to use the global buffer for test queries. If true, all threads share a global buffer to store the prediction results. This is useful to collect the prediction from used GPs.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `queue_priority.max_buf_outdated_count`   | The max buffer-outdated count for a GP in the priority queue. Larger value means higher priority to update.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| `queue_priority.distance_weight`          | Weight for the distance term in the priority calculation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `queue_priority.query_weight_for_loading` | Weight for the query count term when computing the priority for loading GP's training data. <br> $\text{loading_priority}=\text{buffer_outdated_count}(1 + \text{query_weight_for_loading} \times \text{query_count})$.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `queue_priority.query_weight_for_retrain` | Weight for the query count term when computing the priority for retraining GP. <br> $\text{retrain_priority}=\text{gp_outdated_count}(1 + \text{query_weight_for_retrain} \times \text{query_count})$.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `num_threads`                             | Number of threads for SDF GP prediction and update. If set to 0, use the number of hardware threads available. If too large, it is clipped to the number of hardware threads.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `update_hz`                               | Minimum update frequency to maintain. The actual update frequency may be slower depending on the computation time.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `sensor_noise`                            | Standard deviation of the sensor noise (in meters). This is used to model the observation noise for SDF GP training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| **gp_sdf_area_scale**                     | Scale to collect training data for each SDF GP. $\text{gp_sdf_area_size} = \text{gp_sdf_area_scale} \times \text{bhm_node_size}$. This value should be large enough to cover sufficient surface points for training.                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `max_valid_gradient_var`                  | Maximum valid gradient variance for training points. Points with gradient variance larger than this value are considered invalid and assigned a position variance of `invalid_position_var`.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `invalid_position_var`                    | Position variance assigned to invalid training points.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `sdf_gp.sign_method`                      | Method to determine the sign of the SDF value during prediction. We suggest using `external` to get sign from the surface mapping. Options: <br> - `sign_gp`: use the sign GP to determine the sign. <br> - `normal_gp`: use the normal GP's gradient prediction with the EDF GP's gradient to determine the sign. $\text{sdf_sign} = \operatorname{sign}(\mathbf{g}_\text{normal_GP}^\top \mathbf{g}_\text{EDF_GP})$. <br> - `external`: use external sign information (i.e., from the surface mapping). <br> - `hybrid`: use two sign methods for near and far away from surface correspondingly. <br> - `none`: do not use any sign information (predict unsign distance function instead). |
| `sdf_gp.hybrid_sign_methods`              | If `sign_method` is `hybrid`, use the two methods specified in this list for near and far away from surface correspondingly.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `sdf_gp.hybrid_sign_threshold`            | If `sign_method` is `hybrid`, this threshold defines the distance to the surface to switch between the two sign methods.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `sdf_gp.normal_scale`                     | Scale factor for the normal GP's gradient training data. Larger value makes the prediction more stable at distant positions.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `sdf_gp.softmin_temperature`              | Temperature parameter for softmin fusion of SDF variance computation with the training data from the EDF GP. Larger value makes the fusion closer to min operation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `sdf_gp.sign_gp_offset_distance`          | This value makes the training points for the sign GP shifted along the normal direction by this distance. Negative value means shifting towards inside the surface. When it is 0.0, the sign GP relies on the gradients only.                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| `sdf_gp.edf_gp_offset_distance`           | This value makes the training points for the EDF GP shifted along the normal direction by this distance. Positive value means shifting towards outside the surface. For example, 0.015 will cause a shell of thickness 0.03 behind the surface to have negative SDF value. This is a trick to create negative SDF prediction around the surface as a safe margin without getting sign. However, it makes the surface encoded in the SDF prediction inaccurate due to the shift of training points.                                                                                                                                                                                             |
| `sdf_gp.sign_gp.kernel_type`              | Kernel type for the sign GP.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `sdf_gp.sign_gp.kernel_setting_type`      | Kernel setting type for the sign GP.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| `sdf_gp.sign_gp.kernel.x_dim`             | Input dimension for the sign GP kernel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `sdf_gp.sign_gp.kernel.scale`             | Scale parameter for the sign GP kernel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `sdf_gp.sign_gp.kernel.scale_mix`         | Scale mix parameter when the kernel is a rational quadratic kernel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `sdf_gp.sign_gp.max_num_samples`          | Maximum number of training samples for the sign GP.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `sdf_gp.sign_gp.no_gradient_observation`  | If true, do not use gradient observations for the sign GP.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `sdf_gp.edf_gp.kernel_type`               | Kernel type for the EDF GP. Options: `erl::covariance::RadialBiasFunction<float, 3>` or `erl::covariance::Matern32<float, 3>` for 3D; `erl::covariance::RadialBiasFunction<float, 2>` or `erl::covariance::Matern32<float, 2>` for 2D.                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `sdf_gp.edf_gp.kernel_setting_type`       | Kernel setting type for the EDF GP.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `sdf_gp.edf_gp.kernel.x_dim`              | Input dimension for the EDF GP kernel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `sdf_gp.edf_gp.kernel.scale`              | Scale parameter for the EDF GP kernel. This value will be overridden by `log_lambda`. <br> When `kernel_type` is `erl::covariance::RadialBiasFunction`, $\text{scale}=\sqrt{\frac{1}{2\lambda}}$. <br> When `kernel_type` is `erl::covariance::Matern32`, $\text{scale}=\frac{\sqrt{3}}{\lambda}$.                                                                                                                                                                                                                                                                                                                                                                                             |
| `sdf_gp.edf_gp.kernel.scale_mix`          | Scale mix parameter when the kernel is a rational quadratic kernel.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| **sdf_gp.edf_gp.log_lambda**              | Log lambda parameter for the EDF GP. Larger value makes the GP more accurate but less numerically stable. EDF GP is trained on surface points with the following log mapping: <br> $y = \exp(-\lambda d_\text{SDF})$. Since $d_\text{SDF}$ is 0 for surface points, $y=1$ for all training points.                                                                                                                                                                                                                                                                                                                                                                                             |
| `sdf_gp.edf_gp.duplicate_epsilon`         | Minimum distance between two training points. If two points are closer than this value, only one point is kept. This helps to get a better distribution of training points to improve numerical stability. We suggest 0.0 because the Bayesian-Hilbert map based surface mapping does not have duplicate point issues.                                                                                                                                                                                                                                                                                                                                                                         |
| **sdf_gp.edf_gp.max_num_samples**         | Maximum number of training samples for the EDF GP. You may increase this value to make the result more accurate. But a too big `max_num_samples` only slows down the GP.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `sdf_gp.edf_gp.no_gradient_observation`   | Please set to true. Gradient observations are not used for the EDF GP.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
