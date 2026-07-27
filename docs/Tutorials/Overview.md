## ROS YAMLs

Below are the two YAML files required to run `sdf_mapping_node`. You can copy each block into its own `.yaml` file.

---

### 1. Surface Mapping YAML

This file configures the surface‐mapping portion (e.g. `GpOccSurfaceMapping<float,2>`). Save as:

```yaml
# ------------------------------------------------------------
# Generic Sensor‐GP & Mapping Configuration YAML
# ------------------------------------------------------------

sensor_gp:
  # ----------------------------------------------------------
  # 1. Ray‐partitioning & grouping parameters
  # ----------------------------------------------------------
  # If true, partition hit‐rays when building GP training sets
  partition_on_hit_rays: false          

  # If true, enforce symmetric partitions for left/right scans
  symmetric_partitions: true            

  # Number of points per partition group, including overlaps
  group_size: 26                        

  # Number of points to overlap between adjacent groups
  overlap_size: 6                       

  # If > 0, pad each partition by this many points (for margin)
  margin: 0                              

  # Initial variance assigned to new GP training points
  init_variance: 1000000                

  # Assumed variance of range measurements (sensor model)
  sensor_range_var: 0.01               

  # Maximum allowed variance for an individual range reading
  # (drops points whose distance‐variance exceeds this)
  max_valid_range_var: 0.1             

  # Temperature parameter for occupancy‐test softmax (higher = softer)
  occ_test_temperature: 30             

  # ----------------------------------------------------------
  # 2. Sensor‐frame & ray parameters
  # ----------------------------------------------------------
  sensor_frame:
    # Minimum valid radial distance (meters) # rhese are sensor dependent
    valid_range_min: 0.2               

    # Maximum valid radial distance (meters)
    valid_range_max: 30                 

    # Minimum angle of the scan (radians, e.g. -π/4*3)
    angle_min: -2.356194496154785        

    # Maximum angle of the scan (radians, e.g. +π/4*3)
    angle_max: 2.338737726211548         

    # Total number of rays in each scan
    num_rays: 270                        

    # Factor to detect angular discontinuities (higher = more sensitive)
    discontinuity_factor: 10             

    # Discount factor for rolling differences across scans (0–1)
    rolling_diff_discount: 0.9           

    # Minimum number of rays required per partition
    min_partition_size: 5                

  # ----------------------------------------------------------
  # 3. Per‐ray GP settings
  # ----------------------------------------------------------
  gp:
    # Fully qualified C++ covariance class #TODO is there anything else outside of OU?
    kernel_type: "erl::covariance::OrnsteinUhlenbeck<float, 1>"

    # Yaml‐friendly name of the kernel’s Setting struct
    kernel_setting_type: "erl::covariance::Covariance<float>::Setting"

    # Hyperparameters for the 1D GP kernel:
    kernel:
      # Input dimension (1 because each ray is 1D distance)
      x_dim: 1                          
      # Kernel variance parameter (α)
      alpha: 1                          
      # Length‐scale parameter (meters)
      scale: 0.5                        
      # Mixing weight for composite kernels (1 = single kernel)
      scale_mix: 1                     
      # If mixing multiple kernels, list each weight; empty = single
      weights: []                     

    # Maximum number of training samples to keep in memory for the ray‐GP
    max_num_samples: 64              

  # ----------------------------------------------------------
  # 4. Mapping function & scaling
  # ----------------------------------------------------------
  mapping:
    # Type of occupancy‐to‐SDF mapping #TODO is there anything else?
    type: "kInverseSqrt"              

    # Scale factor applied after computing the raw mapping
    scale: 1                          

# ------------------------------------------------------------
# 5. Variance computation thresholds
# ------------------------------------------------------------
compute_variance:
  # If gradient variance < this, treat gradient as zero
  zero_gradient_position_var: 1      

  # If gradient variance < this, ignore gradient component
  zero_gradient_gradient_var: 1      

  # Minimum allowed variance on distance estimates (meters^2)
  min_distance_var: 1                

  # Maximum allowed variance on distance estimates (meters^2)
  max_distance_var: 100              

  # Coefficient α to inflate position variance as a function of distance
  position_var_alpha: 0.01           

  # Minimum allowed variance on gradient estimates
  min_gradient_var: 0.01             

  # Maximum allowed variance on gradient estimates
  max_gradient_var: 1                 

# ------------------------------------------------------------
# 6. Point‐update & occupancy thresholds
# ------------------------------------------------------------
update_map_points:
  # Minimum signed‐distance (negative) to consider a point “occupied”
  min_observable_occ: -0.08          

  # Maximum absolute signed‐distance to consider a point “surface”
  max_surface_abs_occ: 0.02          

  # If gradient variance > this, drop point as invalid
  max_valid_gradient_var: 0.8        

  # Maximum number of attempts to adjust point position if out_of_bounds
  max_adjust_tries: 5                

  # Maximum allowed posterior position variance for Bayes update
  max_bayes_position_var: 0.5       

  # Maximum allowed posterior gradient variance for Bayes update
  max_bayes_gradient_var: 0.3       

  # Minimum allowed position variance to keep point in map
  min_position_var: 0.001           

  # Minimum allowed gradient variance to keep point in map
  min_gradient_var: 0.001           

# ------------------------------------------------------------
# 7. Occupancy tree / Octree settings
# ------------------------------------------------------------
tree:
  # Maximum depth of the octree/quadtree (higher = finer resolution)
  tree_depth: 16                     

  # Minimum log‐odds value (clamp) at which a cell is considered free
  log_odd_min: -2                   

  # Maximum log‐odds value (clamp) at which a cell is considered occupied
  log_odd_max: 10                   

  # Log‐odds increment to apply on a “hit” (occupied measurement)
  log_odd_hit: 0.85                 

  # Log‐odds decrement to apply on a “miss” (free measurement)
  log_odd_miss: -0.4               

  # Threshold log‐odds to decide occupancy (≥ this ⇒ cell occupied)
  log_odd_occ_threshold: 0         

  # Base resolution of the leaf cells (meters)
  resolution: 0.1                   

  # If true, detect and flag tree changes for incremental updates
  use_change_detection: false       

  # If true, enforce an Axis‐Aligned Bounding Box (AABB) to limit mapping
  use_aabb_limit: false             

  aabb:
    # Center of the AABB (x, y) in world coordinates
    center: [0, 0]                   

    # Half‐sizes of the AABB in each axis (meters)
    half_sizes: [0, 0]               

# ------------------------------------------------------------
# 8. Clustering & surface resolution
# ------------------------------------------------------------
# Maximum depth used for clustering adjacent surface points (higher = coarser clusters)
cluster_depth: 14                    

# Resolution (meters) to sample continuous surfaces from discrete points
surface_resolution: 0.0              

# Global scale factor applied to all SDF values (1.0 = no scaling)
scaling: 1.0                         

# Small perturbation added to successive SDF predictions (to smooth over time)
perturb_delta: 0.001                 

# ------------------------------------------------------------
# 9. Gradient & occupancy toggles
# ------------------------------------------------------------
# If gradient magnitude < this threshold, treat as “zero gradient”
zero_gradient_threshold: 1.0e-10     

# If true, update occupancy values in the tree based on new measurements
update_occupancy: true               

```
### 2. SDF Mapping YAML.

```yaml 
# ------------------------------------------------------------
# Generic SDF‐Mapping Configuration YAML
# ------------------------------------------------------------

# ------------------------------------------------------------
# 1. Test‐time query settings 
# ------------------------------------------------------------
test_query:
  # Maximum allowed variance for a test‐point to be considered valid
  max_test_valid_distance_var: 1        

  # Half‐size of the search area (in meters); should cover at least the max SDF distance
  search_area_half_size: 4             

  # Number of neighboring GPs to query around each test point
  num_neighbor_gps: 4                   

  # If true, choose the smallest signed distance among neighbors as the prediction
  use_smallest: true                   

  # Whether to compute the SDF gradient at test points
  compute_gradient: true               

  # Whether to compute variance of the gradient at test points
  compute_gradient_variance: true      

  # Whether to compute full covariance at test points
  compute_covariance: false            

  # Whether to use the GP’s own covariance (instead of approximations)
  use_gp_covariance: false             

  # If true, automatically retrain any GP that has become outdated
  retrain_outdated: true               

  # If true, use a global buffer of points for queries (trades memory for speed)
  use_global_buffer: true              

# ------------------------------------------------------------
# 2. Parallelization & timing
# ------------------------------------------------------------

# Number of threads to use for GP updates and queries
num_threads: 64                       

# Desired update frequency (Hz) for mapping (i.e., how often to call Update(...) per second)
update_hz: 60                         

# ------------------------------------------------------------
# 3. Sensor and variance thresholds
# ------------------------------------------------------------

# Estimated standard deviation of sensor noise (meters)
sensor_noise: 0.01                    

# Scale factor for the GP area (must cover at least twice the max SDF if use_smallest=false)
gp_sdf_area_scale: 5                  

# Maximum allowed variance on the gradient to be considered valid
max_valid_gradient_var: 0.1           

# Maximum allowed variance on position before marking as invalid
invalid_position_var: 0.25            

# ------------------------------------------------------------
# 4. SDF‐GP specific settings
# ------------------------------------------------------------
sdf_gp:
  # Method for determining the sign of the SDF:
  #   "kSignGp" =  use sign
  #   "kNormalGp" = use a GP on normals,
  #   "kExternal" = use an external sign estimator,
  #   "kHybrid" = #TODO I DONT KNOW WHAT THIS IS
  sign_method: "kNormalGp"                     

  # If using a hybrid sign strategy, list the methods in priority order
  hybrid_sign_methods: 
    - "kNormalGp"
    - "kExternal"               

  # Threshold on the normal‐GP’s output to switch to the external method
  hybrid_sign_threshold: 0.2   

  # Scale factor for normal vector magnitudes (if using normal‐GP)
  normal_scale: 100.0         

  # Temperature parameter for soft‐minimum computations (if using softmin)
  softmin_temperature: 10.0    

  # Distance offset to avoid self‐collision when querying sign GP (meters)
  sign_gp_offset_distance: 0.01

  # Distance offset to avoid self‐collision when querying EDF GP (meters)
  edf_gp_offset_distance: 0.0

  # ------------------------------------------------------------
  # 4a. Sign‐GP settings
  # ------------------------------------------------------------
  sign_gp:
    # Fully‐qualified C++ class name of the covariance (e.g. a reduced‐rank Matern) #TODO WE NEED ALL OF THE KERNELS AND WHEN TO USE WHAT
    kernel_type: "erl::covariance::ReducedRankMatern32<double, 2>"

    # The Yamlable name for the corresponding kernel’s Setting struct 
    kernel_setting_type: "erl::covariance::ReducedRankCovariance<double>::Setting"

    # Hyperparameters for the kernel
    kernel:
      x_dim: 2                 # Input dimension (2 for 2D, 3 for 3D)
      alpha: 1                 # Kernel variance parameter
      scale: 2                 # Length‐scale parameter (meters)
      scale_mix: 1             # For mixture kernels (set to 1 if not mixing)
      weights: []              # If mixing multiple kernels, list weight for each; empty = single
      max_num_basis: -1        # Max inducing/basis points; -1 = unlimited
      num_basis: [10, 10]      # If reduced‐rank, basis grid size per dimension
      boundaries: [2.5, 2.5]   # Spatial range (per axis) for basis placement
      accumulated: true        # Whether to accumulate basis functions over time

    # Maximum number of training samples to store for this GP
    max_num_samples: 256    

    # If true, do not use gradient observations to train this GP
    no_gradient_observation: false  

  # ------------------------------------------------------------
  # 4b. EDF‐GP settings
  # ------------------------------------------------------------
  edf_gp:
    # Fully‐qualified C++ class name for the EDF GP covariance (e.g. Matern) #TODO SAME HERE
    kernel_type: "erl::covariance::Matern32<double, 2>"

    # The Yamlable name for the corresponding kernel’s Setting struct
    kernel_setting_type: "erl::covariance::Covariance<double>::Setting"

    # Hyperparameters for the EDF GP kernel
    kernel:
      x_dim: 2                 # Input dimension for EDF GP
      alpha: 1                 # Kernel variance parameter
      scale: 0.3               # Length‐scale parameter (meters)
      scale_mix: 1             # For mixture kernels (if any)
      weights: []              # Empty if single‐kernel

    # Log of the noise precision (λ = 1/σ²) for the EDF GP
    log_lambda: 40           

    # Maximum number of EDF GP training samples to keep in memory
    max_num_samples: 256    

    # If true, do not use gradient information when updating this GP
    no_gradient_observation: true  

```

 
## FAQ

### How to find scan range?

Through the terminal, find the scan topic:

```bash
rostopic info /path/to/scan/topc
```

There should be an output in the form of:

```bash
Type: sensor_msgs/message/type
```

Then use:

```bash 
rosmsg show sensor_msgs/message/type
```