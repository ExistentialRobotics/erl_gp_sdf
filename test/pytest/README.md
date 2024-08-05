PyTest for SDF Mapping
======================

# Load SDF Mapping 2D from Checkpoint

```bash
python test_gp_sdf_mapping_2d.py \
  --sdf-mapping-bin-file <path_to_sdf_mapping_replica.bin>
```

![](assets/gazebo-2d-sdf.png)

# Load SDF Mapping 3D from Checkpoint

```bash
python test_gp_sdf_mapping_3d.py \
  --sdf-mapping-bin-file <path_to_sdf_mapping_replica_lidar.bin> \
  --mesh ../../data/replica-hotel-0.ply
```

![](assets/test_gp_sdf_mapping_3d.png)
