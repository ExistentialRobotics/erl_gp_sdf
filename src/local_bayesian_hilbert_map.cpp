#include "erl_gp_sdf/local_bayesian_hilbert_map.hpp"

#include "erl_common/block_timer.hpp"
#include "erl_common/random.hpp"
#include "erl_geometry/logodd.hpp"

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Voxel::operator==(const Voxel &other) const {
        return surf_config == other.surf_config && edges == other.edges && faces == other.faces;
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Voxel::operator!=(const Voxel &other) const {
        return !(*this == other);
    }

    template<typename Dtype, int Dim>
    LocalBayesianHilbertMap<Dtype, Dim>::LocalBayesianHilbertMap(
        std::shared_ptr<LocalBayesianHilbertMapSetting<Dtype>> setting_,
        MatrixDX hinged_points,
        Aabb map_boundary,
        uint64_t seed,
        Aabb track_surface_boundary_)
        : setting(std::move(setting_)),
          tracked_surface_boundary(std::move(track_surface_boundary_)),
          tracked_surface_resolution(
              tracked_surface_boundary.sizes().array() /
              static_cast<Dtype>(setting->surface_grid_size)),
          bhm(setting->bhm,
              Covariance::CreateCovariance(setting->kernel_type, setting->kernel),
              std::move(hinged_points),
              std::move(map_boundary),
              seed) {
        if (setting->hit_point_buffer_size > 0) {
            hit_point_buffer.reserve(setting->hit_point_buffer_size);
            hit_point_ring_buffer = common::RingBuffer<VectorD>(setting->hit_point_buffer_size);
        }
        if (setting->ray_buffer_size > 0) {
            ray_info_buffer.reserve(setting->ray_buffer_size);
            ray_info_ring_buffer = common::RingBuffer<RayInfo>(setting->ray_buffer_size);
        }
        if (setting->max_dataset_size > 0) {
            dataset_points.resize(Dim, setting->max_dataset_size);
            dataset_labels.resize(setting->max_dataset_size);
        }
        surface_log_odds = setting->surface_log_odds;
        log_odds_count = setting->surface_log_odds_init_count;
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::GenerateDataset(
        const Eigen::Ref<const VectorD> &sensor_position,
        const Eigen::Ref<const MatrixDX> &points,
        const std::vector<long> &point_indices) {

        hit_indices.clear();
        if (points.cols() > 0) {
            // collect rays from the latest sensor data
            geometry::OccupancyMap<Dtype, Dim>::CollectRays(
                // input
                sensor_position,
                points,
                point_indices,
                bhm.GetSamplingBoundary(),
                setting->bhm->min_distance,
                setting->bhm->max_distance,
                setting->bhm->free_sampling_margin,
                setting->bhm->free_points_per_meter,
                // output
                hit_indices,       // cleared inside
                ray_info_buffer);  // append to the buffer
        }

        // move rays from the ring buffer to the ray info buffer
        if (!ray_info_ring_buffer.IsEmpty()) { ray_info_ring_buffer.PopAll(ray_info_buffer); }

        // check if the dataset size limit is exceeded.
        // if exceeded, adjust the number of points to sample.
        long total_num_hit_points = 0;
        long total_num_free_points = 0;
        for (const auto &ray: ray_info_buffer) {
            if (ray.hit_flag) { ++total_num_hit_points; }
            total_num_free_points += ray.num_free_points;
        }
        const long max_dataset_size = setting->max_dataset_size;
        const long max_num_points = total_num_free_points + total_num_hit_points;
        const bool limit_exceeded = max_dataset_size > 0 && max_num_points > max_dataset_size;
        long num_hit_to_sample, num_free_to_sample;
        if (limit_exceeded) {
            num_hit_to_sample = max_dataset_size * total_num_hit_points / max_num_points;
            num_free_to_sample = max_dataset_size * total_num_free_points / max_num_points;
        } else {
            num_hit_to_sample = total_num_hit_points;
            num_free_to_sample = total_num_free_points;
        }

        // generate the dataset from the rays
        MatrixDX *dataset_points_ptr = &dataset_points;
        VectorX *dataset_labels_ptr = &dataset_labels;
        long n_points = 0;
        MatrixDX extra_points;
        VectorX extra_labels;
        bool combine = (dataset_size > 0) && ((dataset_size < setting->min_dataset_size) ||
                                              (dataset_hit_size < setting->min_dataset_hit_size));
        // previous dataset was too small. combine it with new points.
        if (combine) {
            dataset_points_ptr = &extra_points;
            dataset_labels_ptr = &extra_labels;
        }
        const std::size_t n_rays_used = geometry::OccupancyMap<Dtype, Dim>::GenerateSamples(
            // input
            ray_info_buffer,
            bhm.GetRandomGenerator(),
            limit_exceeded,
            num_hit_to_sample,
            num_free_to_sample,
            // output
            num_hit_to_sample,
            n_points,
            *dataset_points_ptr,
            *dataset_labels_ptr);
        unused_ray_count = static_cast<long>(ray_info_buffer.size() - n_rays_used);

        // combine extra points if needed
        if (combine && n_points > 0) {
            // n_points <= max_dataset_size
            // dataset_size >= 0 && dataset_size < setting->min_dataset_size
            long new_dataset_size = dataset_size + n_points;
            if (max_dataset_size > 0 && new_dataset_size > max_dataset_size) {
                dataset_size = max_dataset_size - n_points;
                dataset_hit_size = dataset_labels.head(dataset_size).sum();
                new_dataset_size = max_dataset_size;
            }
            if (dataset_points.cols() < new_dataset_size) {
                dataset_points.conservativeResize(Dim, new_dataset_size);
                dataset_labels.conservativeResize(new_dataset_size);
            }
            dataset_points.block(0, dataset_size, Dim, n_points) = extra_points.leftCols(n_points);
            dataset_labels.segment(dataset_size, n_points) = extra_labels.head(n_points);
            dataset_hit_size += extra_labels.head(n_points).sum();
            dataset_size = new_dataset_size;
        } else if (n_points > 0) {
            dataset_size = n_points;
            dataset_hit_size = num_hit_to_sample;
        }  // else: no new points generated

        if (setting->ray_buffer_size > 0) {
            // move rays to the ring buffer.
            // we will re-use them in the next update.
            if (ray_info_buffer.size() > ray_info_ring_buffer.Capacity()) {
                // used rays are at the back of the buffer and removed
                // only when the buffer has no space to fit them.
                ray_info_buffer.resize(ray_info_ring_buffer.Capacity());
            }
            ray_info_ring_buffer.PushRange(ray_info_buffer.begin(), ray_info_buffer.end());
            ray_info_buffer.clear();
            unused_ray_count = std::min(unused_ray_count, setting->ray_buffer_size);
        } else {
            // remove used rays from the buffer
            ray_info_buffer.resize(ray_info_buffer.size() - n_rays_used);
        }
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::UpdateSurface(
        const Eigen::Ref<const MatrixDX> &points,
        const bool update_surface_voxels) {

        bool updated = false;
        const bool auto_surface_log_odds = setting->auto_surface_log_odds;
        if (points.cols() > 0 && !hit_indices.empty()) {
            const int gs = setting->surface_grid_size;
            const bool include_neighbor_voxels = setting->include_neighbor_voxels;
            // add hit points to the hit point buffer
            for (const auto &index: hit_indices) {
                VectorD point = points.col(index);
                if (auto_surface_log_odds) { hit_point_buffer.emplace_back(point); }
                if (!update_surface_voxels) { continue; }
                GridIndex voxel_coords;
                voxel_coords[Dim] = 0;  // edge coord, not used here
                if (!GetGridCoords(point, true, voxel_coords)) { continue; }
                if (surf_voxels.try_emplace(voxel_coords, Voxel{}).second) { updated = true; }
                if (!include_neighbor_voxels) { continue; }
                for (int i = 0; i < Dim; ++i) {
                    voxel_coords[i] += 1;
                    if (voxel_coords[i] < gs) { surf_voxels.try_emplace(voxel_coords, Voxel{}); }
                    voxel_coords[i] -= 2;
                    if (voxel_coords[i] >= 0) { surf_voxels.try_emplace(voxel_coords, Voxel{}); }
                    voxel_coords[i] += 1;
                }
            }
        }

        if (!auto_surface_log_odds) { return updated; }

        // move hit points from the ring buffer to the hit point buffer
        if (!hit_point_ring_buffer.IsEmpty()) { hit_point_ring_buffer.PopAll(hit_point_buffer); }

        // limit the number of hit points used to update the surface log-odds
        long n_hit_points = static_cast<long>(hit_point_buffer.size());
        n_hit_points = std::min(n_hit_points, setting->hit_point_buffer_size);
        if (n_hit_points == 0) { return updated; }

        // update surface log-odds value
        std::uniform_int_distribution<std::size_t> hit_point_distribution(
            0,
            hit_point_buffer.size() - 1);
        updated = true;
        surface_log_odds *= static_cast<Dtype>(log_odds_count);
        for (long i = 0; i < n_hit_points; ++i) {
            std::size_t idx1 = hit_point_distribution(bhm.GetRandomGenerator());
            idx1 %= (hit_point_buffer.size() - i);
            std::size_t idx2 = hit_point_buffer.size() - 1 - i;
            std::swap(hit_point_buffer[idx1], hit_point_buffer[idx2]);

            Dtype log_odd = 0.0f;
            VectorD gradient;
            bhm.Predict(
                hit_point_buffer[idx2],
                true /*logodd*/,
                setting->faster_prediction,
                false /*compute_gradient*/,
                false /*gradient_with_sigmoid*/,
                log_odd,
                gradient);
            surface_log_odds += log_odd;
        }
        log_odds_count += n_hit_points;
        surface_log_odds /= static_cast<Dtype>(log_odds_count);

        if (setting->hit_point_buffer_size > 0) {
            // move hit points to the ring buffer.
            // we will re-use them in the next update.
            if (hit_point_buffer.size() > hit_point_ring_buffer.Capacity()) {
                // used hit points are at the back of the buffer and removed only when the buffer
                // has no space to fit them.
                hit_point_buffer.resize(hit_point_ring_buffer.Capacity());
            }
            hit_point_ring_buffer.PushRange(hit_point_buffer.begin(), hit_point_buffer.end());
            hit_point_buffer.clear();
        } else {
            // remove used hit points from the buffer
            hit_point_buffer.resize(hit_point_buffer.size() - n_hit_points);
        }

        if (surface_log_odds < setting->surface_log_odds_min ||
            surface_log_odds > setting->surface_log_odds_max) {
            active = false;
        }

        return updated;
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Update(
        const Eigen::Ref<const VectorD> &sensor_origin,
        const Eigen::Ref<const MatrixDX> &points,
        const std::vector<long> &point_indices,
        const bool update_surface_voxels) {

        active = true;  // assume the map is valid first
        GenerateDataset(sensor_origin, points, point_indices);

        bool updated = false;
        if (dataset_size >= setting->min_dataset_size &&
            dataset_hit_size >= setting->min_dataset_hit_size) {
            updated = true;
            bhm.RunExpectationMaximization(dataset_points, dataset_labels, dataset_size);
        }

        updated |= UpdateSurface(points, update_surface_voxels);

        return updated;
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::GetGridCoords(
        const Eigen::Ref<const VectorD> &point,
        const bool check_bounds,
        GridIndex &grid_coords) const {

        using namespace common;

        const auto &map_min = tracked_surface_boundary.min();
        const auto &map_res = tracked_surface_resolution;
        const int grid_size = setting->surface_grid_size;

        if (check_bounds) {
            for (long d = 0; d < Dim; ++d) {
                grid_coords[d] = MeterToGrid<Dtype>(point[d], map_min[d], map_res[d]);
                if (grid_coords[d] < 0 || grid_coords[d] >= grid_size) { return false; }
            }
        } else {
            for (long d = 0; d < Dim; ++d) {
                grid_coords[d] = MeterToGrid<Dtype>(point[d], map_min[d], map_res[d]);
            }
        }

        return true;
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::Predict(
        const Eigen::Ref<const MatrixDX> &points,
        const bool logodd,
        const bool compute_free_space,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        VectorX &prob_occupied,
        Eigen::VectorXb &in_free_space,
        MatrixDX &gradient) const {
        bhm.Predict(
            points,
            logodd,
            setting->faster_prediction,
            compute_gradient,
            gradient_with_sigmoid,
            parallel,
            prob_occupied,
            gradient);
        if (!compute_free_space) { return; }
        if (logodd) {
            in_free_space = prob_occupied.array() < surface_log_odds;
        } else {
            Dtype p = geometry::logodd::Probability(surface_log_odds);
            in_free_space = prob_occupied.array() < p;
        }
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::PredictAt(
        const VectorD &point,
        bool logodd,
        bool compute_free_space,
        bool compute_gradient,
        bool gradient_with_sigmoid,
        Dtype &prob_occupied,
        bool &in_free_space,
        VectorD &gradient) const {
        bhm.Predict(
            point,
            logodd,
            setting->faster_prediction,
            compute_gradient,
            gradient_with_sigmoid,
            prob_occupied,
            gradient);
        if (!compute_free_space) { return; }
        if (logodd) {
            in_free_space = prob_occupied < surface_log_odds;
        } else {
            Dtype p = geometry::logodd::Probability(surface_log_odds);
            in_free_space = prob_occupied < p;
        }
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::PredictGradient(
        const Eigen::Ref<const MatrixDX> &points,
        bool with_sigmoid,
        bool parallel,
        MatrixDX &gradient) const {
        bhm.PredictGradient(points, setting->faster_prediction, with_sigmoid, parallel, gradient);
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Write(std::ostream &stream) const {
        using namespace common;
        using namespace common::serialization;
        static const TokenWriteFunctionPairs<LocalBayesianHilbertMap> token_function_pairs = {
            // setting is loaded externally.
            // tracked_surface_boundary is loaded externally.
            // tracked_surface_resolution is computed by the constructor.
            {
                "bhm",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    return self->bhm.Write(s) && s.good();
                },
            },
            // `strides` is computed by the constructor.
            {
                "surface_indices",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    const std::size_t n = self->surface_indices.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    for (const auto &[index, buf_idx]: self->surface_indices) {
                        s.write(reinterpret_cast<const char *>(&index), sizeof(index));
                        s.write(reinterpret_cast<const char *>(&buf_idx), sizeof(buf_idx));
                    }
                    return s.good();
                },
            },
            {
                "surf_voxels",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    const std::size_t n_voxels = self->surf_voxels.size();
                    s.write(reinterpret_cast<const char *>(&n_voxels), sizeof(n_voxels));
                    for (const auto &[index, voxel]: self->surf_voxels) {
                        s.write(reinterpret_cast<const char *>(&index), sizeof(index));
                        s.write(reinterpret_cast<const char *>(&voxel.good), sizeof(voxel.good));
                        s.write(
                            reinterpret_cast<const char *>(&voxel.surf_config),
                            sizeof(voxel.surf_config));
                        // write edges
                        const std::size_t n_edges = voxel.edges.size();
                        s.write(reinterpret_cast<const char *>(&n_edges), sizeof(n_edges));
                        for (const auto &edge: voxel.edges) {
                            for (long i = 0; i < edge.size(); ++i) {
                                s.write(reinterpret_cast<const char *>(&edge[i]), sizeof(edge[i]));
                            }
                        }
                        // write faces
                        const std::size_t n_faces = voxel.faces.size();
                        s.write(reinterpret_cast<const char *>(&n_faces), sizeof(n_faces));
                        for (const auto &face: voxel.faces) {
                            for (long i = 0; i < face.size(); ++i) {
                                s.write(reinterpret_cast<const char *>(&face[i]), sizeof(face[i]));
                            }
                        }
                    }
                    return s.good();
                },
            },
            {
                "dataset_hit_size",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    s.write(
                        reinterpret_cast<const char *>(&self->dataset_hit_size),
                        sizeof(self->dataset_hit_size));
                    return s.good();
                },
            },
            {
                "dataset_size",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    s.write(
                        reinterpret_cast<const char *>(&self->dataset_size),
                        sizeof(self->dataset_size));
                    return s.good();
                },
            },
            {
                "dataset_points",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    return SaveEigenMatrixToBinaryStream(s, self->dataset_points) && s.good();
                },
            },
            {
                "dataset_labels",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    return SaveEigenMatrixToBinaryStream(s, self->dataset_labels) && s.good();
                },
            },
            {
                "hit_indices",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    const std::size_t n = self->hit_indices.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    s.write(
                        reinterpret_cast<const char *>(self->hit_indices.data()),
                        sizeof(long) * n);
                    return s.good();
                },
            },
            {
                "hit_point_buffer",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    const std::size_t n = self->hit_point_buffer.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    if (n == 0) { return s.good(); }
                    s.write(
                        reinterpret_cast<const char *>(self->hit_point_buffer.data()),
                        sizeof(VectorD) * n);
                    return s.good();
                },
            },
            {
                "ray_info_buffer",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    const std::size_t n = self->ray_info_buffer.size();
                    s.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    if (n == 0) { return s.good(); }
                    s.write(
                        reinterpret_cast<const char *>(self->ray_info_buffer.data()),
                        sizeof(RayInfo) * n);
                    return s.good();
                },
            },
            {
                "hit_point_ring_buffer",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    return self->hit_point_ring_buffer.Write(s) && s.good();
                },
            },
            {
                "ray_info_ring_buffer",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    return self->ray_info_ring_buffer.Write(s) && s.good();
                },
            },
            {
                "unused_ray_count",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    s.write(
                        reinterpret_cast<const char *>(&self->unused_ray_count),
                        sizeof(self->unused_ray_count));
                    return s.good();
                },
            },
            {
                "active",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    s.write(reinterpret_cast<const char *>(&self->active), sizeof(bool));
                    return s.good();
                },
            },
            // surf_data_cache is temporary.
            {
                "surface_log_odds",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    s.write(
                        reinterpret_cast<const char *>(&self->surface_log_odds),
                        sizeof(self->surface_log_odds));
                    return s.good();
                },
            },
            {
                "log_odds_count",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    s.write(
                        reinterpret_cast<const char *>(&self->log_odds_count),
                        sizeof(self->log_odds_count));
                    return s.good();
                },
            },
        };
        return WriteTokens(stream, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Read(std::istream &stream) {
        using namespace common;
        using namespace common::serialization;
        static const TokenReadFunctionPairs<LocalBayesianHilbertMap> token_function_pairs = {
            // setting is loaded externally.
            // tracked_surface_boundary is loaded externally.
            // tracked_surface_resolution is computed by the constructor.
            {
                "bhm",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    return self->bhm.Read(s) && s.good();
                },
            },
            // `strides` is computed by the constructor.
            {
                "surface_indices",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(n));
                    self->surface_indices.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        GridIndex index;
                        s.read(reinterpret_cast<char *>(&index), sizeof(index));
                        std::size_t buf_idx;
                        s.read(reinterpret_cast<char *>(&buf_idx), sizeof(buf_idx));
                        if (!self->surface_indices.try_emplace(index, buf_idx).second) {
                            ERL_WARN("Duplicate surface_indices index: {}.", index);
                            return false;
                        }
                    }
                    return s.good();
                },
            },
            {
                "surf_voxels",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    std::size_t n_voxels;
                    s.read(reinterpret_cast<char *>(&n_voxels), sizeof(n_voxels));
                    if (n_voxels == 0) {
                        self->surf_voxels.clear();
                        return s.good();
                    }
                    self->surf_voxels.reserve(n_voxels);
                    for (std::size_t i = 0; i < n_voxels; ++i) {
                        GridIndex index;
                        s.read(reinterpret_cast<char *>(&index), sizeof(index));
                        if (self->surf_voxels.contains(index)) {
                            ERL_WARN("Duplicate surf_voxels index: {}.", index);
                            return false;
                        }
                        Voxel voxel;
                        s.read(reinterpret_cast<char *>(&voxel.good), sizeof(voxel.good));
                        s.read(
                            reinterpret_cast<char *>(&voxel.surf_config),
                            sizeof(voxel.surf_config));
                        // read edges
                        std::size_t n_edges = 0;
                        s.read(reinterpret_cast<char *>(&n_edges), sizeof(n_edges));
                        if (n_edges == 0) {
                            voxel.edges.clear();
                        } else {
                            voxel.edges.reserve(n_edges);
                            for (std::size_t j = 0; j < n_edges; ++j) {
                                GridIndex edge;
                                for (long k = 0; k < edge.size(); ++k) {
                                    s.read(reinterpret_cast<char *>(&edge[k]), sizeof(edge[k]));
                                }
                                voxel.edges.push_back(edge);
                            }
                        }
                        // read faces
                        std::size_t n_faces = 0;
                        s.read(reinterpret_cast<char *>(&n_faces), sizeof(n_faces));
                        if (n_faces == 0) {
                            voxel.faces.clear();
                        } else {
                            voxel.faces.reserve(n_faces);
                            for (std::size_t j = 0; j < n_faces; ++j) {
                                Face face;
                                for (long k = 0; k < face.size(); ++k) {
                                    s.read(reinterpret_cast<char *>(&face[k]), sizeof(face[k]));
                                }
                                voxel.faces.push_back(face);
                            }
                        }
                        // insert it
                        self->surf_voxels[index] = std::move(voxel);
                    }
                    return s.good();
                },
            },
            {
                "dataset_hit_size",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    s.read(
                        reinterpret_cast<char *>(&self->dataset_hit_size),
                        sizeof(self->dataset_hit_size));
                    return s.good();
                },
            },
            {
                "dataset_size",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    s.read(
                        reinterpret_cast<char *>(&self->dataset_size),
                        sizeof(self->dataset_size));
                    return s.good();
                },
            },
            {
                "dataset_points",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    return LoadEigenMatrixFromBinaryStream(s, self->dataset_points) && s.good();
                },
            },
            {
                "dataset_labels",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    return LoadEigenMatrixFromBinaryStream(s, self->dataset_labels) && s.good();
                },
            },
            {
                "hit_indices",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(n));
                    if (n == 0) {
                        self->hit_indices.clear();
                        return s.good();
                    }
                    self->hit_indices.resize(n);
                    s.read(
                        reinterpret_cast<char *>(self->hit_indices.data()),
                        static_cast<std::streamsize>(sizeof(long) * n));
                    return s.good();
                },
            },
            {
                "hit_point_buffer",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(n));
                    if (n == 0) {
                        self->hit_point_buffer.clear();
                        return s.good();
                    }
                    self->hit_point_buffer.resize(n);
                    s.read(
                        reinterpret_cast<char *>(self->hit_point_buffer.data()),
                        sizeof(VectorD) * n);
                    return s.good();
                },
            },
            {
                "ray_info_buffer",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    std::size_t n;
                    s.read(reinterpret_cast<char *>(&n), sizeof(n));
                    if (n == 0) {
                        self->ray_info_buffer.clear();
                        return s.good();
                    }
                    self->ray_info_buffer.resize(n);
                    s.read(
                        reinterpret_cast<char *>(self->ray_info_buffer.data()),
                        sizeof(RayInfo) * n);
                    return s.good();
                },
            },
            {
                "hit_point_ring_buffer",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    return self->hit_point_ring_buffer.Read(s) && s.good();
                },
            },
            {
                "ray_info_ring_buffer",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    return self->ray_info_ring_buffer.Read(s) && s.good();
                },
            },
            {
                "unused_ray_count",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    s.read(
                        reinterpret_cast<char *>(&self->unused_ray_count),
                        sizeof(self->unused_ray_count));
                    return s.good();
                },
            },
            {
                "active",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    s.read(reinterpret_cast<char *>(&self->active), sizeof(bool));
                    return s.good();
                },
            },
            {
                "surface_log_odds",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    s.read(
                        reinterpret_cast<char *>(&self->surface_log_odds),
                        sizeof(self->surface_log_odds));
                    return s.good();
                },
            },
            {
                "log_odds_count",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    s.read(
                        reinterpret_cast<char *>(&self->log_odds_count),
                        sizeof(self->log_odds_count));
                    return s.good();
                },
            },
        };
        return ReadTokens(stream, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::operator==(const LocalBayesianHilbertMap &other) const {
        if (setting == nullptr && other.setting != nullptr) { return false; }
        if (setting != nullptr && (other.setting == nullptr || *setting != *other.setting)) {
            return false;
        }
        if (tracked_surface_boundary != other.tracked_surface_boundary) { return false; }
        if (bhm != other.bhm) { return false; }
        if (surface_indices != other.surface_indices) { return false; }
        if (surf_voxels != other.surf_voxels) { return false; }
        if (dataset_size != other.dataset_size) { return false; }
        if (!common::SafeEigenMatrixEqual(dataset_points, other.dataset_points)) { return false; }
        if (!common::SafeEigenMatrixEqual(dataset_labels, other.dataset_labels)) { return false; }
        if (hit_indices != other.hit_indices) { return false; }
        if (hit_point_buffer != other.hit_point_buffer) { return false; }
        if (ray_info_buffer != other.ray_info_buffer) { return false; }
        if (hit_point_ring_buffer != other.hit_point_ring_buffer) { return false; }
        if (ray_info_ring_buffer != other.ray_info_ring_buffer) { return false; }
        if (unused_ray_count != other.unused_ray_count) { return false; }
        if (active != other.active) { return false; }
        if (surface_log_odds != other.surface_log_odds) { return false; }
        if (log_odds_count != other.log_odds_count) { return false; }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::operator!=(const LocalBayesianHilbertMap &other) const {
        return !(*this == other);
    }

    template class LocalBayesianHilbertMapSetting<float>;
    template class LocalBayesianHilbertMapSetting<double>;
    template class LocalBayesianHilbertMap<float, 2>;
    template class LocalBayesianHilbertMap<float, 3>;
    template class LocalBayesianHilbertMap<double, 2>;
    template class LocalBayesianHilbertMap<double, 3>;
}  // namespace erl::gp_sdf
