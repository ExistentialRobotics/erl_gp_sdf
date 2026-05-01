#include "erl_gp_sdf/local_bayesian_hilbert_map.hpp"

#include "erl_common/block_timer.hpp"
#include "erl_common/random.hpp"
#include "erl_geometry/logodd.hpp"

#include <absl/container/flat_hash_set.h>

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Voxel::operator==(const Voxel &other) const {
        return surf_config == other.surf_config && color == other.color &&
               color_count == other.color_count && edges == other.edges && faces == other.faces;
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Voxel::operator!=(const Voxel &other) const {
        return !(*this == other);
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Voxel::Write(std::ostream &stream) const {
        stream.write(reinterpret_cast<const char *>(&good), sizeof(good));
        stream.write(reinterpret_cast<const char *>(&neighbors_added), sizeof(neighbors_added));
        stream.write(reinterpret_cast<const char *>(&surf_config), sizeof(surf_config));
        stream.write(reinterpret_cast<const char *>(color.data()), color.size());
        stream.write(reinterpret_cast<const char *>(&color_count), sizeof(color_count));
        // write edges
        const std::size_t n_edges = edges.size();
        stream.write(reinterpret_cast<const char *>(&n_edges), sizeof(n_edges));
        for (const auto &edge: edges) {
            for (long i = 0; i < edge.size(); ++i) {
                stream.write(reinterpret_cast<const char *>(&edge[i]), sizeof(edge[i]));
            }
        }
        // write faces
        const std::size_t n_faces = faces.size();
        stream.write(reinterpret_cast<const char *>(&n_faces), sizeof(n_faces));
        for (const auto &face: faces) {
            for (long i = 0; i < face.size(); ++i) {
                stream.write(reinterpret_cast<const char *>(&face[i]), sizeof(face[i]));
            }
        }
        return stream.good();
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Voxel::Read(std::istream &stream) {
        stream.read(reinterpret_cast<char *>(&good), sizeof(good));
        stream.read(reinterpret_cast<char *>(&neighbors_added), sizeof(neighbors_added));
        stream.read(reinterpret_cast<char *>(&surf_config), sizeof(surf_config));
        stream.read(reinterpret_cast<char *>(color.data()), color.size());
        stream.read(reinterpret_cast<char *>(&color_count), sizeof(color_count));
        // read edges
        std::size_t n_edges = 0;
        stream.read(reinterpret_cast<char *>(&n_edges), sizeof(n_edges));
        if (n_edges == 0) {
            edges.clear();
        } else {
            edges.reserve(n_edges);
            for (std::size_t j = 0; j < n_edges; ++j) {
                GridIndex edge;
                for (long k = 0; k < edge.size(); ++k) {
                    stream.read(reinterpret_cast<char *>(&edge[k]), sizeof(edge[k]));
                }
                edges.push_back(edge);
            }
        }
        // read faces
        std::size_t n_faces = 0;
        stream.read(reinterpret_cast<char *>(&n_faces), sizeof(n_faces));
        if (n_faces == 0) {
            faces.clear();
        } else {
            faces.reserve(n_faces);
            for (std::size_t j = 0; j < n_faces; ++j) {
                Face face;
                for (long k = 0; k < face.size(); ++k) {
                    stream.read(reinterpret_cast<char *>(&face[k]), sizeof(face[k]));
                }
                faces.push_back(face);
            }
        }
        return stream.good();
    }

    template<typename Dtype, int Dim>
    LocalBayesianHilbertMap<Dtype, Dim>::LocalBayesianHilbertMap(
        const std::size_t id_,
        std::shared_ptr<LocalBayesianHilbertMapSetting<Dtype>> setting_,
        MatrixDX hinged_points,
        Aabb map_boundary,
        uint64_t seed,
        Aabb track_surface_boundary_)
        : id(id_),
          setting(std::move(setting_)),
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
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::GenerateDataset(
        const Eigen::Ref<const VectorD> &sensor_position,
        const Eigen::Ref<const MatrixDX> &points,
        const bool collect_rays_only,
        std::vector<long> &point_indices) {

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
                max_used_ray_count * 2,
                bhm.GetRandomGenerator(),
                // output
                hit_indices,       // cleared inside
                ray_info_buffer);  // append to the buffer
        }

        if (collect_rays_only) {
            // only collect rays, do not generate dataset
            if (setting->ray_buffer_size > 0) {
                // move rays to the ring buffer.
                // we will re-use them in the next update.
                if (ray_info_buffer.size() + ray_info_ring_buffer.Size() >
                    ray_info_ring_buffer.Capacity()) {
                    ray_info_ring_buffer.PopAll(ray_info_buffer);
                    std::shuffle(
                        ray_info_buffer.begin(),
                        ray_info_buffer.end(),
                        bhm.GetRandomGenerator());
                    ray_info_buffer.resize(ray_info_ring_buffer.Capacity());
                }
                ray_info_ring_buffer.PushRange(ray_info_buffer.begin(), ray_info_buffer.end());
                ray_info_buffer.clear();
                unused_ray_count = std::min(unused_ray_count, setting->ray_buffer_size);
            }
            return;
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
        const long min_dataset_size = setting->min_dataset_size;
        const long min_dataset_hit_size = setting->min_dataset_hit_size;
        const long max_num_points = total_num_free_points + total_num_hit_points;
        const bool limit_exceeded = max_dataset_size > 0 && max_num_points > max_dataset_size;
        const bool below_min_size =
            max_num_points < min_dataset_size || total_num_hit_points < min_dataset_hit_size;
        long num_hit_to_sample = 0;
        long num_free_to_sample = 0;
        if (limit_exceeded) {
            num_hit_to_sample = max_dataset_size * total_num_hit_points / max_num_points;
            num_free_to_sample = max_dataset_size * total_num_free_points / max_num_points;
        } else if (below_min_size) {
            // impossible to reach the minimum dataset size, skip sampling
            num_hit_to_sample = 0;
            num_free_to_sample = 0;
        } else {
            num_hit_to_sample = total_num_hit_points;
            num_free_to_sample = total_num_free_points;
        }

        // generate the dataset from the rays
        std::size_t n_rays_used = 0;
        if (num_hit_to_sample > 0 && num_free_to_sample > 0) {
            n_rays_used = geometry::OccupancyMap<Dtype, Dim>::GenerateSamples(
                // input
                ray_info_buffer,
                bhm.GetRandomGenerator(),
                limit_exceeded,
                num_hit_to_sample,
                num_free_to_sample,
                // output
                dataset_hit_size,
                dataset_size,
                dataset_points,
                dataset_labels);
            max_used_ray_count = std::max(max_used_ray_count, n_rays_used);
            unused_ray_count = static_cast<long>(ray_info_buffer.size() - n_rays_used);
        }

        if (points.cols() == 0) {                      // no new points, only use cached rays
            ray_info_buffer.resize(unused_ray_count);  // remove used rays (exhausting mode)
        }

        const std::size_t size = ray_info_buffer.size();
        std::size_t j = size - 1;
        for (std::size_t i = size - n_rays_used; i <= j && i < size;) {
            if (!ray_info_buffer[i].hit_flag) {
                std::swap(ray_info_buffer[i], ray_info_buffer[j]);
                --j;
                continue;
            }
            ++i;
        }
        ray_info_buffer.resize(j + 1);  // remove used pass-through rays

        if (setting->ray_buffer_size > 0) {
            // move rays to the ring buffer.
            // we will re-use them in the next update.
            if (ray_info_buffer.size() > ray_info_ring_buffer.Capacity()) {
                // scheme1: used rays are at the back of the buffer and removed
                // scheme2: randomly remove rays
                // we use scheme2 to avoid biasing the rays kept in the buffer
                std::shuffle(
                    ray_info_buffer.begin(),
                    ray_info_buffer.end(),
                    bhm.GetRandomGenerator());
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

            auto add_neighbor_voxels = [this, &gs, &updated](GridIndex voxel_coords) {
                static const Eigen::Matrix<long, Dim, Dim == 2 ? 8 : 26> neighbor_offsets =
                    common::GetGridNeighborOffsets<long, Dim>(true);
                static const long n_neighbors = neighbor_offsets.cols();

                GridIndex coords;
                coords[Dim] = 0;  // edge coord, not used here
                for (long i = 0; i < n_neighbors; ++i) {
                    const long *offset = neighbor_offsets.col(i).data();
                    bool valid = true;
                    for (int d = 0; d < Dim; ++d) {
                        coords[d] = voxel_coords[d] + offset[d];
                        if (coords[d] < 0 || coords[d] >= gs) {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid) { continue; }
                    updated |= surf_voxels.try_emplace(coords, Voxel{}).second;
                }
            };

            absl::flat_hash_set<GridIndex> voxels_to_add_neighbors;

            // add hit points to the hit point buffer
            for (const auto &index: hit_indices) {
                const VectorD point = points.col(index);
                if (auto_surface_log_odds) { hit_point_buffer.emplace_back(point); }
                if (!update_surface_voxels) { continue; }
                GridIndex voxel_coords;
                voxel_coords[Dim] = 0;  // edge coord, not used here

                // check if the point is within the tracked surface boundary
                if (!GetGridCoords(point, true, voxel_coords)) { continue; }

                // if new voxel, mark as updated
                auto [iter, inserted] = surf_voxels.try_emplace(voxel_coords, Voxel{});
                updated |= inserted;

                // add neighbor voxels later
                if (include_neighbor_voxels && !iter->second.neighbors_added) {
                    voxels_to_add_neighbors.insert(voxel_coords);
                    iter->second.neighbors_added = true;
                }
            }
            if (update_surface_voxels && include_neighbor_voxels) {

                for (const auto &[voxel_coords, voxel]: surf_voxels) {
                    if (!voxel.good) { continue; }            // only check good voxels
                    if (voxel.neighbors_added) { continue; }  // already added
                    voxels_to_add_neighbors.insert(voxel_coords);
                }

                for (const auto &voxel_coords: voxels_to_add_neighbors) {
                    add_neighbor_voxels(voxel_coords);
                }
            }
        }

        if (!auto_surface_log_odds) { return updated; }

        // move hit points from the ring buffer to the hit point buffer
        if (!hit_point_ring_buffer.IsEmpty()) { hit_point_ring_buffer.PopAll(hit_point_buffer); }

        // limit the number of hit points used to update the surface log-odds
        long n_hit_points = static_cast<long>(hit_point_buffer.size());
        n_hit_points = std::min(n_hit_points, setting->surface_log_odds_num_points);
        if (n_hit_points == 0) { return updated; }

        // update surface log-odds value
        std::uniform_int_distribution<std::size_t> hit_point_distribution(
            0,
            hit_point_buffer.size() - 1);
        updated = true;
        const Dtype valid_surf_log_odds_max = 10.0f * setting->surface_log_odds_max;
        const Dtype valid_surf_log_odds_min = -valid_surf_log_odds_max;
        // surface_log_odds *= static_cast<Dtype>(log_odds_count);
        Dtype new_surface_log_odds = 0;
        long n_valid_count = 0;
        for (long i = 0; i < n_hit_points; ++i) {
            std::size_t idx1 = hit_point_distribution(bhm.GetRandomGenerator());
            idx1 %= (hit_point_buffer.size() - i);
            const std::size_t idx2 = hit_point_buffer.size() - 1 - i;
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
            if (log_odd < valid_surf_log_odds_min || log_odd > valid_surf_log_odds_max) {
                continue;  // skip outliers
            }
            // surface_log_odds += log_odd;
            // ++log_odds_count;
            new_surface_log_odds += log_odd;
            ++n_valid_count;
        }
        if (n_valid_count == 0) { return updated; }  // failed to get new surface log-odds
        new_surface_log_odds /= static_cast<Dtype>(n_valid_count);  // average log-odds
        const Dtype t = setting->surface_log_odds_lr;               // learning rate
        surface_log_odds = t * new_surface_log_odds + (1 - t) * surface_log_odds;
        log_odds_count += n_valid_count;  // update sample count (just for record)
        // surface_log_odds /= static_cast<Dtype>(n_hit_points);
        // surface_log_odds /= static_cast<Dtype>(log_odds_count);

        if (setting->hit_point_buffer_size > 0) {
            // move hit points to the ring buffer.
            // we will re-use them in the next update.
            if (hit_point_buffer.size() > hit_point_ring_buffer.Capacity()) {
                // used hit points are at the back of the buffer and removed only when the buffer
                // has no space to fit them.
                std::shuffle(
                    hit_point_buffer.begin(),
                    hit_point_buffer.end(),
                    bhm.GetRandomGenerator());
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
        const bool collect_rays_only,
        const bool update_surface_voxels,
        std::vector<long> &point_indices) {

        GenerateDataset(sensor_origin, points, collect_rays_only, point_indices);

        if (collect_rays_only && setting->auto_surface_log_odds) {
            for (const auto &index: hit_indices) {
                hit_point_buffer.emplace_back(points.col(index));
            }
            if (setting->hit_point_buffer_size > 0) {
                // move hit points to the ring buffer.
                // we will re-use them in the next update.
                if (hit_point_buffer.size() > hit_point_ring_buffer.Capacity()) {
                    // used hit points are at the back of the buffer and removed only when the
                    // buffer has no space to fit them.
                    hit_point_buffer.resize(hit_point_ring_buffer.Capacity());
                }
                hit_point_ring_buffer.PushRange(hit_point_buffer.begin(), hit_point_buffer.end());
                hit_point_buffer.clear();
            }
            return false;
        }

        bool updated = false;
        if (dataset_size >= setting->min_dataset_size &&
            dataset_hit_size >= setting->min_dataset_hit_size) {
            active = true;  // map will be updated, set active to true
            updated = true;
            bhm.RunExpectationMaximization(dataset_points, dataset_labels, dataset_size);
        }

        if (updated || points.cols() > 0) {
            updated |= UpdateSurface(points, update_surface_voxels);
        }

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
                grid_coords[d] = MeterToGrid<Dtype, long>(point[d], map_min[d], map_res[d]);
                if (grid_coords[d] < 0 || grid_coords[d] >= grid_size) { return false; }
            }
        } else {
            for (long d = 0; d < Dim; ++d) {
                grid_coords[d] = MeterToGrid<Dtype, long>(point[d], map_min[d], map_res[d]);
            }
        }

        return true;
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::PaintVoxel(
        const Eigen::Ref<const VectorD> &point,
        const std::array<uint8_t, 4> &color,
        const bool overwrite) {

        GridIndex voxel_coords;
        voxel_coords[Dim] = 0;  // edge coord, not used here
        if (!GetGridCoords(point, true, voxel_coords)) { return false; }

        // auto iter = surf_voxels.find(voxel_coords);
        // if (iter == surf_voxels.end()) { return false; }
        // Voxel &v = iter->second;
        Voxel &v = surf_voxels[voxel_coords];  // create voxel if not exist

        if (overwrite || v.color_count == 0) {
            v.color = color;
            v.color_count = 1;
        } else {
            const auto n = static_cast<uint64_t>(v.color_count);
            for (int c = 0; c < 4; ++c) {
                v.color[c] = static_cast<uint8_t>(
                    (static_cast<uint64_t>(v.color[c]) * n + static_cast<uint64_t>(color[c])) /
                    (n + 1));
            }
            if (v.color_count != UINT32_MAX) { ++v.color_count; }
        }

        // Currently, we do not consider color changes as surface update.
        // ++surface_update_timestamp;
        return true;
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::Predict(
        const Eigen::Ref<const MatrixDX> &points,
        const bool logodd,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        VectorX &prob_occupied,
        MatrixDX &gradient) const {
        bhm.Predict(
            points,
            /*logodd=*/true,
            setting->faster_prediction,
            compute_gradient,
            gradient_with_sigmoid,
            parallel,
            prob_occupied,
            gradient);
        prob_occupied.array() -= surface_log_odds;  // convert to relative log-odds
        if (!logodd) {
            for (long i = 0; i < prob_occupied.size(); ++i) {
                prob_occupied[i] = geometry::logodd::Probability(prob_occupied[i]);
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::PredictAt(
        const VectorD &point,
        const bool logodd,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        Dtype &prob_occupied,
        VectorD &gradient) const {
        bhm.Predict(
            point,
            /*logodd=*/true,
            setting->faster_prediction,
            compute_gradient,
            gradient_with_sigmoid,
            prob_occupied,
            gradient);
        prob_occupied -= surface_log_odds;  // convert to relative log-odds
        if (!logodd) { prob_occupied = geometry::logodd::Probability(prob_occupied); }
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
                        if (!voxel.Write(s)) { return false; }
                    }
                    return s.good();
                },
            },
            // num_faces can be computed from surf_voxels.
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
                "max_used_ray_count",
                [](const LocalBayesianHilbertMap *self, std::ostream &s) {
                    s.write(
                        reinterpret_cast<const char *>(&self->max_used_ray_count),
                        sizeof(self->max_used_ray_count));
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
                    std::size_t n = 0;
                    s.read(reinterpret_cast<char *>(&n), sizeof(n));
                    self->surface_indices.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        GridIndex index;
                        s.read(reinterpret_cast<char *>(&index), sizeof(index));
                        std::size_t buf_idx = 0;
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
                    std::size_t n_voxels = 0;
                    s.read(reinterpret_cast<char *>(&n_voxels), sizeof(n_voxels));
                    if (n_voxels == 0) {
                        self->surf_voxels.clear();
                        return s.good();
                    }
                    self->surf_voxels.reserve(n_voxels);
                    self->num_faces = 0;
                    for (std::size_t i = 0; i < n_voxels; ++i) {
                        GridIndex index;
                        s.read(reinterpret_cast<char *>(&index), sizeof(index));
                        auto [it, inserted] = self->surf_voxels.try_emplace(index, Voxel{});
                        if (!inserted) {
                            ERL_WARN("Duplicate surf_voxels index: {}.", index);
                            return false;
                        }
                        if (!it->second.Read(s)) { return false; }
                        self->num_faces += it->second.faces.size();
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
                    std::size_t n = 0;
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
                    std::size_t n = 0;
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
                    std::size_t n = 0;
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
                "max_used_ray_count",
                [](LocalBayesianHilbertMap *self, std::istream &s) {
                    s.read(
                        reinterpret_cast<char *>(&self->max_used_ray_count),
                        sizeof(self->max_used_ray_count));
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
        if (id != other.id) { return false; }
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
        if (max_used_ray_count != other.max_used_ray_count) { return false; }
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
