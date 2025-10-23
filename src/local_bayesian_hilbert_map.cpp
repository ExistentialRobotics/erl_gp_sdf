#include "erl_gp_sdf/local_bayesian_hilbert_map.hpp"

#include "erl_geometry/logodd.hpp"

namespace erl::gp_sdf {
    template<typename Dtype>
    YAML::Node
    LocalBayesianHilbertMapSetting<Dtype>::YamlConvertImpl::encode(
        const LocalBayesianHilbertMapSetting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, bhm);
        ERL_YAML_SAVE_ATTR(node, setting, kernel_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel_setting_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel);
        ERL_YAML_SAVE_ATTR(node, setting, min_dataset_size);
        ERL_YAML_SAVE_ATTR(node, setting, max_dataset_size);
        ERL_YAML_SAVE_ATTR(node, setting, hit_buffer_size);
        ERL_YAML_SAVE_ATTR(node, setting, surface_grid_size);
        ERL_YAML_SAVE_ATTR(node, setting, surface_log_odds);
        ERL_YAML_SAVE_ATTR(node, setting, surface_log_odds_min);
        ERL_YAML_SAVE_ATTR(node, setting, surface_log_odds_max);
        ERL_YAML_SAVE_ATTR(node, setting, faster_prediction);

        return node;
    }

    template<typename Dtype>
    bool
    LocalBayesianHilbertMapSetting<Dtype>::YamlConvertImpl::decode(
        const YAML::Node &node,
        LocalBayesianHilbertMapSetting &setting) {
        if (!node.IsMap()) { return false; }
        if (!ERL_YAML_LOAD_ATTR(node, setting, bhm)) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, kernel_type);
        ERL_YAML_LOAD_ATTR(node, setting, kernel_setting_type);
        setting.kernel = common::YamlableBase::Create<KernelSetting>(setting.kernel_setting_type);
        if (!ERL_YAML_LOAD_ATTR(node, setting, kernel)) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, min_dataset_size);
        ERL_YAML_LOAD_ATTR(node, setting, max_dataset_size);
        ERL_YAML_LOAD_ATTR(node, setting, hit_buffer_size);
        ERL_YAML_LOAD_ATTR(node, setting, surface_grid_size);
        ERL_YAML_LOAD_ATTR(node, setting, surface_log_odds);
        ERL_YAML_LOAD_ATTR(node, setting, surface_log_odds_min);
        ERL_YAML_LOAD_ATTR(node, setting, surface_log_odds_max);
        ERL_YAML_LOAD_ATTR(node, setting, faster_prediction);
        return true;
    }

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
        Positions hinged_points,
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
        if (setting->hit_buffer_size > 0) { hit_buffer.reserve(setting->hit_buffer_size); }
        surface_log_odds = setting->surface_log_odds;
        log_odds_count = 1;
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::Reset() {
        active = false;
        hit_buffer.clear();
        hit_buffer_head = 0;
        // surface_log_odds = setting->surface_log_odds;
        // log_odds_count = 1;
        // bhm.Reset();
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::GenerateDataset(
        const Eigen::Ref<const Position> &sensor_origin,
        const Eigen::Ref<const Positions> &points,
        const std::vector<long> &point_indices) {
        const long max_dataset_size = setting->max_dataset_size;
        if (dataset_size < setting->min_dataset_size) {
            // previous dataset was too small. it is not used yet. combine it with new points.
            long n_points = 0;
            Positions extra_points;
            VectorX extra_labels;
            bhm.GenerateDataset(
                sensor_origin,
                points,
                point_indices,
                max_dataset_size,
                n_points,
                extra_points,
                extra_labels,
                hit_indices);
            if (n_points > 0) {
                dataset_size += n_points;
                dataset_points.conservativeResize(Eigen::NoChange, dataset_size);
                dataset_labels.conservativeResize(dataset_size);
                dataset_points.rightCols(n_points) = extra_points;
                dataset_labels.tail(n_points) = extra_labels;
                return dataset_size >= setting->min_dataset_size;
            }
            return false;
        }
        bhm.GenerateDataset(
            sensor_origin,
            points,
            point_indices,
            max_dataset_size,
            dataset_size,
            dataset_points,
            dataset_labels,
            hit_indices);
        return dataset_size >= setting->min_dataset_size;
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Update(
        const Eigen::Ref<const Position> &sensor_origin,
        const Eigen::Ref<const Positions> &points,
        const std::vector<long> &point_indices,
        const bool update_surface_log_odds) {

        bool updated = GenerateDataset(sensor_origin, points, point_indices);

        // active: whether the local BHM is valid
        // dataset_ok: whether there are enough dataset points

        active = true;  // assume the map is valid first

        if (updated) {
            bhm.RunExpectationMaximization(dataset_points, dataset_labels, dataset_size);
        }

        UpdateHitBuffer(points);

        Dtype log_odd_sum = 0.0f;
        for (const auto &point: hit_buffer) {
            GridIndex voxel_coords;
            voxel_coords[Dim] = 0;  // edge coord, not used here
            if (!GetGridCoords(point, true, voxel_coords)) { continue; }
            auto [itr, inserted] = surf_voxels.try_emplace(voxel_coords, Voxel{});
            if (inserted) { updated = true; }

            if (update_surface_log_odds) {
                updated = true;
                Dtype log_odd = 0.0f;
                Gradient gradient;
                bhm.Predict(
                    point,
                    true /*logodd*/,
                    setting->faster_prediction,
                    false /*compute_gradient*/,
                    false /*gradient_with_sigmoid*/,
                    log_odd,
                    gradient);
                log_odd_sum += log_odd;
            }
        }

        if (update_surface_log_odds && !hit_indices.empty()) {
            surface_log_odds *= static_cast<Dtype>(log_odds_count);
            log_odds_count += hit_indices.size();
            surface_log_odds += log_odd_sum;
            surface_log_odds /= static_cast<Dtype>(log_odds_count);
        }

        // reset if the surface_log_odds is out of bounds
        if (surface_log_odds < setting->surface_log_odds_min ||
            surface_log_odds > setting->surface_log_odds_max) {
            Reset();
        }

        return updated;
    }

    template<typename Dtype, int Dim>
    void
    LocalBayesianHilbertMap<Dtype, Dim>::UpdateHitBuffer(
        const Eigen::Ref<const Positions> &points) {
        if (setting->hit_buffer_size > 0) {
            if (hit_indices.empty()) { return; }  // nothing to add

            // hit buffer has space and there are hit points
            for (const long &hit_index: hit_indices) {
                if (hit_buffer.size() < hit_buffer.capacity()) {
                    hit_buffer.emplace_back(points.col(hit_index));
                    hit_buffer_head = static_cast<long>(hit_buffer.size() % hit_buffer.capacity());
                } else {
                    hit_buffer[hit_buffer_head] = points.col(hit_index);
                    hit_buffer_head = (hit_buffer_head + 1) % hit_buffer.capacity();
                }
            }
        } else {
            hit_buffer.clear();
            for (const long &hit_index: hit_indices) {
                hit_buffer.emplace_back(points.col(hit_index));
            }
        }
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::GetGridCoords(
        const Eigen::Ref<const Position> &point,
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
        const Eigen::Ref<const Positions> &points,
        bool logodd,
        bool compute_free_space,
        bool compute_gradient,
        bool gradient_with_sigmoid,
        bool parallel,
        VectorX &prob_occupied,
        Eigen::VectorXb &in_free_space,
        Gradients &gradient) const {
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
        const Position &point,
        bool logodd,
        bool compute_free_space,
        bool compute_gradient,
        bool gradient_with_sigmoid,
        Dtype &prob_occupied,
        bool &in_free_space,
        Gradient &gradient) const {
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
        const Eigen::Ref<const Positions> &points,
        bool with_sigmoid,
        bool parallel,
        Gradients &gradient) const {
        bhm.PredictGradient(points, setting->faster_prediction, with_sigmoid, parallel, gradient);
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Write(std::ostream &s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<LocalBayesianHilbertMap> token_function_pairs = {
            // setting is loaded externally.
            // tracked_surface_boundary is loaded externally.
            {
                "bhm",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    return self->bhm.Write(stream) && stream.good();
                },
            },
            // `strides` is computed by the constructor.
            {
                "surface_indices",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    const std::size_t n = self->surface_indices.size();
                    stream.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    for (const auto &[index, buf_idx]: self->surface_indices) {
                        stream.write(reinterpret_cast<const char *>(&index), sizeof(index));
                        stream.write(reinterpret_cast<const char *>(&buf_idx), sizeof(buf_idx));
                    }
                    return stream.good();
                },
            },
            {
                "surf_voxels",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    const std::size_t n_voxels = self->surf_voxels.size();
                    stream.write(reinterpret_cast<const char *>(&n_voxels), sizeof(n_voxels));
                    for (const auto &[index, voxel]: self->surf_voxels) {
                        stream.write(reinterpret_cast<const char *>(&index), sizeof(index));
                        stream.write(
                            reinterpret_cast<const char *>(&voxel.good),
                            sizeof(voxel.good));
                        stream.write(
                            reinterpret_cast<const char *>(&voxel.surf_config),
                            sizeof(voxel.surf_config));
                        // write edges
                        const std::size_t n_edges = voxel.edges.size();
                        stream.write(reinterpret_cast<const char *>(&n_edges), sizeof(n_edges));
                        for (const auto &edge: voxel.edges) {
                            for (long i = 0; i < edge.size(); ++i) {
                                stream.write(
                                    reinterpret_cast<const char *>(&edge[i]),
                                    sizeof(edge[i]));
                            }
                        }
                        // write faces
                        const std::size_t n_faces = voxel.faces.size();
                        stream.write(reinterpret_cast<const char *>(&n_faces), sizeof(n_faces));
                        for (const auto &face: voxel.faces) {
                            for (long i = 0; i < face.size(); ++i) {
                                stream.write(
                                    reinterpret_cast<const char *>(&face[i]),
                                    sizeof(face[i]));
                            }
                        }
                    }
                    return stream.good();
                },
            },
            {
                "dataset_size",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    stream.write(
                        reinterpret_cast<const char *>(&self->dataset_size),
                        sizeof(self->dataset_size));
                    return stream.good();
                },
            },
            {
                "dataset_points",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    return SaveEigenMatrixToBinaryStream(stream, self->dataset_points) &&
                           stream.good();
                },
            },
            {
                "dataset_labels",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    return SaveEigenMatrixToBinaryStream(stream, self->dataset_labels) &&
                           stream.good();
                },
            },
            {
                "hit_indices",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    const std::size_t n = self->hit_indices.size();
                    stream.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    stream.write(
                        reinterpret_cast<const char *>(self->hit_indices.data()),
                        sizeof(long) * n);
                    return stream.good();
                },
            },
            {
                "hit_buffer",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    const std::size_t n = self->hit_buffer.size();
                    stream.write(reinterpret_cast<const char *>(&n), sizeof(n));
                    if (n == 0) { return stream.good(); }
                    stream.write(
                        reinterpret_cast<const char *>(self->hit_buffer.data()),
                        sizeof(Position) * n);
                    return stream.good();
                },
            },
            {
                "hit_buffer_head",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    stream.write(
                        reinterpret_cast<const char *>(&self->hit_buffer_head),
                        sizeof(self->hit_buffer_head));
                    return stream.good();
                },
            },
            {
                "active",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    stream.write(reinterpret_cast<const char *>(&self->active), sizeof(bool));
                    return stream.good();
                },
            },
            // surf_data_cache is temporary.
            {
                "surface_log_odds",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    stream.write(
                        reinterpret_cast<const char *>(&self->surface_log_odds),
                        sizeof(self->surface_log_odds));
                    return stream.good();
                },
            },
            {
                "log_odds_count",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    stream.write(
                        reinterpret_cast<const char *>(&self->log_odds_count),
                        sizeof(self->log_odds_count));
                    return stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    LocalBayesianHilbertMap<Dtype, Dim>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<LocalBayesianHilbertMap> token_function_pairs = {
            // setting is loaded externally.
            // tracked_surface_boundary is loaded externally.
            {
                "bhm",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    return self->bhm.Read(stream) && stream.good();
                },
            },
            // `strides` is computed by the constructor.
            {
                "surface_indices",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    std::size_t n;
                    stream.read(reinterpret_cast<char *>(&n), sizeof(n));
                    self->surface_indices.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        GridIndex index;
                        stream.read(reinterpret_cast<char *>(&index), sizeof(index));
                        std::size_t buf_idx;
                        stream.read(reinterpret_cast<char *>(&buf_idx), sizeof(buf_idx));
                        if (!self->surface_indices.try_emplace(index, buf_idx).second) {
                            ERL_WARN("Duplicate surface_indices index: {}.", index);
                            return false;
                        }
                    }
                    return stream.good();
                },
            },
            {
                "surf_voxels",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    std::size_t n_voxels;
                    stream.read(reinterpret_cast<char *>(&n_voxels), sizeof(n_voxels));
                    if (n_voxels == 0) {
                        self->surf_voxels.clear();
                        return stream.good();
                    }
                    self->surf_voxels.reserve(n_voxels);
                    for (std::size_t i = 0; i < n_voxels; ++i) {
                        GridIndex index;
                        stream.read(reinterpret_cast<char *>(&index), sizeof(index));
                        if (self->surf_voxels.contains(index)) {
                            ERL_WARN("Duplicate surf_voxels index: {}.", index);
                            return false;
                        }
                        Voxel voxel;
                        stream.read(reinterpret_cast<char *>(&voxel.good), sizeof(voxel.good));
                        stream.read(
                            reinterpret_cast<char *>(&voxel.surf_config),
                            sizeof(voxel.surf_config));
                        // read edges
                        std::size_t n_edges = 0;
                        stream.read(reinterpret_cast<char *>(&n_edges), sizeof(n_edges));
                        if (n_edges == 0) {
                            voxel.edges.clear();
                        } else {
                            voxel.edges.reserve(n_edges);
                            for (std::size_t j = 0; j < n_edges; ++j) {
                                GridIndex edge;
                                for (long k = 0; k < edge.size(); ++k) {
                                    stream.read(
                                        reinterpret_cast<char *>(&edge[k]),
                                        sizeof(edge[k]));
                                }
                                voxel.edges.push_back(edge);
                            }
                        }
                        // read faces
                        std::size_t n_faces = 0;
                        stream.read(reinterpret_cast<char *>(&n_faces), sizeof(n_faces));
                        if (n_faces == 0) {
                            voxel.faces.clear();
                        } else {
                            voxel.faces.reserve(n_faces);
                            for (std::size_t j = 0; j < n_faces; ++j) {
                                Face face;
                                for (long k = 0; k < face.size(); ++k) {
                                    stream.read(
                                        reinterpret_cast<char *>(&face[k]),
                                        sizeof(face[k]));
                                }
                                voxel.faces.push_back(face);
                            }
                        }
                        // insert it
                        self->surf_voxels[index] = std::move(voxel);
                    }
                    return stream.good();
                },
            },
            {
                "dataset_size",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&self->dataset_size),
                        sizeof(self->dataset_size));
                    return stream.good();
                },
            },
            {
                "dataset_points",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    return LoadEigenMatrixFromBinaryStream(stream, self->dataset_points) &&
                           stream.good();
                },
            },
            {
                "dataset_labels",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    return LoadEigenMatrixFromBinaryStream(stream, self->dataset_labels) &&
                           stream.good();
                },
            },
            {
                "hit_indices",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    std::size_t n;
                    stream.read(reinterpret_cast<char *>(&n), sizeof(n));
                    if (n == 0) {
                        self->hit_indices.clear();
                        return stream.good();
                    }
                    self->hit_indices.resize(n);
                    stream.read(
                        reinterpret_cast<char *>(self->hit_indices.data()),
                        static_cast<std::streamsize>(sizeof(long) * n));
                    return stream.good();
                },
            },
            {
                "hit_buffer",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    std::size_t n;
                    stream.read(reinterpret_cast<char *>(&n), sizeof(n));
                    if (n == 0) {
                        self->hit_buffer.clear();
                        return stream.good();
                    }
                    self->hit_buffer.resize(n);
                    stream.read(
                        reinterpret_cast<char *>(self->hit_buffer.data()),
                        sizeof(Position) * n);
                    return stream.good();
                },
            },
            {
                "hit_buffer_head",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&self->hit_buffer_head),
                        sizeof(self->hit_buffer_head));
                    return stream.good();
                },
            },
            {
                "active",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    stream.read(reinterpret_cast<char *>(&self->active), sizeof(bool));
                    return stream.good();
                },
            },
            {
                "surface_log_odds",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&self->surface_log_odds),
                        sizeof(self->surface_log_odds));
                    return stream.good();
                },
            },
            {
                "log_odds_count",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&self->log_odds_count),
                        sizeof(self->log_odds_count));
                    return stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
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
        if (hit_buffer != other.hit_buffer) { return false; }
        if (hit_buffer_head != other.hit_buffer_head) { return false; }
        if (active != other.active) { return false; }
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
