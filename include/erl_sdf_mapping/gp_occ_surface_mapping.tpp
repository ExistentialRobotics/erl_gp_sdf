#pragma once
#include "erl_common/clip.hpp"
#include "erl_common/template_helper.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    YAML::Node
    GpOccSurfaceMapping<Dtype, Dim>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node cv_node;
        auto &compute_variance = setting.compute_variance;
        ERL_YAML_SAVE_ATTR(cv_node, compute_variance, zero_gradient_position_var);
        ERL_YAML_SAVE_ATTR(cv_node, compute_variance, zero_gradient_gradient_var);
        ERL_YAML_SAVE_ATTR(cv_node, compute_variance, position_var_alpha);
        ERL_YAML_SAVE_ATTR(cv_node, compute_variance, min_distance_var);
        ERL_YAML_SAVE_ATTR(cv_node, compute_variance, max_distance_var);
        ERL_YAML_SAVE_ATTR(cv_node, compute_variance, min_gradient_var);
        ERL_YAML_SAVE_ATTR(cv_node, compute_variance, max_gradient_var);

        YAML::Node ump_node;
        auto &update_map_points = setting.update_map_points;
        ERL_YAML_SAVE_ATTR(ump_node, update_map_points, max_adjust_tries);
        ERL_YAML_SAVE_ATTR(ump_node, update_map_points, min_observable_occ);
        ERL_YAML_SAVE_ATTR(ump_node, update_map_points, min_position_var);
        ERL_YAML_SAVE_ATTR(ump_node, update_map_points, min_gradient_var);
        ERL_YAML_SAVE_ATTR(ump_node, update_map_points, max_surface_abs_occ);
        ERL_YAML_SAVE_ATTR(ump_node, update_map_points, max_valid_gradient_var);
        ERL_YAML_SAVE_ATTR(ump_node, update_map_points, max_bayes_position_var);
        ERL_YAML_SAVE_ATTR(ump_node, update_map_points, max_bayes_gradient_var);

        YAML::Node node;
        node["compute_variance"] = cv_node;
        node["update_map_points"] = ump_node;
        ERL_YAML_SAVE_ATTR(node, setting, sensor_gp);
        ERL_YAML_SAVE_ATTR(node, setting, tree);
        ERL_YAML_SAVE_ATTR(node, setting, surface_resolution);
        ERL_YAML_SAVE_ATTR(node, setting, scaling);
        ERL_YAML_SAVE_ATTR(node, setting, perturb_delta);
        ERL_YAML_SAVE_ATTR(node, setting, zero_gradient_threshold);
        ERL_YAML_SAVE_ATTR(node, setting, update_occupancy);
        ERL_YAML_SAVE_ATTR(node, setting, cluster_depth);
        return node;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }

        const YAML::Node cv_node = node["compute_variance"];
        auto &compute_variance = setting.compute_variance;
        ERL_YAML_LOAD_ATTR(cv_node, compute_variance, zero_gradient_position_var);
        ERL_YAML_LOAD_ATTR(cv_node, compute_variance, zero_gradient_gradient_var);
        ERL_YAML_LOAD_ATTR(cv_node, compute_variance, position_var_alpha);
        ERL_YAML_LOAD_ATTR(cv_node, compute_variance, min_distance_var);
        ERL_YAML_LOAD_ATTR(cv_node, compute_variance, max_distance_var);
        ERL_YAML_LOAD_ATTR(cv_node, compute_variance, min_gradient_var);
        ERL_YAML_LOAD_ATTR(cv_node, compute_variance, max_gradient_var);

        const YAML::Node ump_node = node["update_map_points"];
        auto &update_map_points = setting.update_map_points;
        ERL_YAML_LOAD_ATTR(ump_node, update_map_points, max_adjust_tries);
        ERL_YAML_LOAD_ATTR(ump_node, update_map_points, min_observable_occ);
        ERL_YAML_LOAD_ATTR(ump_node, update_map_points, min_position_var);
        ERL_YAML_LOAD_ATTR(ump_node, update_map_points, min_gradient_var);
        ERL_YAML_LOAD_ATTR(ump_node, update_map_points, max_surface_abs_occ);
        ERL_YAML_LOAD_ATTR(ump_node, update_map_points, max_valid_gradient_var);
        ERL_YAML_LOAD_ATTR(ump_node, update_map_points, max_bayes_position_var);
        ERL_YAML_LOAD_ATTR(ump_node, update_map_points, max_bayes_gradient_var);

        ERL_YAML_LOAD_ATTR(node, setting, sensor_gp);
        ERL_YAML_LOAD_ATTR(node, setting, tree);
        ERL_YAML_LOAD_ATTR(node, setting, surface_resolution);
        ERL_YAML_LOAD_ATTR(node, setting, scaling);
        ERL_YAML_LOAD_ATTR(node, setting, perturb_delta);
        ERL_YAML_LOAD_ATTR(node, setting, zero_gradient_threshold);
        ERL_YAML_LOAD_ATTR(node, setting, update_occupancy);
        ERL_YAML_LOAD_ATTR(node, setting, cluster_depth);
        return true;
    }

    template<typename Dtype, int Dim>
    GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator::SurfaceDataIterator(
        GpOccSurfaceMapping *mapping)
        : m_mapping_(mapping) {
        if (m_mapping_ == nullptr) { return; }
        m_use0_ = m_mapping_->m_setting_->surface_resolution <= 0;
        if (m_use0_) {
            m_it0_ = m_mapping_->m_surf_indices0_.begin();
            // no surface data available
            if (m_it0_ == m_mapping_->m_surf_indices0_.end()) { m_mapping_ = nullptr; }
            return;
        }
        m_it1_ = m_mapping_->m_surf_indices1_.begin();
        if (m_it1_ == m_mapping_->m_surf_indices1_.end()) {  // no surface data available
            m_mapping_ = nullptr;
            return;
        }
        while (m_it1_ != m_mapping_->m_surf_indices1_.end()) {
            m_it2_ = m_it1_->second.begin();
            if (m_it2_ != m_it1_->second.end()) { break; }  // found a valid surface data
            ++m_it1_;                                       // move to the next entry
        }
        if (m_it1_ == m_mapping_->m_surf_indices1_.end()) {  // no valid surface data available
            m_mapping_ = nullptr;
            return;
        }
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator::operator==(
        const SurfaceDataIterator &other) const {
        if (m_mapping_ != other.m_mapping_) { return false; }
        // both iterators are at the end (m_mapping_ == nullptr)
        if (m_mapping_ == nullptr) { return true; }
        if (m_use0_ != other.m_use0_) { return false; }
        // compare the iterators for surface data with resolution <= 0
        if (m_use0_) { return m_it0_ == other.m_it0_; }
        // compare the iterators for surface data with resolution > 0
        return m_it1_ == other.m_it1_ && m_it2_ == other.m_it2_;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator::operator!=(
        const SurfaceDataIterator &other) const {
        return !(*this == other);
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::SurfData &
    GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator::operator*() {
        ERL_DEBUG_ASSERT(
            m_mapping_ != nullptr,
            "invalid SurfaceDataIterator, m_mapping_ is nullptr.");
        if (m_use0_) { return m_mapping_->m_surf_data_manager_[m_it0_->second]; }
        return m_mapping_->m_surf_data_manager_[m_it2_->second];
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::SurfData *
    GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator::operator->() {
        return &operator*();
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator &
    GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator::operator++() {
        if (m_mapping_ == nullptr) { return *this; }
        if (m_use0_) {
            if (m_it0_ != m_mapping_->m_surf_indices0_.end()) { ++m_it0_; }
            // no more surface data available
            if (m_it0_ == m_mapping_->m_surf_indices0_.end()) { m_mapping_ = nullptr; }
            return *this;
        }
        if (m_it2_ != m_it1_->second.end()) { ++m_it2_; }
        if (m_it2_ == m_it1_->second.end()) {
            ++m_it1_;  // move to the next entry
            while (m_it1_ != m_mapping_->m_surf_indices1_.end()) {
                m_it2_ = m_it1_->second.begin();
                if (m_it2_ != m_it1_->second.end()) { break; }  // found a valid surface data
                ++m_it1_;                                       // move to the next entry
            }
            // no valid surface data available
            if (m_it1_ == m_mapping_->m_surf_indices1_.end()) { m_mapping_ = nullptr; }
        }
        return *this;
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator
    GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator::operator++(int) {
        SurfaceDataIterator tmp(*this);
        ++*this;
        return tmp;
    }

    template<typename Dtype, int Dim>
    GpOccSurfaceMapping<Dtype, Dim>::GpOccSurfaceMapping(std::shared_ptr<Setting> setting)
        : m_setting_(NotNull(std::move(setting), true, "setting is nullptr")),
          m_sensor_gp_(std::make_shared<SensorGp>(m_setting_->sensor_gp)),
          m_tree_(std::make_shared<Tree>(m_setting_->tree)) {

        if (m_setting_->surface_resolution > 0) {
            m_surface_resolution_inv_ = 1.0f / m_setting_->surface_resolution;
            Eigen::Vector<int, Dim> grid_shape;
            grid_shape.fill(static_cast<int>(
                std::ceil(m_setting_->tree->resolution * m_surface_resolution_inv_)));
            m_strides_ = common::ComputeCStrides<int>(grid_shape, 1);
        }

        const Dtype d = m_setting_->perturb_delta;
        m_pos_perturb_.setZero();
        for (int i = 0; i < Dim; i++) {
            m_pos_perturb_(i, 2 * i) = d;
            m_pos_perturb_(i, 2 * i + 1) = -d;
        }
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename GpOccSurfaceMapping<Dtype, Dim>::Setting>
    GpOccSurfaceMapping<Dtype, Dim>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename GpOccSurfaceMapping<Dtype, Dim>::SensorGp>
    GpOccSurfaceMapping<Dtype, Dim>::GetSensorGp() const {
        return m_sensor_gp_;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename GpOccSurfaceMapping<Dtype, Dim>::Tree>
    GpOccSurfaceMapping<Dtype, Dim>::GetTree() const {
        return m_tree_;
    }

    template<typename Dtype, int Dim>
    const typename GpOccSurfaceMapping<Dtype, Dim>::SurfDataManager &
    GpOccSurfaceMapping<Dtype, Dim>::GetSurfaceDataManager() const {
        return m_surf_data_manager_;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Update(
        const Eigen::Ref<const Rotation> &rotation,
        const Eigen::Ref<const Translation> &translation,
        const Eigen::Ref<const Ranges> &ranges) {

        m_changed_keys_.clear();
        if (const Dtype s = m_setting_->scaling; s != 1.0f) {
            if (!m_sensor_gp_->Train(rotation, translation.array() * s, ranges.array() * s)) {
                return false;
            }
        } else {
            if (!m_sensor_gp_->Train(rotation, translation, ranges)) { return false; }
        }

        {
            auto lock_guard = GetLockGuard();  // CRITICAL SECTION
            if (m_setting_->update_occupancy) { UpdateOccupancy(); }
            if (m_setting_->surface_resolution <= 0) {
                UpdateMapPoints0();
                AddNewMeasurement0();
            } else {
                UpdateMapPoints1();
                AddNewMeasurement1();
            }
        }

        return true;
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator
    GpOccSurfaceMapping<Dtype, Dim>::BeginSurfaceData() {
        return SurfaceDataIterator(this);
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::SurfaceDataIterator
    GpOccSurfaceMapping<Dtype, Dim>::EndSurfaceData() {
        return SurfaceDataIterator(nullptr);
    }

    template<typename Dtype, int Dim>
    std::lock_guard<std::mutex>
    GpOccSurfaceMapping<Dtype, Dim>::GetLockGuard() {
        return std::lock_guard(m_mutex_);
    }

    template<typename Dtype, int Dim>
    Dtype
    GpOccSurfaceMapping<Dtype, Dim>::GetScaling() const {
        return m_setting_->scaling;
    }

    template<typename Dtype, int Dim>
    Dtype
    GpOccSurfaceMapping<Dtype, Dim>::GetClusterSize() const {
        return m_tree_->GetNodeSize(m_setting_->cluster_depth);
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::Position
    GpOccSurfaceMapping<Dtype, Dim>::GetClusterCenter(const Key &key) const {
        return m_tree_->KeyToCoord(key, m_setting_->cluster_depth);
    }

    template<typename Dtype, int Dim>
    const typename GpOccSurfaceMapping<Dtype, Dim>::KeySet &
    GpOccSurfaceMapping<Dtype, Dim>::GetChangedClusters() const {
        return m_changed_keys_;
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::IterateClustersInAabb(
        const Aabb &aabb,
        std::function<void(const Key &)> callback) const {
        const uint32_t cluster_depth = m_setting_->cluster_depth;
        for (auto it = m_tree_->BeginTreeInAabb(aabb, cluster_depth),
                  end = m_tree_->EndTreeInAabb();
             it != end;
             ++it) {
            if (it->GetDepth() != cluster_depth) { continue; }
            callback(m_tree_->AdjustKeyToDepth(it.GetKey(), cluster_depth));
        }
    }

    template<typename Dtype, int Dim>
    const std::vector<typename GpOccSurfaceMapping<Dtype, Dim>::SurfData> &
    GpOccSurfaceMapping<Dtype, Dim>::GetSurfaceDataBuffer() const {
        return m_surf_data_manager_.GetEntries();
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::CollectSurfaceDataInAabb(
        const Aabb &aabb,
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const {
        surface_data_indices.clear();
        if (m_setting_->surface_resolution <= 0) {  // zero resolution, use m_surf_indices0_
            for (auto it = m_tree_->BeginLeafInAabb(aabb), end = m_tree_->EndLeafInAabb();
                 it != end;
                 ++it) {
                Key key = it.GetKey();
                auto surf_it = m_surf_indices0_.find(key);
                // no surface data for this key
                if (surf_it == m_surf_indices0_.end()) { continue; }
                const auto &surface_data = m_surf_data_manager_[surf_it->second];
                surface_data_indices.emplace_back(
                    (aabb.center - surface_data.position).norm(),
                    surf_it->second);
            }
            return;
        }
        for (auto it = m_tree_->BeginLeafInAabb(aabb), end = m_tree_->EndLeafInAabb(); it != end;
             ++it) {
            Key key = it.GetKey();
            auto surf_it = m_surf_indices1_.find(key);
            if (surf_it == m_surf_indices1_.end()) { continue; }  // no surface data for this key
            const auto &surf_indices = surf_it->second;
            if (surf_indices.empty()) { continue; }  // no surface data for this key
            for (const auto &[grid_index, surf_index]: surf_indices) {
                const auto &surface_data = m_surf_data_manager_[surf_index];
                surface_data_indices.emplace_back(
                    (aabb.center - surface_data.position).norm(),
                    surf_index);
            }
        }
        if (std::abs(aabb.center[0] - 6.3f) < 0.001f && std::abs(aabb.center[1] + 2.7f) < 0.001f) {
            ERL_INFO(
                "{} surface data indices collected in Aabb({}, {})",
                surface_data_indices.size(),
                aabb.center.transpose(),
                aabb.half_sizes[0]);
            std::cout << std::flush;
        }
    }

    template<typename Dtype, int Dim>
    typename GpOccSurfaceMapping<Dtype, Dim>::Aabb
    GpOccSurfaceMapping<Dtype, Dim>::GetMapBoundary() const {
        Position min, max;
        m_tree_->GetMetricMinMax(min, max);
        return Aabb(min, max);
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::IsInFreeSpace(
        const Positions &positions,
        VectorX &in_free_space) const {
        if (!m_setting_->update_occupancy) {
            ERL_WARN("update_occupancy is false, cannot check if positions are in free space.");
            return false;
        }
        const long num_positions = positions.cols();
        if (in_free_space.size() < num_positions) { in_free_space.resize(num_positions); }
        const Dtype s = m_setting_->scaling;

#pragma omp parallel for default(none) shared(positions, in_free_space, num_positions, s)
        for (long i = 0; i < num_positions; ++i) {
            const auto node = m_tree_->Search(positions.col(i) * s);
            if (node == nullptr || m_tree_->IsNodeOccupied(node)) {
                in_free_space[i] = -1.0f;
            } else {
                in_free_space[i] = 1.0f;
            }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::operator==(const AbstractSurfaceMapping &other) const {
        const auto *other_ptr = dynamic_cast<const GpOccSurfaceMapping *>(&other);
        if (other_ptr == nullptr) { return false; }
        if (m_setting_ == nullptr && other_ptr->m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other_ptr->m_setting_ == nullptr || *m_setting_ != *other_ptr->m_setting_)) {
            return false;
        }
        if (m_sensor_gp_ == nullptr && other_ptr->m_sensor_gp_ != nullptr) { return false; }
        if (m_sensor_gp_ != nullptr &&
            (other_ptr->m_sensor_gp_ == nullptr || *m_sensor_gp_ != *other_ptr->m_sensor_gp_)) {
            return false;
        }
        if (m_tree_ == nullptr && other_ptr->m_tree_ != nullptr) { return false; }
        if (m_tree_ != nullptr &&
            (other_ptr->m_tree_ == nullptr || *m_tree_ != *other_ptr->m_tree_)) {
            return false;
        }
        if (m_strides_ != other_ptr->m_strides_) { return false; }
        if (m_surf_indices0_ != other_ptr->m_surf_indices0_) { return false; }
        if (m_surf_indices1_ != other_ptr->m_surf_indices1_) { return false; }
        if (m_surf_data_manager_ != other_ptr->m_surf_data_manager_) { return false; }
        if (m_pos_perturb_ != other_ptr->m_pos_perturb_) { return false; }
        if (m_surface_resolution_inv_ != other_ptr->m_surface_resolution_inv_) { return false; }
        if (m_changed_keys_ != other_ptr->m_changed_keys_) { return false; }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Write(std::ostream &s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<GpOccSurfaceMapping> token_function_pairs = {
            {
                "setting",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    return gp->m_setting_->Write(stream) && stream.good();
                },
            },
            {
                "sensor_gp",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    return gp->m_sensor_gp_->Write(stream) && stream.good();
                },
            },
            {
                "tree",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    return gp->m_tree_->Write(stream) && stream.good();
                },
            },
            {
                "strides",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_strides_) && stream.good();
                },
            },
            {
                "surf_indices0",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    const std::size_t num_entries = gp->m_surf_indices0_.size();
                    stream.write(reinterpret_cast<const char *>(&num_entries), sizeof(num_entries));
                    for (const auto &[key, surf_index]: gp->m_surf_indices0_) {
                        for (int i = 0; i < Dim; ++i) {
                            stream.write(
                                reinterpret_cast<const char *>(&key[i]),
                                sizeof(typename Key::KeyType));
                        }
                        stream.write(
                            reinterpret_cast<const char *>(&surf_index),
                            sizeof(surf_index));
                    }
                    return stream.good();
                },
            },
            {
                "surf_indices1",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    const std::size_t num_entries = gp->m_surf_indices1_.size();
                    stream.write(reinterpret_cast<const char *>(&num_entries), sizeof(num_entries));
                    for (const auto &[key, surf_indices]: gp->m_surf_indices1_) {
                        for (int i = 0; i < Dim; ++i) {
                            stream.write(
                                reinterpret_cast<const char *>(&key[i]),
                                sizeof(typename Key::KeyType));
                        }
                        const std::size_t num_surf_indices = surf_indices.size();
                        stream.write(
                            reinterpret_cast<const char *>(&num_surf_indices),
                            sizeof(num_surf_indices));
                        for (const auto &[grid_index, surf_index]: surf_indices) {
                            stream.write(
                                reinterpret_cast<const char *>(&grid_index),
                                sizeof(grid_index));
                            stream.write(
                                reinterpret_cast<const char *>(&surf_index),
                                sizeof(surf_index));
                        }
                    }
                    return stream.good();
                },
            },
            {
                "surf_data_manager",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    return gp->m_surf_data_manager_.Write(stream) && stream.good();
                },
            },
            {
                "pos_perturb",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    return SaveEigenMatrixToBinaryStream(stream, gp->m_pos_perturb_) &&
                           stream.good();
                },
            },
            {
                "surface_resolution_inv",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    return stream.write(
                               reinterpret_cast<const char *>(&gp->m_surface_resolution_inv_),
                               sizeof(gp->m_surface_resolution_inv_)) &&
                           stream.good();
                },
            },
            {
                "changed_keys",
                [](const GpOccSurfaceMapping *gp, std::ostream &stream) {
                    const std::size_t num_entries = gp->m_changed_keys_.size();
                    stream.write(reinterpret_cast<const char *>(&num_entries), sizeof(num_entries));
                    for (const Key &key: gp->m_changed_keys_) {
                        for (int i = 0; i < Dim; ++i) {
                            stream.write(
                                reinterpret_cast<const char *>(&key[i]),
                                sizeof(typename Key::KeyType));
                        }
                    }
                    return stream.good();
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<GpOccSurfaceMapping> token_function_pairs = {
            {
                "setting",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    return gp->m_setting_->Read(stream) && stream.good();
                },
            },
            {
                "sensor_gp",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    gp->m_sensor_gp_ = std::make_shared<SensorGp>(gp->m_setting_->sensor_gp);
                    return gp->m_sensor_gp_->Read(stream) && stream.good();
                },
            },
            {
                "tree",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    gp->m_tree_ = std::make_shared<Tree>(gp->m_setting_->tree);
                    return gp->m_tree_->Read(stream) && stream.good();
                },
            },
            {
                "strides",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    return LoadEigenMatrixFromBinaryStream(stream, gp->m_strides_) && stream.good();
                },
            },
            {
                "surf_indices0",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    std::size_t num_entries;
                    stream.read(reinterpret_cast<char *>(&num_entries), sizeof(num_entries));
                    gp->m_surf_indices0_.clear();
                    for (std::size_t i = 0; i < num_entries; ++i) {
                        Key key;
                        for (int j = 0; j < Dim; ++j) {
                            stream.read(
                                reinterpret_cast<char *>(&key[j]),
                                sizeof(typename Key::KeyType));
                        }
                        std::size_t surf_index;
                        stream.read(reinterpret_cast<char *>(&surf_index), sizeof(surf_index));
                        gp->m_surf_indices0_[key] = surf_index;
                    }
                    return stream.good();
                },
            },
            {
                "surf_indices1",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    std::size_t num_entries;
                    stream.read(reinterpret_cast<char *>(&num_entries), sizeof(num_entries));
                    gp->m_surf_indices1_.clear();
                    for (std::size_t i = 0; i < num_entries; ++i) {
                        Key key;
                        for (int j = 0; j < Dim; ++j) {
                            stream.read(
                                reinterpret_cast<char *>(&key[j]),
                                sizeof(typename Key::KeyType));
                        }
                        std::size_t num_surf_indices;
                        stream.read(
                            reinterpret_cast<char *>(&num_surf_indices),
                            sizeof(num_surf_indices));
                        auto &surf_indices = gp->m_surf_indices1_[key];
                        for (std::size_t j = 0; j < num_surf_indices; ++j) {
                            int grid_index;
                            std::size_t surf_index;
                            stream.read(reinterpret_cast<char *>(&grid_index), sizeof(grid_index));
                            stream.read(reinterpret_cast<char *>(&surf_index), sizeof(surf_index));
                            surf_indices[grid_index] = surf_index;
                        }
                    }
                    return stream.good();
                },
            },
            {
                "surf_data_manager",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    return gp->m_surf_data_manager_.Read(stream) && stream.good();
                },
            },
            {
                "pos_perturb",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    return LoadEigenMatrixFromBinaryStream(stream, gp->m_pos_perturb_) &&
                           stream.good();
                },
            },
            {
                "surface_resolution_inv",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&gp->m_surface_resolution_inv_),
                        sizeof(gp->m_surface_resolution_inv_));
                    return stream.good();
                },
            },
            {
                "changed_keys",
                [](GpOccSurfaceMapping *gp, std::istream &stream) {
                    std::size_t num_entries;
                    stream.read(reinterpret_cast<char *>(&num_entries), sizeof(num_entries));
                    gp->m_changed_keys_.clear();
                    for (std::size_t i = 0; i < num_entries; ++i) {
                        Key key;
                        for (int j = 0; j < Dim; ++j) {
                            stream.read(
                                reinterpret_cast<char *>(&key[j]),
                                sizeof(typename Key::KeyType));
                        }
                        gp->m_changed_keys_.insert(key);
                    }
                    return stream.good();
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    std::pair<Dtype, Dtype>
    GpOccSurfaceMapping<Dtype, Dim>::Cartesian2Polar(const Dtype x, const Dtype y) {
        Dtype r = std::sqrt(x * x + y * y);
        Dtype angle = std::atan2(y, x);
        return {r, angle};
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::UpdateMapPoints0() {
        ERL_DEBUG_ASSERT(
            m_setting_->surface_resolution <= 0,
            "UpdateMapPoints0() should only be called when the surface resolution is <= 0");

        if (!m_sensor_gp_->IsTrained()) { return; }

        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        const Position &sensor_pos = sensor_frame->GetTranslationVector();
        const Dtype max_sensor_range = sensor_frame->GetMaxValidRange();
        const Aabb observed_area(sensor_pos, max_sensor_range);
        const bool update_occupancy = m_setting_->update_occupancy;  // occupancy is available.
        const uint32_t cluster_depth = m_setting_->cluster_depth;
        const Dtype cluster_half_size = m_tree_->GetNodeSize(cluster_depth) * 0.5f;
        const Dtype squared_dist_max =
            max_sensor_range * max_sensor_range +
            cluster_half_size * cluster_half_size * static_cast<Dtype>(Dim);

        // key, surf_index, updated, to_remove, new_key
        std::vector<std::tuple<Key, std::size_t, bool, bool, std::optional<Key>>> nodes_in_aabb;
        for (auto it = m_tree_->BeginLeafInAabb(observed_area), end = m_tree_->EndLeafInAabb();
             it != end;
             ++it) {
            Key key = it.GetKey();
            // find the surface index of the node
            auto surf_it = m_surf_indices0_.find(key);
            // skip nodes without surface data
            if (surf_it == m_surf_indices0_.end() || surf_it->second < 0) { continue; }
            if (update_occupancy && !m_tree_->IsNodeOccupied(*it)) {
                // skip unoccupied nodes and clear the surface data in them.
                m_surf_data_manager_.RemoveEntry(surf_it->second);
                m_surf_indices0_.erase(key);
                continue;
            }
            if (const Position cluster_position = m_tree_->KeyToCoord(key, cluster_depth);
                (cluster_position - sensor_pos).squaredNorm() > squared_dist_max) {
                continue;  // skip nodes that are too far away
            }
            nodes_in_aabb.emplace_back(key, surf_it->second, false, false, std::nullopt);
        }

        // update the surface data in the nodes
#pragma omp parallel for default(none) \
    shared(nodes_in_aabb, max_sensor_range, sensor_pos, sensor_frame)
        for (auto &[key, surf_index, updated, to_remove, new_key]: nodes_in_aabb) {
            auto &surface_data = m_surf_data_manager_[surf_index];
            UpdateMapPoint(surface_data, updated, to_remove);
            if (!updated) { continue; }
            // if the surface data is updated, we need to check if the node key is changed
            if (Key tmp = m_tree_->CoordToKey(surface_data.position); tmp != key) { new_key = tmp; }
        }

        for (auto &[key, surf_index, updated, to_remove, new_key_opt]: nodes_in_aabb) {
            if (to_remove) {  // too bad, the surface data should be removed
                // this surface data is not used anymore
                m_surf_data_manager_.RemoveEntry(surf_index);
                m_surf_indices0_.erase(key);  // the node has no surface data now
                RecordChangedKey(key);
                continue;
            }
            if (updated) { RecordChangedKey(key); }
            if (!new_key_opt.has_value()) { continue; }  // the surface data is not moved

            // the node is moved to a new position
            const auto &new_key = new_key_opt.value();
            auto new_surf_it = m_surf_indices0_.find(new_key);
            if (new_surf_it == m_surf_indices0_.end()) {  // the new key is not in the index
                m_surf_indices0_[new_key] = surf_index;   // move to the new index
                RecordChangedKey(new_key);
            } else {  // the new key is already in the index
                // this surface data is not used anymore
                m_surf_data_manager_.RemoveEntry(surf_index);
            }
            m_surf_indices0_.erase(key);  // remove the old index
            RecordChangedKey(key);
        }
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::UpdateMapPoints1() {
        ERL_DEBUG_ASSERT(
            m_setting_->surface_resolution > 0,
            "UpdateMapPoints1() should only be called when the surface resolution is > 0");

        if (!m_sensor_gp_->IsTrained()) { return; }

        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        const Position &sensor_pos = sensor_frame->GetTranslationVector();
        const Dtype max_sensor_range = sensor_frame->GetMaxValidRange();
        const Aabb observed_area(sensor_pos, max_sensor_range);
        const bool update_occupancy = m_setting_->update_occupancy;  // occupancy is available.
        const uint32_t cluster_depth = m_setting_->cluster_depth;
        const Dtype cluster_half_size = m_tree_->GetNodeSize(cluster_depth) * 0.5f;
        const Dtype squared_dist_max =
            max_sensor_range * max_sensor_range +
            cluster_half_size * cluster_half_size * static_cast<Dtype>(Dim);

        struct Index {
            Key key;
            int grid_index;
            std::size_t surf_index;
        };

        // index, updated, to_remove, new_index
        std::vector<std::tuple<Index, bool, bool, std::optional<Index>>> surf_in_aabb;
        for (auto it = m_tree_->BeginLeafInAabb(observed_area), end = m_tree_->EndLeafInAabb();
             it != end;
             ++it) {
            Key key = it.GetKey();
            auto surf_it = m_surf_indices1_.find(key);  // find the surface indices of the node
            // skip nodes without surface data
            if (surf_it == m_surf_indices1_.end() || surf_it->second.empty()) { continue; }
            if (update_occupancy && !m_tree_->IsNodeOccupied(*it)) {
                // skip unoccupied nodes and clear the surface data in them.
                for (const auto &[grid_index, surf_index]: surf_it->second) {
                    m_surf_data_manager_.RemoveEntry(surf_index);
                }
                surf_it->second.clear();  // remove the surface data from the buffer and the indices
                continue;
            }
            if (const Position cluster_position = m_tree_->KeyToCoord(key, cluster_depth);
                (cluster_position - sensor_pos).squaredNorm() > squared_dist_max) {
                continue;  // skip nodes that are too far away
            }
            for (const auto &[grid_index, surf_index]: surf_it->second) {
                surf_in_aabb
                    .emplace_back(Index{key, grid_index, surf_index}, false, false, std::nullopt);
            }
        }

        // update the surface data in the nodes
#pragma omp parallel for default(none) shared(surf_in_aabb)
        for (auto &[index, updated, to_remove, new_index]: surf_in_aabb) {
            auto &surface_data = m_surf_data_manager_[index.surf_index];
            UpdateMapPoint(surface_data, updated, to_remove);
            if (updated) {
                const auto [new_key, grid_index] = ComputeSurfaceIndex1(surface_data.position);
                if (index.key != new_key || index.grid_index != grid_index) {
                    new_index = {new_key, grid_index, index.surf_index};
                }
            }
        }

        for (const auto &[index, updated, to_remove, new_index]: surf_in_aabb) {
            if (to_remove) {  // too bad, the surface data should be removed.
                // remove the surface data from the buffer
                m_surf_data_manager_.RemoveEntry(index.surf_index);
                m_surf_indices1_[index.key].erase(index.grid_index);  // remove the index
                RecordChangedKey(index.key);
                continue;
            }
            if (updated) { RecordChangedKey(index.key); }
            if (!new_index.has_value()) { continue; }  // the surface data is not moved

            const auto &[new_key, new_grid_index, surf_index] = new_index.value();
            auto new_surf_it = m_surf_indices1_.find(new_key);
            if (new_surf_it == m_surf_indices1_.end()) {  // the new key is not in the index
                // move to the new index
                m_surf_indices1_[new_key].emplace(new_grid_index, surf_index);
                RecordChangedKey(new_key);
            } else {
                if (absl::flat_hash_map<int, std::size_t> &new_surf_indices = new_surf_it->second;
                    new_surf_indices.try_emplace(new_grid_index, surf_index).second) {
                    // move to the new index
                    RecordChangedKey(new_key);  // record the changed key
                } else {                        // the new grid index is already occupied
                    // this surface data is not used anymore
                    m_surf_data_manager_.RemoveEntry(index.surf_index);
                }
            }
            m_surf_indices1_[index.key].erase(index.grid_index);  // remove the old index
            RecordChangedKey(index.key);
        }
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::UpdateMapPoint(
        SurfData &surface_data,
        bool &updated,
        bool &to_remove) {
        const Dtype sensor_range_var = m_setting_->sensor_gp->sensor_range_var;
        const Dtype min_comp_gradient_var = m_setting_->compute_variance.min_gradient_var;
        const Dtype max_comp_gradient_var = m_setting_->compute_variance.max_gradient_var;
        const int max_adjust_tries = m_setting_->update_map_points.max_adjust_tries;
        const Dtype min_observable_occ = m_setting_->update_map_points.min_observable_occ;
        const Dtype max_surface_abs_occ = m_setting_->update_map_points.max_surface_abs_occ;
        const Dtype max_valid_gradient_var = m_setting_->update_map_points.max_valid_gradient_var;
        const Dtype min_position_var = m_setting_->update_map_points.min_position_var;
        const Dtype min_gradient_var = m_setting_->update_map_points.min_gradient_var;
        const Dtype max_bayes_position_var = m_setting_->update_map_points.max_bayes_position_var;
        const Dtype max_bayes_gradient_var = m_setting_->update_map_points.max_bayes_gradient_var;
        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();

        updated = false;
        to_remove = false;

        const Position &pos_global_old = surface_data.position;
        Position pos_local_old = sensor_frame->PosWorldToFrame(pos_global_old);
        if (!sensor_frame->PointIsInFrame(pos_local_old)) { return; }

        Dtype occ, distance_old, distance_pred;
        if (!ComputeOcc(pos_local_old, distance_old, distance_pred, occ)) { return; }
        if (occ < min_observable_occ) { return; }

        const Gradient &grad_global_old = surface_data.normal;
        Gradient grad_local_old = sensor_frame->DirWorldToFrame(grad_global_old);

        // compute a new position for the point
        Position pos_local_new = pos_local_old;
        Dtype delta = m_setting_->perturb_delta;
        int num_adjust_tries = 0;
        Dtype occ_abs = std::fabs(occ);
        Dtype distance_new = distance_old;
        while (num_adjust_tries < max_adjust_tries && occ_abs > max_surface_abs_occ) {
            // move one step
            // the direction is determined by the occupancy sign, the step size is heuristically
            // determined according to iteration.
            if (occ < 0.0f) {
                pos_local_new += grad_local_old * delta;  // the point is inside the obstacle.
            } else if (occ > 0.0f) {
                pos_local_new -= grad_local_old * delta;  // the point is outside the obstacle.
            }

            // test the new point
            Dtype occ_new;
            if (!ComputeOcc(pos_local_new, distance_new, distance_pred, occ_new)) { break; }
            occ_abs = std::fabs(occ_new);
            distance_new = distance_pred;
            if (occ_abs < max_surface_abs_occ) { break; }
            if (occ * occ_new < 0.0f) {
                delta *= 0.5f;  // too big, make it smaller
            } else {
                delta *= 1.1f;
            }
            occ = occ_new;
            ++num_adjust_tries;
        }

        // compute new gradient and uncertainty
        Dtype occ_mean, var_distance;
        Gradient grad_local_new;
        if (!ComputeGradient1(pos_local_new, grad_local_new, occ_mean, var_distance)) {
            return;  // failed to compute gradient
        }
        Gradient grad_global_new = sensor_frame->DirFrameToWorld(grad_local_new);
        Dtype var_position_new, var_gradient_new;
        ComputeVariance(
            pos_local_new,
            grad_local_new,
            distance_new,
            var_distance,
            std::fabs(occ_mean),
            occ_abs,
            false,
            var_position_new,
            var_gradient_new);

        Position pos_global_new = sensor_frame->PosFrameToWorld(pos_local_new);
        if (const Dtype var_position_old = surface_data.var_position,
            var_gradient_old = surface_data.var_normal;
            var_gradient_old <= max_valid_gradient_var) {
            // Perform bayes update when the old result is not too bad but not good enough.
            const Dtype var_position_sum = var_position_new + var_position_old;
            const Dtype var_gradient_sum = var_gradient_new + var_gradient_old;

            // position Update
            pos_global_new =
                (pos_global_new * var_position_old + pos_global_old * var_position_new) /
                var_position_sum;
            // gradient Update
            UpdateGradient(var_gradient_new, var_gradient_sum, grad_global_old, grad_global_new);

            // variance Update
            const Dtype distance = (pos_global_new - pos_global_old).norm() * 0.5f;
            var_position_new = std::max(
                var_position_new * var_position_old / var_position_sum + distance,
                sensor_range_var);
            var_gradient_new = common::ClipRange(
                var_gradient_new * var_gradient_old / var_gradient_sum + distance,
                min_comp_gradient_var,
                max_comp_gradient_var);
        }
        var_position_new = std::max(var_position_new, min_position_var);
        var_gradient_new = std::max(var_gradient_new, min_gradient_var);

        // Update the surface data
        if (var_position_new > max_bayes_position_var &&
            var_gradient_new > max_bayes_gradient_var) {
            to_remove = true;  // too bad, remove it
            return;
        }
        surface_data.position = pos_global_new;
        surface_data.normal = grad_global_new;
        surface_data.var_position = var_position_new;
        surface_data.var_normal = var_gradient_new;
        updated = true;
        ERL_DEBUG_ASSERT(
            std::abs(surface_data.normal.norm() - 1.0f) < 1.e-5f,
            "surface_data->normal.norm() = {:.6f}",
            surface_data.normal.norm());
    }

    template<typename Dtype, int Dim>
    std::pair<typename GpOccSurfaceMapping<Dtype, Dim>::Key, int>
    GpOccSurfaceMapping<Dtype, Dim>::ComputeSurfaceIndex1(const Position &pos_global) const {
        const Key new_key = m_tree_->CoordToKey(pos_global);
        const Position grid_min = m_tree_->KeyToCoord(new_key, m_tree_->GetTreeDepth()).array() -
                                  m_tree_->GetResolution() * 0.5f - 1.0e-6f;
        Eigen::Vector<int, Dim> grid_coords;
        for (long dim = 0; dim < Dim; ++dim) {
            grid_coords[dim] = static_cast<int>(
                std::floor((pos_global[dim] - grid_min[dim]) * m_surface_resolution_inv_));
        }
        const int grid_index = common::CoordsToIndex<Dim>(m_strides_, grid_coords);
        return {new_key, grid_index};
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 2, bool>
    GpOccSurfaceMapping<Dtype, Dim>::ComputeOcc(
        const Position &pos_local,
        Dtype &distance_local,
        Dtype &distance_pred,
        Dtype &occ) const {

        Scalar angle_local;
        distance_local = pos_local.norm();
        angle_local[0] = std::atan2(pos_local.y(), pos_local.x());
        bool success = m_sensor_gp_->ComputeOcc(angle_local, distance_local, distance_pred, occ);
        return success;
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 3, bool>
    GpOccSurfaceMapping<Dtype, Dim>::ComputeOcc(
        const Position &pos_local,
        Dtype &distance_local,
        Dtype &distance_pred,
        Dtype &occ) const {

        distance_local = pos_local.norm();
        bool success = m_sensor_gp_->ComputeOcc(
            pos_local / distance_local,
            distance_local,
            distance_pred,
            occ);
        return success;
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 2>
    GpOccSurfaceMapping<Dtype, Dim>::UpdateGradient(
        const Dtype var_new,
        const Dtype var_sum,
        const Gradient &grad_old,
        Gradient &grad_new) {
        const Dtype &old_x = grad_old.x();
        const Dtype &old_y = grad_old.y();
        Dtype &new_x = grad_new.x();
        Dtype &new_y = grad_new.y();
        const Dtype angle_dist =
            std::atan2(old_x * new_y - old_y * new_x, old_x * new_x + old_y * new_y) * var_new /
            var_sum;
        const Dtype sin = std::sin(angle_dist);
        const Dtype cos = std::cos(angle_dist);
        // rotate grad_old by angle_dist
        new_x = cos * old_x - sin * old_y;
        new_y = sin * old_x + cos * old_y;
        ERL_DEBUG_ASSERT(
            std::abs(grad_new.norm() - 1.0f) < 1.e-5f,
            "grad_new.norm() = {:.6f}, diff = {}.",
            grad_new.norm(),
            std::abs(grad_new.norm() - 1.0f));
    }

    template<typename Dtype, int Dim>
    template<int D>
    std::enable_if_t<D == 3>
    GpOccSurfaceMapping<Dtype, Dim>::UpdateGradient(
        const Dtype var_new,
        const Dtype var_sum,
        const Gradient &grad_old,
        Gradient &grad_new) {
        Gradient rot_axis = grad_old.cross(grad_new);
        const Dtype axis_norm = rot_axis.norm();
        if (axis_norm < 1.0e-6f) {
            rot_axis = grad_old;  // parallel
        } else {
            rot_axis /= axis_norm;
        }
        const Dtype angle_dist = std::atan2(axis_norm, grad_old.dot(grad_new)) * var_new / var_sum;
        Eigen::AngleAxis<Dtype> rot(angle_dist, rot_axis);
        grad_new = rot * grad_old;
        ERL_DEBUG_ASSERT(
            std::abs(grad_new.norm() - 1.0f) < 1.e-5f,
            "grad_new.norm() = {:.6f}",
            grad_new.norm());
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::UpdateOccupancy() {
        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        // In AddNewMeasurement(), only rays classified as hit are used. So, we use the same here to
        // avoid inconsistency. Experiments show that this achieves higher fps and better results.
        const Eigen::Map<const Positions> map_points(
            sensor_frame->GetHitPointsWorld().data()->data(),
            Dim,
            sensor_frame->GetNumHitRays());
        constexpr bool parallel = true;
        constexpr bool lazy_eval = true;
        constexpr bool discrete = true;
        m_tree_->InsertPointCloud(
            map_points,
            sensor_frame->GetTranslationVector(),
            sensor_frame->GetMaxValidRange(),
            parallel,
            lazy_eval,
            discrete);
        if (lazy_eval) {
            m_tree_->UpdateInnerOccupancy();  // update the occupancy
            m_tree_->Prune();
        }
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::AddNewMeasurement0() {
        ERL_DEBUG_ASSERT(
            m_setting_->surface_resolution <= 0,
            "AddNewMeasurement0() should only be called when the surface resolution is <= 0");

        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        const Position sensor_pos = sensor_frame->GetTranslationVector();
        const std::vector<Position> &hit_points_local = sensor_frame->GetHitPointsFrame();
        const std::vector<Position> &hit_points_global = sensor_frame->GetHitPointsWorld();
        const long num_hit_rays = sensor_frame->GetNumHitRays();
        const Dtype min_position_var = m_setting_->update_map_points.min_position_var;
        const Dtype min_gradient_var = m_setting_->update_map_points.min_gradient_var;
        const bool update_occupancy = m_setting_->update_occupancy;  // occupancy is available.

        ERL_DEBUG("Collecting new measurements");
        // collect new measurements.
        // if we iterate over the hit rays directly, some computations are unnecessary
        // [key, hit_idx, invalid_flag, occ_mean, gradient_local]
        std::vector<std::tuple<Key, long, bool, Dtype, Gradient>> new_measurements;
        new_measurements.reserve(num_hit_rays);
        KeySet new_measurement_keys;
        new_measurement_keys.reserve(num_hit_rays);
        for (long i = 0; i < num_hit_rays; ++i) {
            const Position &hit_point_global = hit_points_global[i];
            Key key = m_tree_->CoordToKey(hit_point_global);
            if (!new_measurement_keys.insert(key).second) { continue; }  // already in the set
            if (m_surf_indices0_.contains(key)) { continue; }  // the key is already in the index
            if (!update_occupancy) { m_tree_->InsertNode(key); }
            new_measurements.emplace_back(key, i, false, 0.0f, Gradient::Zero());
        }

        ERL_DEBUG("Check validity of new measurements");
#pragma omp parallel for default(none) shared(new_measurements, hit_points_local)
        for (auto &[key, hit_idx, invalid_flag, occ_mean, gradient_local]: new_measurements) {
            invalid_flag = !ComputeGradient2(hit_points_local[hit_idx], gradient_local, occ_mean);
        }

        ERL_DEBUG("Add new measurements");
        for (const auto &[key, hit_idx, invalid_flag, occ_mean, gradient_local]: new_measurements) {
            if (invalid_flag) { continue; }  // invalid measurement
            const Dtype distance = (hit_points_global[hit_idx] - sensor_pos).norm();
            Dtype var_position, var_gradient;
            ComputeVariance(
                hit_points_local[hit_idx],
                gradient_local,
                distance,
                0,
                std::fabs(occ_mean),
                0,
                true,
                var_position,
                var_gradient);
            var_position = std::max(var_position, min_position_var);
            var_gradient = std::max(var_gradient, min_gradient_var);
            m_surf_indices0_[key] = m_surf_data_manager_.AddEntry(
                hit_points_global[hit_idx],
                sensor_frame->DirFrameToWorld(gradient_local),
                var_position,
                var_gradient);
            RecordChangedKey(key);
        }
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::AddNewMeasurement1() {
        ERL_DEBUG_ASSERT(
            m_setting_->surface_resolution > 0.0f,
            "AddNewMeasurement1() should only be called when the surface resolution is > 0");

        const auto sensor_frame = m_sensor_gp_->GetSensorFrame();
        const long num_hit_rays = sensor_frame->GetNumHitRays();
        const std::vector<Position> &hit_points_global = sensor_frame->GetHitPointsWorld();
        const std::vector<Position> &hit_points_local = sensor_frame->GetHitPointsFrame();
        const Position sensor_pos = sensor_frame->GetTranslationVector();
        const Dtype min_position_var = m_setting_->update_map_points.min_position_var;
        const Dtype min_gradient_var = m_setting_->update_map_points.min_gradient_var;
        const bool update_occupancy = m_setting_->update_occupancy;  // occupancy is available.

        ERL_DEBUG("Collecting new measurements");
        // key -> grid_index set
        absl::flat_hash_map<Key, absl::flat_hash_set<int>> new_measurement_indices;
        // [hit_idx, key, grid_index, invalid_flag, occ_mean, gradient_local]
        std::vector<std::tuple<long, Key, int, bool, Dtype, Gradient>> new_measurements;
        new_measurement_indices.reserve(num_hit_rays);
        new_measurements.reserve(num_hit_rays);
        for (long i = 0; i < num_hit_rays; ++i) {
            const Position &hit_point_global = hit_points_global[i];
            const auto [key, grid_index] = ComputeSurfaceIndex1(hit_point_global);
            // the grid index is already in the set
            if (!new_measurement_indices[key].insert(grid_index).second) { continue; }
            const auto surf_it = m_surf_indices1_.find(key);
            if (!update_occupancy && surf_it == m_surf_indices1_.end()) {
                m_tree_->InsertNode(key);  // insert the key into the tree for future use
            }
            if (surf_it != m_surf_indices1_.end() && surf_it->second.contains(grid_index)) {
                continue;  // the index is already occupied
            }
            new_measurements.emplace_back(i, key, grid_index, false, 0.0f, Gradient::Zero());
        }

        ERL_DEBUG("Check validity of new measurements");
#pragma omp parallel for default(none) shared(new_measurements, hit_points_local)
        for (auto &[hit_idx, key, grid_index, invalid_flag, occ_mean, gradient_local]:
             new_measurements) {
            invalid_flag = !ComputeGradient2(hit_points_local[hit_idx], gradient_local, occ_mean);
        }

        ERL_DEBUG("Add new measurements");
        for (auto &[hit_idx, key, grid_index, invalid_flag, occ_mean, gradient_local]:
             new_measurements) {
            if (invalid_flag) { continue; }  // invalid measurement
            const Dtype distance = (hit_points_global[hit_idx] - sensor_pos).norm();
            Dtype var_position, var_gradient;
            ComputeVariance(
                hit_points_local[hit_idx],
                gradient_local,
                distance,
                0,
                std::fabs(occ_mean),
                0,
                true,
                var_position,
                var_gradient);
            var_position = std::max(var_position, min_position_var);
            var_gradient = std::max(var_gradient, min_gradient_var);
            std::size_t surf_index = m_surf_data_manager_.AddEntry(
                hit_points_global[hit_idx],
                sensor_frame->DirFrameToWorld(gradient_local),
                var_position,
                var_gradient);
            const bool inserted = m_surf_indices1_[key].try_emplace(grid_index, surf_index).second;
            (void) inserted;
            ERL_DEBUG_ASSERT(
                inserted,
                "surface index for key {} and grid index {} already exists.",
                std::string(key),
                grid_index);
            RecordChangedKey(key);
        }
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::RecordChangedKey(const Key &key) {
        m_changed_keys_.insert(m_tree_->AdjustKeyToDepth(key, m_setting_->cluster_depth));
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::ComputeGradient1(
        const Position &pos_local,
        Gradient &gradient,
        Dtype &occ_mean,
        Dtype &distance_var) {

        gradient.setZero();
        occ_mean = 0;
        distance_var = std::numeric_limits<Dtype>::infinity();

        Dtype occ[Dim << 1];
        Dtype dist_sum = 0.0f;
        Dtype dist_square_mean = 0.0f;
        const Dtype delta = m_setting_->perturb_delta;

        for (int i = 0; i < Dim; ++i) {
            for (int j: {i << 1, (i << 1) + 1}) {
                const Position pos_perturbed = pos_local + m_pos_perturb_.col(j);
                Dtype dist_local, dist_pred;
                if (!ComputeOcc(pos_perturbed, dist_local, dist_pred, occ[j])) { return false; }
                occ_mean += occ[j];
                dist_sum += dist_pred;
                dist_square_mean += dist_pred * dist_pred;
            }
            gradient[i] = (occ[i << 1] - occ[(i << 1) + 1]) / delta;
        }

        constexpr Dtype Dim2 = static_cast<Dtype>(Dim << 1);
        occ_mean /= Dim2;  // occ_mean = sum(occ) / size(occ)
        // 2*Dim samples in total, to calculate the unbiased variance
        // var(r) = sum((r_i - mean(r))^2) / (2*Dim-1)
        //        = (mean(r^2) - mean(r) * mean(r)) * (2*Dim) / (2*Dim-1)
        //        = (sum(r^2) - sum(r) * sum(r) / (2*Dim)) / (2*Dim-1)
        // to remove the numerical approximation's influence, let var(r) = var(r) / delta
        distance_var = (dist_square_mean - dist_sum * dist_sum / Dim2) / ((Dim2 - 1.0f) * delta);
        const Dtype gradient_norm = gradient.norm();
        // zero gradient
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }
        gradient /= gradient_norm;
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    GpOccSurfaceMapping<Dtype, Dim>::ComputeGradient2(
        const Eigen::Ref<const Position> &pos_local,
        Gradient &gradient,
        Dtype &occ_mean) {
        occ_mean = 0;
        gradient.setZero();

        const Dtype valid_range_min = m_setting_->sensor_gp->sensor_frame->valid_range_min;
        const Dtype valid_range_max = m_setting_->sensor_gp->sensor_frame->valid_range_max;
        const Dtype delta = m_setting_->perturb_delta;

        Dtype occ[Dim << 1];

        for (int i = 0; i < Dim; ++i) {
            for (int j: {i << 1, (i << 1) + 1}) {
                const Position pos_perturbed = pos_local + m_pos_perturb_.col(j);
                Dtype distance_pred;
                if (Dtype distance; !ComputeOcc(pos_perturbed, distance, distance_pred, occ[j]) ||
                                    distance_pred < valid_range_min ||
                                    distance_pred > valid_range_max) {
                    return false;
                }
                occ_mean += occ[j];
            }
            gradient[i] = (occ[i << 1] - occ[(i << 1) + 1]) / delta;
        }

        occ_mean /= static_cast<Dtype>(Dim << 1);
        const Dtype gradient_norm = gradient.norm();
        // zero gradient
        if (gradient_norm <= m_setting_->zero_gradient_threshold) { return false; }
        gradient /= gradient_norm;
        return true;
    }

    template<typename Dtype, int Dim>
    void
    GpOccSurfaceMapping<Dtype, Dim>::ComputeVariance(
        const Eigen::Ref<const Position> &pos_local,
        const Gradient &grad_local,
        const Dtype &distance,
        const Dtype &distance_var,
        const Dtype &occ_mean_abs,
        const Dtype &occ_abs,
        const bool new_point,
        Dtype &var_position,
        Dtype &var_gradient) const {

        auto &compute_variance = m_setting_->compute_variance;
        const Dtype &position_var_alpha = compute_variance.position_var_alpha;
        const Dtype &min_distance_var = compute_variance.min_distance_var;
        const Dtype &max_distance_var = compute_variance.max_distance_var;
        const Dtype &min_gradient_var = compute_variance.min_gradient_var;
        const Dtype &max_gradient_var = compute_variance.max_gradient_var;

        const Dtype var_distance =
            common::ClipRange(distance * distance, min_distance_var, max_distance_var);
        const Dtype cos_view_angle = -pos_local.dot(grad_local) / pos_local.norm();
        const Dtype cos2_view_angle =  // avoid zero division
            std::max(cos_view_angle * cos_view_angle, static_cast<Dtype>(1.0e-2f));
        const Dtype var_direction = (1.0f - cos2_view_angle) / cos2_view_angle;

        if (new_point) {
            var_position = position_var_alpha * (var_distance + var_direction);
            var_gradient = common::ClipRange(occ_mean_abs, min_gradient_var, max_gradient_var);
        } else {  // compute variance for update_map_points
            var_position = position_var_alpha * (var_distance + var_direction) + occ_abs;
            var_gradient =
                common::ClipRange(occ_mean_abs + distance_var, min_gradient_var, max_gradient_var) +
                0.1f * var_direction;
        }
    }

}  // namespace erl::sdf_mapping
