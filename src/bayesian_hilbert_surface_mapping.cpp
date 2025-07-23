#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"

#include "erl_common/angle_utils.hpp"
#include "erl_geometry/bayesian_hilbert_map_torch.hpp"
#include "erl_gp_sdf/surface_data_manager.hpp"

namespace erl::gp_sdf {
    template<typename Dtype>
    YAML::Node
    RaySelector2D<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, angle_min);
        ERL_YAML_SAVE_ATTR(node, setting, angle_max);
        ERL_YAML_SAVE_ATTR(node, setting, num_angles);
        return node;
    }

    template<typename Dtype>
    bool
    RaySelector2D<Dtype>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, angle_min);
        ERL_YAML_LOAD_ATTR(node, setting, angle_max);
        ERL_YAML_LOAD_ATTR(node, setting, num_angles);
        return true;
    }

    template<typename Dtype>
    RaySelector2D<Dtype>::RaySelector2D(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {
        ERL_ASSERTM(m_setting_ != nullptr, "RaySelector2D setting is nullptr.");
        m_angle_resolution_ =
            (m_setting_->angle_max - m_setting_->angle_min) / (m_setting_->num_angles - 1);
        m_ray_indices_.resize(m_setting_->num_angles);
    }

    template<typename Dtype>
    void
    RaySelector2D<Dtype>::UpdateRays(
        const Vector2 &sensor_origin,
        const Matrix2 &sensor_rotation,
        const Eigen::Ref<const Matrix2X> &ray_end_points) {
        using namespace common;

        for (auto &ray_indices: m_ray_indices_) { ray_indices.clear(); }

        for (long i = 0; i < ray_end_points.cols(); ++i) {
            Vector2 p = sensor_rotation.transpose() * (ray_end_points.col(i) - sensor_origin);
            Dtype angle = std::atan2(p.y(), p.x());
            angle = WrapAnglePi(angle);
            angle = std::max(m_setting_->angle_min, std::min(angle, m_setting_->angle_max));
            long idx = MeterToGrid(angle, m_setting_->angle_min, m_angle_resolution_);
            m_ray_indices_[idx].push_back(i);
        }
    }

    template<typename Dtype>
    void
    RaySelector2D<Dtype>::SelectRays(
        const Vector2 &sensor_origin,
        const Matrix2 &sensor_rotation,
        Vector2 point,
        Dtype radius,
        std::vector<long> &ray_indices) const {
        using namespace common;
        ray_indices.clear();

        point = sensor_rotation.transpose() * (point - sensor_origin);
        const Dtype dist = point.norm();
        const Dtype angle = std::atan2(point.y(), point.x());
        const Dtype theta = std::atan2(radius, dist);
        Dtype min_angle = WrapAnglePi(angle - theta);
        min_angle = std::max(m_setting_->angle_min, std::min(min_angle, m_setting_->angle_max));
        Dtype max_angle = WrapAnglePi(angle + theta);
        max_angle = std::max(m_setting_->angle_min, std::min(max_angle, m_setting_->angle_max));
        const auto min_idx = MeterToGrid(min_angle, m_setting_->angle_min, m_angle_resolution_);
        const auto max_idx = MeterToGrid(max_angle, m_setting_->angle_min, m_angle_resolution_);

        if (min_angle < max_angle) {
            for (long i = min_idx; i <= max_idx; ++i) {
                ray_indices.insert(
                    ray_indices.end(),
                    m_ray_indices_[i].begin(),
                    m_ray_indices_[i].end());
            }
            return;
        }

        // wrap around case
        for (long i = min_idx; i < static_cast<long>(m_ray_indices_.size()); ++i) {
            ray_indices.insert(
                ray_indices.end(),
                m_ray_indices_[i].begin(),
                m_ray_indices_[i].end());
        }
        for (long i = 0; i <= max_idx; ++i) {
            ray_indices.insert(
                ray_indices.end(),
                m_ray_indices_[i].begin(),
                m_ray_indices_[i].end());
        }
    }

    template<typename Dtype>
    YAML::Node
    RaySelector3D<Dtype>::Setting::YamlConvertImpl::encode(const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, azimuth_min);
        ERL_YAML_SAVE_ATTR(node, setting, azimuth_max);
        ERL_YAML_SAVE_ATTR(node, setting, elevation_min);
        ERL_YAML_SAVE_ATTR(node, setting, elevation_max);
        ERL_YAML_SAVE_ATTR(node, setting, num_azimuth_angles);
        ERL_YAML_SAVE_ATTR(node, setting, num_elevation_angles);
        return node;
    }

    template<typename Dtype>
    bool
    RaySelector3D<Dtype>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, azimuth_min);
        ERL_YAML_LOAD_ATTR(node, setting, azimuth_max);
        ERL_YAML_LOAD_ATTR(node, setting, elevation_min);
        ERL_YAML_LOAD_ATTR(node, setting, elevation_max);
        ERL_YAML_LOAD_ATTR(node, setting, num_azimuth_angles);
        ERL_YAML_LOAD_ATTR(node, setting, num_elevation_angles);
        return true;
    }

    template<typename Dtype>
    RaySelector3D<Dtype>::RaySelector3D(std::shared_ptr<Setting> setting)
        : m_setting_(std::move(setting)) {
        ERL_ASSERTM(m_setting_ != nullptr, "RaySelector3D setting is nullptr");
        m_azimuth_res_ = (m_setting_->azimuth_max - m_setting_->azimuth_min) /
                         (m_setting_->num_azimuth_angles - 1);
        m_elevation_res_ = (m_setting_->elevation_max - m_setting_->elevation_min) /
                           (m_setting_->num_elevation_angles - 1);
        m_ray_indices_.resize(m_setting_->num_azimuth_angles, m_setting_->num_elevation_angles);
    }

    template<typename Dtype>
    void
    RaySelector3D<Dtype>::UpdateRays(
        const Vector3 &sensor_origin,
        const Matrix3 &sensor_rotation,
        const Eigen::Ref<const Matrix3X> &ray_end_points) {
        using namespace common;
        const Dtype azimuth_min = m_setting_->azimuth_min;
        const Dtype azimuth_max = m_setting_->azimuth_max;
        const Dtype elevation_min = m_setting_->elevation_min;
        const Dtype elevation_max = m_setting_->elevation_max;

        auto *ptr = m_ray_indices_.data();
        for (long i = 0; i < m_ray_indices_.size(); ++i) { ptr[i].clear(); }

        Eigen::Matrix2X<long> coords(2, ray_end_points.cols());
#pragma omp parallel for default(none) \
    shared(ray_end_points,             \
               coords,                 \
               sensor_rotation,        \
               sensor_origin,          \
               azimuth_min,            \
               azimuth_max,            \
               elevation_min,          \
               elevation_max) schedule(static)
        for (long i = 0; i < ray_end_points.cols(); ++i) {
            Vector3 p = sensor_rotation.transpose() * (ray_end_points.col(i) - sensor_origin);
            p.normalize();
            Dtype azimuth, elevation;
            DirectionToAzimuthElevation<Dtype>(p, azimuth, elevation);
            azimuth = std::max(azimuth_min, std::min(azimuth, azimuth_max));
            elevation = std::max(elevation_min, std::min(elevation, elevation_max));
            auto coord = coords.col(i);
            coord[0] = MeterToGrid(azimuth, azimuth_min, m_azimuth_res_);
            coord[1] = MeterToGrid(elevation, elevation_min, m_elevation_res_);
        }

        for (long i = 0; i < coords.cols(); ++i) {
            auto coord = coords.col(i);
            m_ray_indices_(coord[0], coord[1]).push_back(i);
        }
    }

    template<typename Dtype>
    void
    RaySelector3D<Dtype>::SelectRays(
        const Vector3 &sensor_origin,
        const Matrix3 &sensor_rotation,
        Vector3 point,
        Dtype radius,
        std::vector<long> &ray_indices) const {
        using namespace common;
        ray_indices.clear();

        point = sensor_rotation.transpose() * (point - sensor_origin);
        const Dtype dist = point.norm();
        point /= dist;
        Dtype azimuth, elevation;
        DirectionToAzimuthElevation<Dtype>(point, azimuth, elevation);
        const Dtype theta = std::atan2(radius, dist);

        const Dtype azimuth_min = m_setting_->azimuth_min;
        const Dtype azimuth_max = m_setting_->azimuth_max;
        const Dtype elevation_min = m_setting_->elevation_min;
        const Dtype elevation_max = m_setting_->elevation_max;
        constexpr auto kPi = static_cast<Dtype>(M_PI);
        constexpr auto kPi2 = static_cast<Dtype>(M_PI_2);

        Dtype ele_max = elevation + theta;
        if (ele_max > kPi2) {
            elevation = std::min(elevation - theta, kPi - ele_max);
            elevation = std::max(elevation_min, std::min(elevation, elevation_max));
            auto ele_idx = MeterToGrid(elevation, elevation_min, m_elevation_res_);
            for (long i = ele_idx; i < m_ray_indices_.cols(); ++i) {
                auto *ptr = m_ray_indices_.col(i).data();
                for (long j = 0; j < m_ray_indices_.rows(); ++j) {
                    ray_indices.insert(ray_indices.end(), ptr[j].begin(), ptr[j].end());
                }
            }
            return;
        }

        Dtype ele_min = elevation - theta;
        if (ele_min < -kPi2) {
            elevation = std::max(elevation + theta, -kPi - ele_min);
            elevation = std::max(elevation_min, std::min(elevation, elevation_max));
            auto ele_idx = MeterToGrid(elevation, elevation_min, m_elevation_res_);
            for (long i = 0; i <= ele_idx; ++i) {
                auto *ptr = m_ray_indices_.col(i).data();
                for (long j = 0; j < m_ray_indices_.rows(); ++j) {
                    ray_indices.insert(ray_indices.end(), ptr[j].begin(), ptr[j].end());
                }
            }
            return;
        }

        ele_min = std::max(elevation_min, std::min(ele_min, elevation_max));
        ele_max = std::max(elevation_min, std::min(ele_max, elevation_max));
        auto min_eidx = MeterToGrid(ele_min, elevation_min, m_elevation_res_);
        auto max_eidx = MeterToGrid(ele_max, elevation_min, m_elevation_res_);

        Dtype azi_max = WrapAnglePi(azimuth + theta);
        Dtype azi_min = WrapAnglePi(azimuth - theta);
        if (azi_min < azi_max) {
            azi_min = std::max(azimuth_min, std::min(azi_min, azimuth_max));
            azi_max = std::max(azimuth_min, std::min(azi_max, azimuth_max));
            auto min_aidx = MeterToGrid(azi_min, azimuth_min, m_azimuth_res_);
            auto max_aidx = MeterToGrid(azi_max, azimuth_min, m_azimuth_res_);

            for (long i = min_eidx; i < max_eidx; ++i) {
                auto *ptr = m_ray_indices_.col(i).data();
                for (long j = min_aidx; j < max_aidx; ++j) {
                    ray_indices.insert(ray_indices.end(), ptr[j].begin(), ptr[j].end());
                }
            }
            return;
        }

        // wrap around case
        azi_min = std::max(azimuth_min, std::min(azi_min, azimuth_max));
        azi_max = std::max(azimuth_min, std::min(azi_max, azimuth_max));
        auto min_aidx = MeterToGrid(azi_min, azimuth_min, m_azimuth_res_);
        auto max_aidx = MeterToGrid(azi_max, azimuth_min, m_azimuth_res_);
        for (long i = min_eidx; i < max_eidx; ++i) {
            auto *ptr = m_ray_indices_.col(i).data();
            for (long j = min_aidx; j < m_ray_indices_.rows(); ++j) {
                ray_indices.insert(ray_indices.end(), ptr[j].begin(), ptr[j].end());
            }
            for (long j = 0; j <= max_aidx; ++j) {
                ray_indices.insert(ray_indices.end(), ptr[j].begin(), ptr[j].end());
            }
        }
    }

    template<typename Dtype>
    YAML::Node
    LocalBayesianHilbertMapSetting<Dtype>::YamlConvertImpl::encode(
        const LocalBayesianHilbertMapSetting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, bhm);
        ERL_YAML_SAVE_ATTR(node, setting, kernel_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel_setting_type);
        ERL_YAML_SAVE_ATTR(node, setting, kernel);
        ERL_YAML_SAVE_ATTR(node, setting, max_dataset_size);
        ERL_YAML_SAVE_ATTR(node, setting, hit_buffer_size);
        ERL_YAML_SAVE_ATTR(node, setting, surface_resolution);
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
        ERL_YAML_LOAD_ATTR(node, setting, max_dataset_size);
        ERL_YAML_LOAD_ATTR(node, setting, hit_buffer_size);
        ERL_YAML_LOAD_ATTR(node, setting, surface_resolution);
        return true;
    }

    template<typename Dtype, int Dim>
    YAML::Node
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Setting::YamlConvertImpl::encode(
        const Setting &setting) {
        YAML::Node node;
        ERL_YAML_SAVE_ATTR(node, setting, local_bhm);
        ERL_YAML_SAVE_ATTR(node, setting, ray_selector);
        ERL_YAML_SAVE_ATTR(node, setting, tree);
        ERL_YAML_SAVE_ATTR(node, setting, hinged_grid_size);
        ERL_YAML_SAVE_ATTR(node, setting, surface_max_abs_logodd);
        ERL_YAML_SAVE_ATTR(node, setting, surface_bad_abs_logodd);
        ERL_YAML_SAVE_ATTR(node, setting, surface_step_size);
        ERL_YAML_SAVE_ATTR(node, setting, faster_prediction);
        ERL_YAML_SAVE_ATTR(node, setting, max_adjust_tries);
        ERL_YAML_SAVE_ATTR(node, setting, var_scale);
        ERL_YAML_SAVE_ATTR(node, setting, var_max);
        ERL_YAML_SAVE_ATTR(node, setting, scaling);
        ERL_YAML_SAVE_ATTR(node, setting, bhm_depth);
        ERL_YAML_SAVE_ATTR(node, setting, bhm_overlap);
        ERL_YAML_SAVE_ATTR(node, setting, update_with_cuda);
        ERL_YAML_SAVE_ATTR(node, setting, cuda_device_id);
        ERL_YAML_SAVE_ATTR(node, setting, update_batch_size);
        ERL_YAML_SAVE_ATTR(node, setting, bhm_test_margin);
        ERL_YAML_SAVE_ATTR(node, setting, test_knn);
        ERL_YAML_SAVE_ATTR(node, setting, test_batch_size);
        ERL_YAML_SAVE_ATTR(node, setting, build_bhm_on_hit);
        return node;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Setting::YamlConvertImpl::decode(
        const YAML::Node &node,
        Setting &setting) {
        if (!node.IsMap()) { return false; }
        if (!ERL_YAML_LOAD_ATTR(node, setting, local_bhm)) { return false; }
        if (!ERL_YAML_LOAD_ATTR(node, setting, ray_selector)) { return false; }
        if (!ERL_YAML_LOAD_ATTR(node, setting, tree)) { return false; }
        ERL_YAML_LOAD_ATTR(node, setting, hinged_grid_size);
        ERL_YAML_LOAD_ATTR(node, setting, surface_max_abs_logodd);
        ERL_YAML_LOAD_ATTR(node, setting, surface_bad_abs_logodd);
        ERL_YAML_LOAD_ATTR(node, setting, surface_step_size);
        ERL_YAML_LOAD_ATTR(node, setting, faster_prediction);
        ERL_YAML_LOAD_ATTR(node, setting, max_adjust_tries);
        ERL_YAML_LOAD_ATTR(node, setting, var_scale);
        ERL_YAML_LOAD_ATTR(node, setting, var_max);
        ERL_YAML_LOAD_ATTR(node, setting, scaling);
        ERL_YAML_LOAD_ATTR(node, setting, bhm_depth);
        ERL_YAML_LOAD_ATTR(node, setting, bhm_overlap);
        ERL_YAML_LOAD_ATTR(node, setting, update_with_cuda);
        ERL_YAML_LOAD_ATTR(node, setting, cuda_device_id);
        ERL_YAML_LOAD_ATTR(node, setting, update_batch_size);
        ERL_YAML_LOAD_ATTR(node, setting, bhm_test_margin);
        ERL_YAML_LOAD_ATTR(node, setting, test_knn);
        ERL_YAML_LOAD_ATTR(node, setting, test_batch_size);
        ERL_YAML_LOAD_ATTR(node, setting, build_bhm_on_hit);
        return true;
    }

    template<typename Dtype, int Dim>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::LocalBayesianHilbertMap(
        std::shared_ptr<LocalBayesianHilbertMapSetting<Dtype>> setting_,
        Positions hinged_points,
        Aabb map_boundary,
        uint64_t seed,
        std::optional<Aabb> track_surface_boundary_)
        : setting(std::move(setting_)),
          tracked_surface_boundary(
              track_surface_boundary_.has_value() ? track_surface_boundary_.value() : map_boundary),
          bhm(setting->bhm,
              Covariance::CreateCovariance(setting->kernel_type, setting->kernel),
              std::move(hinged_points),
              std::move(map_boundary),
              seed) {

        const Position map_size = tracked_surface_boundary.sizes();
        for (long dim = 0; dim < Dim; ++dim) {
            Dtype dim_shape = map_size[dim] / setting->surface_resolution;
            grid_shape[dim] = static_cast<int>(std::round(dim_shape));
        }
        strides = common::ComputeCStrides<int>(grid_shape, 1);

        if (setting->hit_buffer_size > 0) { hit_buffer.reserve(setting->hit_buffer_size); }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::GenerateDataset(
        const Eigen::Ref<const Position> &sensor_origin,
        const Eigen::Ref<const Positions> &points,
        const std::vector<long> &point_indices) {
        const long max_dataset_size = setting->max_dataset_size;
        bhm.GenerateDataset(
            sensor_origin,
            points,
            point_indices,
            max_dataset_size,
            num_dataset_points,
            dataset_points,
            dataset_labels,
            hit_indices);

        if (num_dataset_points == 0) { return; }
        if (!hit_buffer.empty() &&
            (max_dataset_size < 0 || num_dataset_points < max_dataset_size)) {

            // there is data in the hit buffer, and
            // 1. no dataset size limit
            // 2. dataset size limit but not reached
            long n = num_dataset_points + static_cast<long>(hit_buffer.size());
            if (max_dataset_size > 0) { n = std::min(max_dataset_size, n); }
            if (n > dataset_points.cols()) {
                dataset_points.conservativeResize(Dim, n);
                dataset_labels.conservativeResize(n);
            }
            for (const Position &point: hit_buffer) {
                dataset_points.col(num_dataset_points) = point;
                dataset_labels[num_dataset_points] = 1;
                ++num_dataset_points;
                if (num_dataset_points >= n) { break; }  // the dataset size limit is reached
            }
        }
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::Update(
        const Eigen::Ref<const Position> &sensor_origin,
        const Eigen::Ref<const Positions> &points,
        const std::vector<long> &point_indices) {

        GenerateDataset(sensor_origin, points, point_indices);
        if (num_dataset_points == 0) { return false; }
        bhm.RunExpectationMaximization(dataset_points, dataset_labels, num_dataset_points);
        UpdateHitBuffer(points);
        return true;
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::UpdateHitBuffer(
        const Eigen::Ref<const Positions> &points) {
        if (setting->hit_buffer_size > 0 && !hit_indices.empty()) {
            // hit buffer has space and there are hit points
            for (const long &hit_index: hit_indices) {
                hit_buffer[hit_buffer_head] = points.col(hit_index);
                hit_buffer_head = (hit_buffer_head + 1) % hit_buffer.capacity();
            }
        }
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::Write(
        std::ostream &s) const {
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
                "num_dataset_points",
                [](const LocalBayesianHilbertMap *self, std::ostream &stream) {
                    stream.write(
                        reinterpret_cast<const char *>(&self->num_dataset_points),
                        sizeof(self->num_dataset_points));
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
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::Read(std::istream &s) {
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
                        int index;
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
                "num_dataset_points",
                [](LocalBayesianHilbertMap *self, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&self->num_dataset_points),
                        sizeof(self->num_dataset_points));
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
        };
        return ReadTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::operator==(
        const LocalBayesianHilbertMap &other) const {
        if (setting == nullptr && other.setting != nullptr) { return false; }
        if (setting != nullptr && (other.setting == nullptr || *setting != *other.setting)) {
            return false;
        }
        if (tracked_surface_boundary != other.tracked_surface_boundary) { return false; }
        if (bhm != other.bhm) { return false; }
        if (strides != other.strides) { return false; }
        if (surface_indices != other.surface_indices) { return false; }
        if (num_dataset_points != other.num_dataset_points) { return false; }

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
    BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap::operator!=(
        const LocalBayesianHilbertMap &other) const {
        return !(*this == other);
    }

    template<typename Dtype, int Dim>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::BayesianHilbertSurfaceMapping(
        std::shared_ptr<Setting> setting)
        : m_setting_(NotNull(std::move(setting), true, "setting is nullptr")),
          m_ray_selector_(m_setting_->ray_selector) {
        m_tree_ = std::make_shared<Tree>(m_setting_->tree);
        GenerateHingedPoints();

        // do it on the main thread only
#pragma omp parallel default(none)
#pragma omp critical
        {
            if (omp_get_thread_num() == 0) {
                m_ray_indices_.resize(omp_get_num_threads());
                for (auto &indices: m_ray_indices_) { indices.reserve(10240); }
            }
        }
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Setting>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetSetting() const {
        return m_setting_;
    }

    template<typename Dtype, int Dim>
    std::shared_ptr<const typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Tree>
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetTree() const {
        return m_tree_;
    }

    template<typename Dtype, int Dim>
    const absl::flat_hash_map<
        typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Key,
        std::shared_ptr<
            typename BayesianHilbertSurfaceMapping<Dtype, Dim>::LocalBayesianHilbertMap>> &
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetLocalBayesianHilbertMaps() const {
        return m_key_bhm_dict_;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Update(
        const Eigen::Ref<const Rotation> &rotation,
        const Eigen::Ref<const Translation> &translation,
        const Eigen::Ref<const Ranges> &scan,
        const bool are_points,
        const bool are_local) {
        ERL_ASSERTM(are_points, "scan must be points, not range data.");

        if (scan.cols() == 0) {
            ERL_WARN("No points in the scan, nothing to update.");
            return false;
        }

        Positions points;
        if (are_local) {
            points.resize(Dim, scan.cols());
#pragma omp parallel for default(none) shared(scan, points, rotation, translation) schedule(static)
            for (long i = 0; i < scan.cols(); ++i) {
                points.col(i) = rotation * scan.col(i) + translation;
            }
        } else {
            points = scan;
        }
        {
            ERL_BLOCK_TIMER_MSG("BHSM Update");
            return Update(rotation, translation, points, true /*parallel*/);
        }
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Update(
        const Eigen::Ref<const Rotation> &sensor_rotation,
        const Eigen::Ref<const Position> &sensor_origin,
        const Eigen::Ref<const Positions> &points,
        const bool parallel) {

        if (points.cols() == 0) {
            ERL_WARN("No points in the scan, nothing to update.");
            return false;
        }

        Position sensor_origin_s = sensor_origin;
        Positions points_s = points;
        if (m_setting_->scaling != 1.0f) {
            sensor_origin_s.array() *= m_setting_->scaling;
            points_s.array() *= m_setting_->scaling;
        }

        // to update the occupancy tree first, the resolution of the tree should not be too high so
        // that we will not spend too much time on the tree update. the tree helps us find where to
        // place local Bayesian Hilbert maps.
        constexpr bool discrete = true;
        {
            constexpr bool lazy_eval = true;
            ERL_BLOCK_TIMER_MSG("tree update");
            m_tree_->InsertPointCloud(
                points_s,
                sensor_origin_s,
                m_setting_->local_bhm->bhm->min_distance,
                m_setting_->local_bhm->bhm->max_distance,
                true /*parallel*/,
                lazy_eval,
                discrete);
            if (lazy_eval) {
                m_tree_->UpdateInnerOccupancy();
                m_tree_->Prune();
            }
        }

        // find the local Bayesian Hilbert maps to build or update
        KeySet &bhm_keys_set = m_changed_clusters_;
        bhm_keys_set.clear();
        const uint32_t bhm_depth = m_setting_->bhm_depth;
        {
            const auto &end_point_maps =
                discrete ? m_tree_->GetDiscreteEndPointMaps() : m_tree_->GetEndPointMaps();
            ERL_BLOCK_TIMER_MSG("bhm find");
            if (m_setting_->build_bhm_on_hit) {
                // any hit point will trigger building the corresponding local Bayesian Hilbert map
                for (const auto &[key, hit_indices]: end_point_maps) {
                    bhm_keys_set.insert(m_tree_->AdjustKeyToDepth(key, bhm_depth)).second;
                }
            } else {
                // only the occupied node will trigger building the corresponding local BHM
                for (const auto &[key, hit_indices]: end_point_maps) {
                    if (const TreeNode *node = m_tree_->Search(key);
                        node != nullptr && m_tree_->IsNodeOccupied(node)) {
                        bhm_keys_set.insert(m_tree_->AdjustKeyToDepth(key, bhm_depth));
                    }
                }
            }
        }

        // create bhm for new keys
        KeyVector bhm_keys(bhm_keys_set.begin(), bhm_keys_set.end());
        const Dtype half_surface_size = m_tree_->GetNodeSize(bhm_depth) * 0.5f;
        const Dtype half_bhm_size = half_surface_size + m_setting_->bhm_overlap;
        {
            ERL_BLOCK_TIMER_MSG("bhm create");
            for (const Key &key: bhm_keys) {
                auto it = m_key_bhm_dict_.find(key);
                if (it != m_key_bhm_dict_.end()) { continue; }  // already exist

                Position map_center = m_tree_->KeyToCoord(key, bhm_depth);
                Positions hinged_points = m_hinged_points_.colwise() + map_center;
                m_key_bhm_positions_.emplace_back(key, map_center);
                m_key_bhm_dict_.insert({
                    key,
                    std::make_shared<LocalBayesianHilbertMap>(
                        m_setting_->local_bhm,
                        hinged_points,
                        Aabb(map_center, half_bhm_size) /*map_boundary*/,
                        typename Key::KeyHash()(key) /*seed*/,
                        Aabb(map_center, half_surface_size) /*track_surface_boundary*/),
                });
                // need to update the kdtree after adding new bhm
                m_bhm_kdtree_needs_update_ = true;
            }
        }

        m_ray_selector_.UpdateRays(sensor_origin_s, sensor_rotation, points_s);
        const Dtype radius = std::sqrt(static_cast<Dtype>(Dim) * half_bhm_size * half_bhm_size);

        // update the local Bayesian Hilbert maps
        (void) parallel;
        std::vector<int> updated(bhm_keys.size(), 0);  // don't use bool, it is not atomic
        ERL_INFO("{} local bhm(s) to update", bhm_keys.size());

#ifdef ERL_USE_LIBTORCH
        bool use_cuda = m_setting_->update_with_cuda;
        if (use_cuda && !torch::cuda::is_available()) {
            ERL_WARN_ONCE(
                "update_with_cuda is set to true, but CUDA is not available. "
                "Falling back to CPU update.");
            use_cuda = false;
        }
        if (use_cuda) {
            ERL_BLOCK_TIMER_MSG("bhm update");
            // generate dataset
    #pragma omp parallel for if (parallel) default(none)                              \
        shared(bhm_keys, points_s, sensor_origin_s, sensor_rotation, radius, updated) \
        schedule(static)
            for (std::size_t i = 0; i < bhm_keys.size(); ++i) {
                auto &bhm_key = bhm_keys[i];
                std::vector<long> &ray_indices = m_ray_indices_[omp_get_thread_num()];
                // ray_indices.reserve(points_s.cols() / 10);
                LocalBayesianHilbertMap &local_bhm = *m_key_bhm_dict_[bhm_key];
                local_bhm.active = true;
                m_ray_selector_.SelectRays(
                    sensor_origin_s,
                    sensor_rotation,
                    local_bhm.bhm.GetMapBoundary().center,
                    radius,
                    ray_indices);
                local_bhm.GenerateDataset(sensor_origin_s, points_s, ray_indices);
            }

            geometry::BayesianHilbertMapTorch<Dtype, Dim> bhm_torch(m_setting_->local_bhm->bhm);
            bhm_torch.SetDevice(torch::Device(torch::kCUDA, m_setting_->cuda_device_id));
            bhm_torch.SetKernelScale(m_setting_->local_bhm->kernel->scale);
            bhm_torch.LoadHingedPoints(m_hinged_points_);

            for (std::size_t i = 0; i < bhm_keys.size(); i += m_setting_->update_batch_size) {
                std::size_t end = std::min(i + m_setting_->update_batch_size, bhm_keys.size());
                std::vector<std::pair<std::size_t, long>> map_indices;
                map_indices.reserve(end - i);

                {
                    ERL_BLOCK_TIMER_MSG("bhm load weights and dataset");
                    long num_maps = 0;
                    long max_dataset_size = m_setting_->local_bhm->max_dataset_size;
                    for (std::size_t j = i; j < end; ++j) {
                        auto &bhm_key = bhm_keys[j];
                        LocalBayesianHilbertMap &local_bhm = *m_key_bhm_dict_[bhm_key];
                        if (local_bhm.num_dataset_points == 0) { continue; }
                        map_indices.emplace_back(j, num_maps);
                        ++num_maps;
                        max_dataset_size = std::max(max_dataset_size, local_bhm.num_dataset_points);
                    }
                    ERL_INFO("num_maps: {}, max_dataset_size: {}", num_maps, max_dataset_size);
                    bhm_torch.PrepareMemory(num_maps, max_dataset_size);

    #pragma omp parallel for if (parallel) default(none) \
        shared(bhm_keys, bhm_torch, points_s, sensor_origin_s, map_indices) schedule(static)
                    for (auto &[j, k]: map_indices) {
                        auto &bhm_key = bhm_keys[j];
                        LocalBayesianHilbertMap &local_bhm = *m_key_bhm_dict_[bhm_key];
                        bhm_torch.LoadWeightsAndCovariance(
                            k,
                            local_bhm.bhm.GetWeights(),
                            local_bhm.bhm.GetWeightsCovariance());
                        bhm_torch.LoadDataset(
                            k,
                            local_bhm.bhm.GetMapBoundary().center,
                            local_bhm.dataset_points,
                            local_bhm.dataset_labels,
                            local_bhm.num_dataset_points);
                        local_bhm.UpdateHitBuffer(points_s);
                    }
                }

                {
                    ERL_BLOCK_TIMER_MSG("bhm em");
                    bhm_torch.RunExpectationMaximization();
                }

                {
                    ERL_BLOCK_TIMER_MSG("bhm get weights and covariance");
    #pragma omp parallel for if (parallel) default(none) \
        shared(bhm_keys, bhm_torch, map_indices, updated) schedule(static)
                    for (auto &[j, k]: map_indices) {
                        auto &bhm_key = bhm_keys[j];
                        LocalBayesianHilbertMap &local_bhm = *m_key_bhm_dict_[bhm_key];
                        bhm_torch.GetWeightsAndCovariance(
                            k,
                            local_bhm.bhm.GetWeights(),
                            local_bhm.bhm.GetWeightsCovariance());
                        updated[j] = 1;
                    }
                }
            }
            bhm_torch.Reset();
        }
#else
        constexpr bool use_cuda = false;
        ERL_WARN_ONCE_COND(
            m_setting_->update_with_cuda,
            "update_with_cuda is set to true, but the erl_geometry library is not compiled with "
            "CUDA support.");
#endif

        if (!use_cuda) {
            ERL_BLOCK_TIMER_MSG("bhm update");
#pragma omp parallel for if (parallel) default(none) schedule(dynamic) \
    shared(bhm_keys, points_s, sensor_origin_s, sensor_rotation, radius, updated)
            for (std::size_t i = 0; i < bhm_keys.size(); ++i) {
                auto &bhm_key = bhm_keys[i];
                std::vector<long> &ray_indices = m_ray_indices_[omp_get_thread_num()];
                // ray_indices.reserve(points_s.cols() / 10);
                LocalBayesianHilbertMap &local_bhm = *m_key_bhm_dict_[bhm_key];
                local_bhm.active = true;
                m_ray_selector_.SelectRays(
                    sensor_origin_s,
                    sensor_rotation,
                    local_bhm.bhm.GetMapBoundary().center,
                    radius,
                    ray_indices);
                updated[i] = local_bhm.Update(sensor_origin_s, points_s, ray_indices);
            }
        }

        // turn off local Bayesian Hilbert maps whose node in the tree is not occupied
        for (auto &[key, local_bhm]: m_key_bhm_dict_) {
            if (!local_bhm->active) { continue; }
            const TreeNode *node = m_tree_->Search(key, bhm_depth);
            if (node == nullptr || !m_tree_->IsNodeOccupied(node)) { local_bhm->active = false; }
        }

        const bool any_update =
            std::any_of(updated.begin(), updated.end(), [](const int i) { return i > 0; });

        if (any_update) {
            ERL_BLOCK_TIMER_MSG("bhm update map points");
            UpdateMapPoints(points_s);
        }
        return any_update;
    }

    template<typename Dtype, int Dim>
    typename AbstractSurfaceMapping<Dtype, Dim>::SurfDataManager::Iterator
    BayesianHilbertSurfaceMapping<Dtype, Dim>::BeginSurfaceData() {
        return this->m_surf_data_manager_.begin();
    }

    template<typename Dtype, int Dim>
    typename AbstractSurfaceMapping<Dtype, Dim>::SurfDataManager::Iterator
    BayesianHilbertSurfaceMapping<Dtype, Dim>::EndSurfaceData() {
        return this->m_surf_data_manager_.end();
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Predict(
        const Eigen::Ref<const Positions> &points,
        const bool logodd,
        const bool faster,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        VectorX &prob_occupied,
        Gradients &gradient) const {

        Positions points_s = points;
        if (m_setting_->scaling != 1.0f) { points_s.array() *= m_setting_->scaling; }

        const long num_points = points_s.cols();
        if (prob_occupied.size() < num_points) { prob_occupied.resize(num_points); }
        if (compute_gradient && gradient.cols() < num_points) { gradient.resize(Dim, num_points); }
        prob_occupied.fill(logodd ? 0.0f : 0.5f);  // initialize to unknown
        gradient.fill(0.0f);                       // initialize to zero
        BuildBhmKdtree();

        const int batch_size = m_setting_->test_batch_size;
        if (batch_size > num_points) {  // no need to run in parallel
            PredictThread(
                points_s.data(),
                0,
                num_points,
                logodd,
                faster,
                compute_gradient,
                gradient_with_sigmoid,
                parallel,  // let the thread decide
                prob_occupied.data(),
                gradient.data());
            return;
        }

        const uint32_t num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        for (long i = 0; i < num_points; i += batch_size) {
            long start = i;
            long end = std::min(i + batch_size, num_points);

            bool no_free = threads.size() >= num_threads;
            while (no_free) {
                for (auto it = threads.begin(); it != threads.end(); ++it) {
                    if (!it->joinable()) { continue; }
                    it->join();
                    threads.erase(it);
                    no_free = false;
                    break;
                }
                if (no_free) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); }
            }
            threads.emplace_back(
                &BayesianHilbertSurfaceMapping::PredictThread,
                this,
                points_s.data(),
                start,
                end,
                logodd,
                faster,
                compute_gradient,
                gradient_with_sigmoid,
                false /*parallel*/,  // no need to run in parallel within the thread
                prob_occupied.data(),
                gradient.data());
        }

        for (auto &thread: threads) { thread.join(); }
        threads.clear();
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::PredictGradient(
        const Eigen::Ref<const Positions> &points,
        const bool faster,
        const bool with_sigmoid,
        const bool parallel,
        Gradients &gradient) const {

        // we can only predict the gradient when bhm is available for the point

        if (gradient.cols() < points.cols()) { gradient.resize(Dim, points.cols()); }
        gradient.fill(0.0f);
        BuildBhmKdtree();

        absl::flat_hash_map<Key, std::vector<long>> key_to_point_indices;
        key_to_point_indices.reserve(points.size());
        std::vector<Key> bhm_keys_set;
        bhm_keys_set.reserve(points.size());
        const uint32_t bhm_depth = m_setting_->bhm_depth;
        const Dtype half_size =
            0.5f * m_tree_->GetNodeSize(bhm_depth) + m_setting_->bhm_test_margin;
        long bhm_index = -1;
        Dtype bhm_distance = 0;
        for (long i = 0; i < points.size(); ++i) {
            m_bhm_kdtree_->Nearest(points.col(i), bhm_index, bhm_distance);  // find the nearest bhm
            bhm_distance = std::sqrt(bhm_distance);                          // distance is squared
            if (bhm_distance > half_size) { continue; }                      // too far from the bhm

            const Key bhm_key = m_key_bhm_positions_[bhm_index].first;
            auto it = key_to_point_indices.insert({bhm_key, std::vector<long>()});
            // if the key is new, add it to the set
            if (it.second) { bhm_keys_set.push_back(bhm_key); }
            it.first->second.push_back(i);
        }

        // dynamic scheduling because some local Bayesian Hilbert maps may have more points.
#pragma omp parallel for if (parallel) default(none) schedule(dynamic) \
    shared(bhm_keys_set, key_to_point_indices, points, faster, with_sigmoid, parallel, gradient)
        for (const Key &bhm_key: bhm_keys_set) {
            const auto &indices = key_to_point_indices[bhm_key];

            // copy the points of this key to a new matrix
            Positions points_of_key(Dim, static_cast<long>(indices.size()));
            for (long i = 0; i < points_of_key.cols(); ++i) {
                points_of_key.col(i) = points.col(indices[i]);
            }

            // predict
            Gradients gradients_of_key;
            BayesianHilbertMap &bhm = m_key_bhm_dict_.at(bhm_key)->bhm;
            bhm.PredictGradient(points_of_key, faster, with_sigmoid, !parallel, gradients_of_key);

            // copy the results back to the original matrix
            for (long i = 0; i < points_of_key.cols(); ++i) {
                gradient.col(indices[i]) = gradients_of_key.col(i);
            }
        }
    }

    template<typename Dtype, int Dim>
    Dtype
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetScaling() const {
        return m_setting_->scaling;
    }

    template<typename Dtype, int Dim>
    Dtype
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetClusterSize() const {
        return m_tree_->GetNodeSize(m_setting_->bhm_depth);
    }

    template<typename Dtype, int Dim>
    typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Position
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetClusterCenter(const Key &key) const {
        return m_tree_->KeyToCoord(key, m_setting_->bhm_depth);
    }

    template<typename Dtype, int Dim>
    const typename BayesianHilbertSurfaceMapping<Dtype, Dim>::KeySet &
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetChangedClusters() const {
        return m_changed_clusters_;
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::IterateClustersInAabb(
        const Aabb &aabb,
        std::function<void(const Key &)> callback) const {
        const uint32_t cluster_depth = m_setting_->bhm_depth;
        for (auto it = m_tree_->BeginTreeInAabb(aabb, cluster_depth),
                  end = m_tree_->EndTreeInAabb();
             it != end;
             ++it) {
            if (it->GetDepth() != cluster_depth) { continue; }
            callback(m_tree_->AdjustKeyToDepth(it.GetKey(), cluster_depth));
        }
    }

    template<typename Dtype, int Dim>
    const std::vector<typename BayesianHilbertSurfaceMapping<Dtype, Dim>::SurfData> &
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetSurfaceDataBuffer() const {
        return m_surf_data_manager_.GetBuffer();
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::CollectSurfaceDataInAabb(
        const Aabb &aabb,
        std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const {
        surface_data_indices.clear();
        for (auto it = m_tree_->BeginTreeInAabb(aabb, m_setting_->bhm_depth),
                  end = m_tree_->EndTreeInAabb();
             it != end;
             ++it) {
            if (it.GetDepth() != m_setting_->bhm_depth) { continue; }
            Key key = it.GetIndexKey();
            auto bhm_it = m_key_bhm_dict_.find(key);
            if (bhm_it == m_key_bhm_dict_.end()) { continue; }
            const LocalBayesianHilbertMap &local_bhm = *bhm_it->second;
            for (const auto &[local_idx, surf_idx]: local_bhm.surface_indices) {
                ERL_DEBUG_ASSERT(static_cast<long>(surf_idx) != -1l, "surf_idx should not be -1");
                const SurfData &surf_data = m_surf_data_manager_[surf_idx];
                surface_data_indices.emplace_back(
                    (aabb.center - surf_data.position).norm(),
                    surf_idx);
            }
        }
    }

    template<typename Dtype, int Dim>
    typename BayesianHilbertSurfaceMapping<Dtype, Dim>::Aabb
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GetMapBoundary() const {
        Position min, max;
        m_tree_->GetMetricMinMax(min, max);
        return Aabb(min, max);
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::IsInFreeSpace(
        const Positions &positions,
        VectorX &in_free_space) const {
        if (positions.cols() == 0) {
            ERL_WARN("No points in the positions, nothing to check.");
            return false;
        }
        in_free_space.resize(positions.cols());
        Gradients gradients;
        Predict(
            positions,
            true /*logodd*/,
            m_setting_->faster_prediction,
            false /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            true /*parallel*/,
            in_free_space,
            gradients);
        for (long i = 0; i < in_free_space.size(); ++i) {
            in_free_space[i] = (in_free_space[i] < 0.0f) ? 1.0f : -1.0f;  // convert to binary
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::operator==(const Super &other) const {
        const auto *other_ptr = dynamic_cast<const BayesianHilbertSurfaceMapping *>(&other);
        if (other_ptr == nullptr) { return false; }
        if (m_setting_ == nullptr && other_ptr->m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr &&
            (other_ptr->m_setting_ == nullptr || *m_setting_ != *other_ptr->m_setting_)) {
            return false;
        }
        if (m_tree_ == nullptr && other_ptr->m_tree_ != nullptr) { return false; }
        if (m_tree_ != nullptr &&
            (other_ptr->m_tree_ == nullptr || *m_tree_ != *other_ptr->m_tree_)) {
            return false;
        }
        if (m_hinged_points_ != other_ptr->m_hinged_points_) { return false; }
        if (m_key_bhm_positions_ != other_ptr->m_key_bhm_positions_) { return false; }
        // because m_key_bhm_dict_ maps a key to a shared pointer,
        // we cannot use the operator!= directly.
        if (m_key_bhm_dict_.size() != other_ptr->m_key_bhm_dict_.size()) { return false; }
        for (const auto &[key, bhm_ptr]: m_key_bhm_dict_) {
            auto it = other_ptr->m_key_bhm_dict_.find(key);
            if (it == other_ptr->m_key_bhm_dict_.end()) { return false; }
            if (bhm_ptr == nullptr && it->second != nullptr) { return false; }
            if (bhm_ptr != nullptr && (it->second == nullptr || *bhm_ptr != *(it->second))) {
                return false;
            }
        }
        return true;
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Write(std::ostream &s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<BayesianHilbertSurfaceMapping> pairs = {
            {
                "setting",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &stream) {
                    return self->m_setting_->Write(stream) && stream.good();
                },
            },
            {
                "tree",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &stream) {
                    return self->m_tree_->Write(stream) && stream.good();
                },
            },
            {
                "hinged_points",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &stream) {
                    return SaveEigenMatrixToBinaryStream(stream, self->m_hinged_points_) &&
                           stream.good();
                },
            },
            {
                "key_bhm_positions",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &stream) {
                    const std::size_t n = self->m_key_bhm_positions_.size();
                    stream.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (const auto &[key, center]: self->m_key_bhm_positions_) {
                        stream.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                        stream.write(
                            reinterpret_cast<const char *>(center.data()),
                            sizeof(Position));
                    }
                    return stream.good();
                },
            },
            {
                "key_bhm_dict",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &stream) {
                    const std::size_t n = self->m_key_bhm_dict_.size();
                    stream.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (const auto &[key, bhm]: self->m_key_bhm_dict_) {
                        stream.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                        const bool has_bhm = bhm != nullptr;
                        stream.write(reinterpret_cast<const char *>(&has_bhm), sizeof(bool));
                        if (has_bhm && !bhm->Write(stream)) { return false; }
                    }
                    return stream.good();
                },
            },
            {
                "surf_data_manager",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &stream) {
                    return self->m_surf_data_manager_.Write(stream) && stream.good();
                },
            },
            {
                "changed_clusters",
                [](const BayesianHilbertSurfaceMapping *self, std::ostream &stream) {
                    const std::size_t n = self->m_changed_clusters_.size();
                    stream.write(reinterpret_cast<const char *>(&n), sizeof(std::size_t));
                    for (const Key &key: self->m_changed_clusters_) {
                        stream.write(reinterpret_cast<const char *>(&key), sizeof(Key));
                    }
                    return stream.good();
                },
            },
        };
        return WriteTokens(s, this, pairs);
    }

    template<typename Dtype, int Dim>
    bool
    BayesianHilbertSurfaceMapping<Dtype, Dim>::Read(std::istream &s) {
        using namespace common;
        m_bhm_kdtree_needs_update_ = true;
        static const TokenReadFunctionPairs<BayesianHilbertSurfaceMapping> pairs = {
            {
                "setting",
                [](BayesianHilbertSurfaceMapping *self, std::istream &stream) {
                    return self->m_setting_->Read(stream) && stream.good();
                },
            },
            {
                "tree",
                [](BayesianHilbertSurfaceMapping *self, std::istream &stream) {
                    return self->m_tree_->Read(stream) && stream.good();
                },
            },

            {
                "hinged_points",
                [](BayesianHilbertSurfaceMapping *self, std::istream &stream) {
                    return LoadEigenMatrixFromBinaryStream(stream, self->m_hinged_points_) &&
                           stream.good();
                },
            },
            {
                "key_bhm_positions",
                [](BayesianHilbertSurfaceMapping *self, std::istream &stream) {
                    std::size_t n;
                    stream.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_key_bhm_positions_.clear();
                    self->m_key_bhm_positions_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        Position center;
                        stream.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        stream.read(reinterpret_cast<char *>(center.data()), sizeof(Position));
                        self->m_key_bhm_positions_.emplace_back(key, center);
                    }
                    return stream.good();
                },
            },
            {
                "key_bhm_dict",
                [](BayesianHilbertSurfaceMapping *self, std::istream &stream) {
                    std::size_t n;
                    stream.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_key_bhm_dict_.clear();
                    self->m_key_bhm_dict_.reserve(n);
                    const uint32_t bhm_depth = self->m_setting_->bhm_depth;
                    const Dtype half_surface_size = self->m_tree_->GetNodeSize(bhm_depth) * 0.5f;
                    const Dtype half_bhm_size = half_surface_size + self->m_setting_->bhm_overlap;
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        stream.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        auto [it, inserted] = self->m_key_bhm_dict_.try_emplace(key, nullptr);
                        if (!inserted) {
                            ERL_WARN("Duplicate key {} in key_bhm_dict", std::string(key));
                            return false;
                        }
                        bool has_bhm;
                        stream.read(reinterpret_cast<char *>(&has_bhm), sizeof(bool));
                        if (!has_bhm) { continue; }
                        Position map_center = self->m_tree_->KeyToCoord(key, bhm_depth);
                        Positions hinged_points = self->m_hinged_points_.colwise() + map_center;
                        it->second = std::make_shared<LocalBayesianHilbertMap>(
                            self->m_setting_->local_bhm,
                            hinged_points,
                            Aabb(map_center, half_bhm_size) /*map_boundary*/,
                            typename Key::KeyHash()(key) /*seed*/,
                            Aabb(map_center, half_surface_size) /*track_surface_boundary*/);
                        if (!it->second->Read(stream)) {
                            ERL_WARN("Failed to read bhm for key {}", std::string(key));
                            return false;
                        }
                    }
                    return stream.good();
                },
            },
            {
                "surf_data_manager",
                [](BayesianHilbertSurfaceMapping *self, std::istream &stream) {
                    return self->m_surf_data_manager_.Read(stream) && stream.good();
                },
            },
            {
                "changed_clusters",
                [](BayesianHilbertSurfaceMapping *self, std::istream &stream) {
                    std::size_t n;
                    stream.read(reinterpret_cast<char *>(&n), sizeof(std::size_t));
                    self->m_changed_clusters_.clear();
                    self->m_changed_clusters_.reserve(n);
                    for (std::size_t i = 0; i < n; ++i) {
                        Key key;
                        stream.read(reinterpret_cast<char *>(&key), sizeof(Key));
                        const auto [it, inserted] = self->m_changed_clusters_.insert(key);
                        if (!inserted) {
                            ERL_WARN("Duplicate key {} in changed_clusters", std::string(key));
                            return false;
                        }
                    }
                    return stream.good();
                },
            },
        };
        return ReadTokens(s, this, pairs);
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::GenerateHingedPoints() {
        const Dtype map_size = m_setting_->tree->resolution + 2 * m_setting_->bhm_overlap;
        const Dtype half_map_size = map_size / 2;
        const Eigen::Vector<int, Dim> grid_size =
            Eigen::Vector<int, Dim>::Constant(m_setting_->hinged_grid_size);
        const Eigen::Vector<Dtype, Dim> grid_half_size =
            Eigen::Vector<Dtype, Dim>::Constant(half_map_size);
        common::GridMapInfo<Dtype, Dim> grid_map_info(grid_size, -grid_half_size, grid_half_size);
        m_hinged_points_ = grid_map_info.GenerateMeterCoordinates(true);
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::BuildBhmKdtree() const {
        if (!m_bhm_kdtree_needs_update_ && m_bhm_kdtree_ != nullptr) { return; }
        Positions bhm_positions(Dim, m_key_bhm_positions_.size());
        long i = 0;
        for (const auto &[key, center]: m_key_bhm_positions_) { bhm_positions.col(i++) = center; }
        const_cast<BayesianHilbertSurfaceMapping *>(this)->m_bhm_kdtree_ =
            std::make_shared<Kdtree>(bhm_positions);
        const_cast<BayesianHilbertSurfaceMapping *>(this)->m_bhm_kdtree_needs_update_ = true;
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::PredictThread(
        const Dtype *points_ptr,
        const long start,
        const long end,
        const bool logodd,
        const bool faster,
        const bool compute_gradient,
        const bool gradient_with_sigmoid,
        const bool parallel,
        Dtype *prob_occupied_ptr,
        Dtype *gradient_ptr) const {

        ERL_DEBUG_ASSERT(points_ptr != nullptr, "points_ptr is nullptr.");
        ERL_DEBUG_ASSERT(prob_occupied_ptr != nullptr, "prob_occupied_ptr is nullptr.");
        ERL_DEBUG_ASSERT(!compute_gradient || gradient_ptr != nullptr, "gradient_ptr is nullptr.");
        ERL_DEBUG_ASSERT(start >= 0, "start is negative.");
        ERL_DEBUG_ASSERT(end > start, "end is not greater than start.");

        points_ptr += start * Dim;
        prob_occupied_ptr += start;
        if (compute_gradient) { gradient_ptr += start * Dim; }

        const long num_points = end - start;
        const long knn = m_setting_->test_knn;
        const uint32_t bhm_depth = m_setting_->bhm_depth;
        Dtype half_size = 0.5f * m_tree_->GetNodeSize(bhm_depth) + m_setting_->bhm_test_margin;

        (void) parallel;
        // dynamic scheduling because some points may cause `continue`.
#pragma omp parallel for if (parallel) default(none) schedule(dynamic) \
    shared(num_points,                                                 \
               knn,                                                    \
               half_size,                                              \
               logodd,                                                 \
               faster,                                                 \
               compute_gradient,                                       \
               gradient_with_sigmoid,                                  \
               points_ptr,                                             \
               prob_occupied_ptr,                                      \
               gradient_ptr)
        for (long i = 0; i < num_points; ++i) {

            Eigen::Map<const Position> point(points_ptr + i * Dim, Dim);

            // is there a free node that covers the corresponding bhm completely?
            Key key;  // use the tree to predict the occupancy
            if (!m_tree_->CoordToKeyChecked(point, key)) { continue; }  // outside the map
            const TreeNode *node = m_tree_->Search(key);
            if (node == nullptr) { continue; }  // no observation of this point at all
            if (!m_tree_->IsNodeOccupied(node) &&
                m_tree_->GetNodeSize(node->GetDepth()) * 0.5f > half_size) {
                // use the tree to predict the occupancy
                prob_occupied_ptr[i] = logodd ? node->GetLogOdds() : node->GetOccupancy();
                continue;
            }

            if (knn == 1) {
                long bhm_index = -1;
                Dtype bhm_distance = 0.0f;
                m_bhm_kdtree_->Nearest(point, bhm_index, bhm_distance);      // find the nearest bhm
                bhm_distance = std::sqrt(bhm_distance);                      // distance is squared
                const Key &bhm_key = m_key_bhm_positions_[bhm_index].first;  // the key of the bhm
                const auto &local_bhm = m_key_bhm_dict_.at(bhm_key);         // obtain the bhm

                if (bhm_index < 0 || bhm_distance > half_size || !local_bhm->active) {
                    if (node == nullptr) { continue; }  // unknown
                    // get the occupancy from the tree
                    prob_occupied_ptr[i] = logodd ? node->GetLogOdds() : node->GetOccupancy();
                    continue;
                }

                Gradient grad;
                local_bhm->bhm.Predict(
                    point,
                    logodd,
                    faster,
                    compute_gradient,
                    gradient_with_sigmoid,
                    prob_occupied_ptr[i],
                    grad);
                if (compute_gradient) {
                    Dtype *grad_ptr = gradient_ptr + i * Dim;
                    for (int dim = 0; dim < Dim; ++dim) { grad_ptr[dim] = grad[dim]; }
                }
                continue;
            }

            // m_bhm_kdtree_->Knn(knn, point, bhm_indices, bhm_distances);
            std::vector<nanoflann::ResultItem<long, Dtype>> bhm_indices_dists;
            m_bhm_kdtree_->RadiusSearch(point, half_size, bhm_indices_dists, true /*sorted*/);
            Dtype weight_sum = 0.0f;
            Dtype prob_sum = 0.0f;
            Gradient gradient_sum = Gradient::Zero();
            long cnt = 0;
            for (auto &idx_dist: bhm_indices_dists) {  // iterate over the neighbors
                if (cnt >= knn) { break; }             // only use the first knn active neighbors
                const long &bhm_index = idx_dist.first;
                const Key &bhm_key = m_key_bhm_positions_[bhm_index].first;    // obtain the bhm key
                const auto &local_bhm = m_key_bhm_dict_.at(bhm_key);           // obtain the bhm
                if (local_bhm == nullptr || !local_bhm->active) { continue; }  // not active bhm

                ++cnt;
                Dtype prob;
                Gradient grad;
                local_bhm->bhm.Predict(
                    point,
                    logodd,
                    faster,
                    compute_gradient,
                    gradient_with_sigmoid,
                    prob,
                    grad);
                Dtype weight =
                    1.0f / (local_bhm->bhm.GetMapBoundary().center - point).cwiseAbs().prod();
                weight_sum += weight;
                prob_sum += prob * weight;
                if (compute_gradient) {
                    for (int dim = 0; dim < Dim; ++dim) { gradient_sum[dim] += grad[dim] * weight; }
                }
            }
            if (cnt == 0) {                         // no neighboring bhm
                if (node == nullptr) { continue; }  // unknown
                // get the occupancy from the tree, gradient is not available
                prob_occupied_ptr[i] = logodd ? node->GetLogOdds() : node->GetOccupancy();
            } else {
                prob_occupied_ptr[i] = prob_sum / weight_sum;
                if (compute_gradient) {
                    Dtype *grad_ptr = gradient_ptr + i * Dim;
                    for (int dim = 0; dim < Dim; ++dim) {
                        grad_ptr[dim] = gradient_sum[dim] / weight_sum;
                    }
                }
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::UpdateMapPoints(
        const Eigen::Ref<const Positions> &points) {
        if (m_changed_clusters_.empty()) { return; }

        // sequential:
        // collect hit points from the local Bayesian Hilbert maps
        // collect the map pointer and the local index of the hit points
        // may need to ask the surface data manager to allocate new points

        const Dtype surf_res = m_setting_->local_bhm->surface_resolution;
        m_hit_points_.clear();
        m_hit_points_.reserve(m_changed_clusters_.size());

        {
            ERL_BLOCK_TIMER_MSG("collect hit points from local BHMs");
            for (const Key &key: m_changed_clusters_) {
                auto it = m_key_bhm_dict_.find(key);
                ERL_DEBUG_ASSERT(
                    it != m_key_bhm_dict_.end(),
                    "Key {} not found.",
                    std::string(key));
                if (it == m_key_bhm_dict_.end()) { continue; }  // should not happen

                LocalBayesianHilbertMap &local_bhm = *(it->second);
                if (!local_bhm.active) { continue; }

                /// collect existing hit points
                std::vector<PointInfo> existing_points;
                existing_points.reserve(local_bhm.surface_indices.size());
                for (const auto &[local_idx, surf_idx]: local_bhm.surface_indices) {
                    existing_points.emplace_back(local_idx, surf_idx, false, -1);
                }

                /// add new hit points
                std::vector<PointInfo> new_points;
                const std::size_t max_capacity = local_bhm.grid_shape.prod();
                if (existing_points.size() >= max_capacity) {
                    m_hit_points_.emplace_back(  // no new points
                        key,
                        std::move(new_points),
                        std::move(existing_points));
                    continue;
                }
                new_points.reserve(max_capacity - existing_points.size());
                const auto &map_min = local_bhm.tracked_surface_boundary.min();
                Eigen::Vector<int, Dim> grid_coords;
                int local_index;
                for (const long &hit_index: local_bhm.hit_indices) {
                    auto point = points.col(hit_index);
                    bool contained = true;
                    for (long d = 0; d < Dim; ++d) {
                        grid_coords[d] = common::MeterToGrid<Dtype>(point[d], map_min[d], surf_res);
                        if (grid_coords[d] < 0 || grid_coords[d] >= local_bhm.grid_shape[d]) {
                            contained = false;
                            break;
                        }
                    }
                    if (!contained) { continue; }  // point is out of the local BHM boundary
                    local_index = common::CoordsToIndex<Dim>(local_bhm.strides, grid_coords);
                    auto [it, inserted] = local_bhm.surface_indices.try_emplace(local_index, -1);
                    if (!inserted) { continue; }

                    it->second = m_surf_data_manager_.AddEntry(point, Gradient::Zero(), 0.0f, 0.0f);
                    ERL_DEBUG_ASSERT(
                        static_cast<long>(it->second) != -1l,
                        "Failed to add entry to the surface data manager.");
                    new_points.emplace_back(local_index, it->second, false, -1);
                    if (new_points.size() >= new_points.capacity()) { break; }  // no more needed
                }
                m_hit_points_.emplace_back(key, std::move(new_points), std::move(existing_points));
            }
        }

        // parallel:
        // for each point to update, compute the logodd and the gradient
        // adjust the point to make |logodd| close to 0 as much as possible
        // if |logodd| is too large, the point should be removed.
        // after the move, the local index may change

        {
            ERL_BLOCK_TIMER_MSG("compute logodd and gradient for points");
            // need dynamic scheduling here because the workload is not evenly distributed.
            // some local BHMs may have more points than others.
            // some points need to be removed so that the computation stops earlier.
#pragma omp parallel for default(none) shared(surf_res) schedule(dynamic)
            for (auto &[key, new_points, existing_points]: m_hit_points_) {
                // abs(logodd) may be larger than the threshold, but for new points, we don't remove
                // them immediately, we will check when we try to update the point again.
                LocalBayesianHilbertMap &local_bhm = *m_key_bhm_dict_[key];
                for (PointInfo &new_point: new_points) {
                    SurfData &surf_data = m_surf_data_manager_[new_point.surf_idx];
                    InitMapPoint(local_bhm.bhm, surf_data, new_point.to_remove);
                }
                for (PointInfo &existing_point: existing_points) {
                    SurfData &surf_data = m_surf_data_manager_[existing_point.surf_idx];
                    UpdateMapPoint(local_bhm.bhm, surf_data, existing_point.to_remove);
                    if (!local_bhm.tracked_surface_boundary.contains(surf_data.position)) {
                        existing_point.to_remove = true;
                    }
                    if (existing_point.to_remove) { continue; }
                    // if the point is not removed, we need to update the local index
                    Eigen::Vector<int, Dim> grid_coords;
                    const auto &map_min = local_bhm.tracked_surface_boundary.min();
                    for (long dim = 0; dim < Dim; ++dim) {
                        grid_coords[dim] = common::MeterToGrid<Dtype>(
                            surf_data.position[dim],
                            map_min[dim],
                            surf_res);
                    }
                    existing_point.new_local_idx =
                        common::CoordsToIndex<Dim>(local_bhm.strides, grid_coords);
                    if (existing_point.new_local_idx == existing_point.local_idx) {
                        existing_point.new_local_idx = -1;  // no change in local index
                    }
                }
            }
        }

#ifndef NDEBUG
        Dtype init_logodd_abs = 0.0f;
        Dtype init_logodd_abs_min = std::numeric_limits<Dtype>::max();
        Dtype init_logodd_abs_max = std::numeric_limits<Dtype>::lowest();
        int cnt_new_points = 0;
        Dtype adjust_logodd_abs = 0.0f;
        Dtype adjust_logodd_abs_min = std::numeric_limits<Dtype>::max();
        Dtype adjust_logodd_abs_max = std::numeric_limits<Dtype>::lowest();
        int cnt_existing_points = 0;
        for (const auto &[key, new_points, existing_points]: m_hit_points_) {
            for (const PointInfo &new_point: new_points) {
                const SurfData &surf_data = m_surf_data_manager_[new_point.surf_idx];
                init_logodd_abs += surf_data.var_position;
                init_logodd_abs_min = std::min(init_logodd_abs_min, surf_data.var_position);
                init_logodd_abs_max = std::max(init_logodd_abs_max, surf_data.var_position);
                ++cnt_new_points;
            }
            for (const PointInfo &existing_point: existing_points) {
                const SurfData &surf_data = m_surf_data_manager_[existing_point.surf_idx];
                adjust_logodd_abs += surf_data.var_position;
                adjust_logodd_abs_min = std::min(adjust_logodd_abs_min, surf_data.var_position);
                adjust_logodd_abs_max = std::max(adjust_logodd_abs_max, surf_data.var_position);
                ++cnt_existing_points;
            }
        }
        init_logodd_abs /= static_cast<Dtype>(cnt_new_points) * m_setting_->var_scale;
        init_logodd_abs_min /= m_setting_->var_scale;
        init_logodd_abs_max /= m_setting_->var_scale;
        ERL_INFO(
            "Initial logodd abs: {} (min: {}, max: {})",
            init_logodd_abs,
            init_logodd_abs_min,
            init_logodd_abs_max);
        adjust_logodd_abs /= static_cast<Dtype>(cnt_existing_points) * m_setting_->var_scale;
        adjust_logodd_abs_min /= m_setting_->var_scale;
        adjust_logodd_abs_max /= m_setting_->var_scale;
        ERL_INFO(
            "Adjusted logodd abs: {} (min: {}, max: {})",
            adjust_logodd_abs,
            adjust_logodd_abs_min,
            adjust_logodd_abs_max);
#endif

        // sequential:
        // update the local Bayesian Hilbert maps with the new points
        for (auto &[key, new_points, existing_points]: m_hit_points_) {
            LocalBayesianHilbertMap &local_bhm = *m_key_bhm_dict_[key];
            for (PointInfo &new_point: new_points) {
                if (!new_point.to_remove) { continue; }
                local_bhm.surface_indices.erase(new_point.local_idx);
                m_surf_data_manager_.RemoveEntry(new_point.surf_idx);
            }
            for (PointInfo &existing_point: existing_points) {
                if (existing_point.to_remove) {
                    m_surf_data_manager_.RemoveEntry(existing_point.surf_idx);
                    local_bhm.surface_indices.erase(existing_point.local_idx);
                    continue;
                }
                if (existing_point.new_local_idx == -1) { continue; }  // no change in local index

                auto new_surf_it = local_bhm.surface_indices.find(existing_point.new_local_idx);
                if (new_surf_it == local_bhm.surface_indices.end()) {
                    local_bhm.surface_indices.emplace(
                        existing_point.new_local_idx,
                        existing_point.surf_idx);
                } else {
                    m_surf_data_manager_.RemoveEntry(existing_point.surf_idx);
                }
                local_bhm.surface_indices.erase(existing_point.local_idx);
            }
        }
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::InitMapPoint(
        BayesianHilbertMap &bhm,
        SurfData &surf_data,
        bool &to_remove) const {
        Dtype logodd;
        bhm.Predict(
            surf_data.position,
            true /*logodd*/,
            m_setting_->faster_prediction,
            true /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            logodd, /*logodd output*/
            surf_data.normal /*gradient output*/);
        Dtype norm = surf_data.normal.norm();
        if (norm < 1e-6f) {
            to_remove = true;  // if the normal is too small, remove the point
            return;
        }
        surf_data.normal = surf_data.normal / -norm;  // normal = -gradient
        surf_data.var_position =
            std::min(m_setting_->var_scale * std::abs(logodd), m_setting_->var_max);
        surf_data.var_normal = surf_data.var_position;  // use the same variance for normal
    }

    template<typename Dtype, int Dim>
    void
    BayesianHilbertSurfaceMapping<Dtype, Dim>::UpdateMapPoint(
        BayesianHilbertMap &bhm,
        SurfData &surf_data,
        bool &to_remove) const {
        const Dtype max_logodd_abs = m_setting_->surface_max_abs_logodd;
        const int max_num_adjusts = m_setting_->max_adjust_tries;
        int num_adjusts = 0;
        Dtype logodd;
        Gradient &gradient = surf_data.normal;  // reuse the normal as the gradient
        bhm.Predict(
            surf_data.position,
            true /*logodd*/,
            m_setting_->faster_prediction,
            true /*compute_gradient*/,
            false /*gradient_with_sigmoid*/,
            logodd, /*logodd output*/
            gradient /*gradient output*/);
        Dtype norm = gradient.norm();
        if (norm < 1e-6f) {
            to_remove = true;  // if the gradient is too small, remove the point
            return;
        }
        Dtype logodd_abs = std::abs(logodd);
#ifndef NDEBUG
        Dtype logodd_init = logodd;
        Dtype logodd_abs_init = logodd_abs;
        Dtype norm_init = norm;
#endif

        Dtype delta = m_setting_->surface_step_size;
        while (num_adjusts < max_num_adjusts && logodd_abs > max_logodd_abs) {
            // logodd > 0, prob(occupied) > 0.5, move the point along -gradient
            // logodd < 0, prob(occupied) < 0.5, move the point along gradient
            Dtype step = -logodd * delta / (norm * norm);
            surf_data.position += step * gradient;
            Dtype logodd_new;
            bhm.Predict(
                surf_data.position,
                true /*logodd*/,
                m_setting_->faster_prediction,
                true /*compute_gradient*/,
                false /*gradient_with_sigmoid*/,
                logodd_new, /*logodd output*/
                gradient /*gradient output*/);
            norm = gradient.norm();
            if (norm < 1e-6f) {
                to_remove = true;  // if the gradient is too small, remove the point
                break;
            }
            logodd_abs = std::abs(logodd_new);
            if (logodd_abs <= max_logodd_abs) { break; }
            if (logodd_new * logodd < 0) {  // logodd changed sign, reduce the step size
                delta *= 0.5f;
            } else {
                delta *= 1.1f;  // increase the step size
            }
            logodd = logodd_new;
            ++num_adjusts;
        }
        if (logodd_abs >= m_setting_->surface_bad_abs_logodd) {
            to_remove = true;
            return;
        }
        ERL_DEBUG_WARN_COND(
            logodd_abs > logodd_abs_init,
            "logodd_abs {} is larger than initial {} after {} adjustments. logodd: {} (initial: "
            "{}), norm: {} (initial: {}).",
            logodd_abs,
            logodd_abs_init,
            num_adjusts,
            logodd,
            logodd_init,
            norm,
            norm_init);
        surf_data.normal = gradient / -norm;  // normal = -gradient
        surf_data.var_position = std::min(m_setting_->var_scale * logodd_abs, m_setting_->var_max);
        surf_data.var_normal = surf_data.var_position;  // use the same variance for normal
    }

    template class RaySelector2D<float>;
    template class RaySelector2D<double>;
    template class RaySelector3D<float>;
    template class RaySelector3D<double>;
    template class LocalBayesianHilbertMapSetting<float>;
    template class LocalBayesianHilbertMapSetting<double>;
    template class BayesianHilbertSurfaceMapping<float, 2>;
    template class BayesianHilbertSurfaceMapping<float, 3>;
    template class BayesianHilbertSurfaceMapping<double, 2>;
    template class BayesianHilbertSurfaceMapping<double, 3>;
}  // namespace erl::gp_sdf
