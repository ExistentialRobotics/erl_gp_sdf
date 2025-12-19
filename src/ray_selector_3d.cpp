#include "erl_gp_sdf/ray_selector_3d.hpp"

#include "erl_common/angle_utils.hpp"

namespace erl::gp_sdf {

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

        auto *ptr = m_ray_indices_.data();
        for (long i = 0; i < m_ray_indices_.size(); ++i) { ptr[i].clear(); }

        Eigen::Matrix2Xl coords(2, ray_end_points.cols());
#pragma omp parallel for default(none) \
    shared(ray_end_points, coords, sensor_rotation, sensor_origin) schedule(static)
        for (long i = 0; i < ray_end_points.cols(); ++i) {
            const Dtype &azimuth_min = m_setting_->azimuth_min;
            const Dtype &azimuth_max = m_setting_->azimuth_max;
            const Dtype &elevation_min = m_setting_->elevation_min;
            const Dtype &elevation_max = m_setting_->elevation_max;

            Vector3 p = sensor_rotation.transpose() * (ray_end_points.col(i) - sensor_origin);
            p = m_setting_->transform.template leftCols<3>() * p + m_setting_->transform.col(3);
            p.normalize();
            Dtype azimuth, elevation;
            DirectionToAzimuthElevation<Dtype>(p, azimuth, elevation);
            azimuth = std::max(azimuth_min, std::min(azimuth, azimuth_max));
            elevation = std::max(elevation_min, std::min(elevation, elevation_max));
            auto coord = coords.col(i);
            coord[0] = MeterToGrid<Dtype, long>(azimuth, azimuth_min, m_azimuth_res_);
            coord[1] = MeterToGrid<Dtype, long>(elevation, elevation_min, m_elevation_res_);
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
        point = m_setting_->transform.template leftCols<3>() * point + m_setting_->transform.col(3);
        const Dtype dist = point.norm();
        point /= dist;
        Dtype azimuth, elevation;
        DirectionToAzimuthElevation<Dtype>(point, azimuth, elevation);
        const Dtype theta = std::atan2(radius, dist);

        const Dtype &azimuth_min = m_setting_->azimuth_min;
        const Dtype &azimuth_max = m_setting_->azimuth_max;
        const Dtype &elevation_min = m_setting_->elevation_min;
        const Dtype &elevation_max = m_setting_->elevation_max;
        constexpr auto kPi = static_cast<Dtype>(M_PI);
        constexpr auto kPi2 = static_cast<Dtype>(M_PI_2);

        Dtype ele_max = elevation + theta;
        if (ele_max > kPi2) {
            elevation = std::min(elevation - theta, kPi - ele_max);
            elevation = std::max(elevation_min, std::min(elevation, elevation_max));
            auto ele_idx = MeterToGrid<Dtype, long>(elevation, elevation_min, m_elevation_res_);
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
            auto ele_idx = MeterToGrid<Dtype, long>(elevation, elevation_min, m_elevation_res_);
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
        auto min_eidx = MeterToGrid<Dtype, long>(ele_min, elevation_min, m_elevation_res_);
        auto max_eidx = MeterToGrid<Dtype, long>(ele_max, elevation_min, m_elevation_res_);

        Dtype azi_max = WrapAnglePi(azimuth + theta);
        Dtype azi_min = WrapAnglePi(azimuth - theta);
        if (azi_min < azi_max) {
            azi_min = std::max(azimuth_min, std::min(azi_min, azimuth_max));
            azi_max = std::max(azimuth_min, std::min(azi_max, azimuth_max));
            auto min_aidx = MeterToGrid<Dtype, long>(azi_min, azimuth_min, m_azimuth_res_);
            auto max_aidx = MeterToGrid<Dtype, long>(azi_max, azimuth_min, m_azimuth_res_);

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
        auto min_aidx = MeterToGrid<Dtype, long>(azi_min, azimuth_min, m_azimuth_res_);
        auto max_aidx = MeterToGrid<Dtype, long>(azi_max, azimuth_min, m_azimuth_res_);
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

    template class RaySelector3D<float>;
    template class RaySelector3D<double>;
}  // namespace erl::gp_sdf
