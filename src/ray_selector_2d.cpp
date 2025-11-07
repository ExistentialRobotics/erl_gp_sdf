#include "erl_gp_sdf/ray_selector_2d.hpp"

#include "erl_common/angle_utils.hpp"

namespace erl::gp_sdf {

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
            p = m_setting_->transform.template leftCols<2>() * p + m_setting_->transform.col(2);
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
        point = m_setting_->transform.template leftCols<2>() * point + m_setting_->transform.col(2);
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

    template class RaySelector2D<float>;
    template class RaySelector2D<double>;
}  // namespace erl::gp_sdf
