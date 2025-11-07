#pragma once

#include "erl_common/yaml.hpp"

namespace erl::gp_sdf {
    /**
     * Select 2D rays for Bayesian Hilbert Map.
     */
    template<typename Dtype>
    class RaySelector2D {

    public:
        struct Setting : common::Yamlable<Setting> {
            // minimum ray angle in radians in the local frame
            Dtype angle_min = -M_PI / 4.0;
            // maximum ray angle in radians in the local frame
            Dtype angle_max = M_PI / 4.0;
            // number of angles
            long num_angles = 91;
            // transform
            Eigen::Matrix<Dtype, 2, 3> transform = Eigen::Matrix<Dtype, 2, 3>::Identity();

            ERL_REFLECT_SCHEMA(
                Setting,
                ERL_REFLECT_MEMBER(Setting, angle_min),
                ERL_REFLECT_MEMBER(Setting, angle_max),
                ERL_REFLECT_MEMBER(Setting, num_angles),
                ERL_REFLECT_MEMBER(Setting, transform));
        };

        using Vector2 = Eigen::Vector2<Dtype>;
        using Matrix2 = Eigen::Matrix2<Dtype>;
        using Matrix2X = Eigen::Matrix2X<Dtype>;

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        Dtype m_angle_resolution_ = 0.0f;
        std::vector<std::vector<long>> m_ray_indices_;  // ray indices for each angle

    public:
        explicit RaySelector2D(std::shared_ptr<Setting> setting);

        void
        UpdateRays(
            const Vector2 &sensor_origin,
            const Matrix2 &sensor_rotation,
            const Eigen::Ref<const Matrix2X> &ray_end_points);

        void
        SelectRays(
            const Vector2 &sensor_origin,
            const Matrix2 &sensor_rotation,
            Vector2 point,
            Dtype radius,
            std::vector<long> &ray_indices) const;
    };

    extern template class RaySelector2D<float>;
    extern template class RaySelector2D<double>;

    using RaySelector2Df = RaySelector2D<float>;
    using RaySelector2Dd = RaySelector2D<double>;
}  // namespace erl::gp_sdf
