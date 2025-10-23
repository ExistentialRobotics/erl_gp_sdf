#pragma once

#include "erl_common/yaml.hpp"

namespace erl::gp_sdf {
    template<typename Dtype>
    class RaySelector3D {
    public:
        struct Setting : common::Yamlable<Setting> {
            // minimum azimuth angle in radians
            Dtype azimuth_min = -M_PI;
            // maximum azimuth angle in radians
            Dtype azimuth_max = M_PI;
            // minimum elevation angle in radians
            Dtype elevation_min = -M_PI / 2.0;
            // maximum elevation angle in radians
            Dtype elevation_max = M_PI / 2.0;
            // number of azimuth angles
            Dtype num_azimuth_angles = 181;
            // number of elevation angles
            Dtype num_elevation_angles = 91;
            // transform
            Eigen::Matrix<Dtype, 3, 4> transform = Eigen::Matrix<Dtype, 3, 4>::Identity();

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        using Vector3 = Eigen::Vector3<Dtype>;
        using Matrix3 = Eigen::Matrix3<Dtype>;
        using Matrix3X = Eigen::Matrix3X<Dtype>;

    private:
        std::shared_ptr<Setting> m_setting_ = nullptr;
        Dtype m_azimuth_res_ = 0.0f;
        Dtype m_elevation_res_ = 0.0f;
        Eigen::MatrixX<std::vector<long>> m_ray_indices_;  // ray indices for each angle

    public:
        explicit RaySelector3D(std::shared_ptr<Setting> setting);

        void
        UpdateRays(
            const Vector3 &sensor_origin,
            const Matrix3 &sensor_rotation,
            const Eigen::Ref<const Matrix3X> &ray_end_points);

        /**
         *
         * @param sensor_origin position of the sensor in the world frame.
         * @param sensor_rotation rotation of the sensor in the world frame.
         * @param point center of the selection sphere in the world frame.
         * @param radius radius of the selection sphere.
         * @param ray_indices output vector of selected ray indices.
         */
        void
        SelectRays(
            const Vector3 &sensor_origin,
            const Matrix3 &sensor_rotation,
            Vector3 point,
            Dtype radius,
            std::vector<long> &ray_indices) const;
    };

    extern template class RaySelector3D<float>;
    extern template class RaySelector3D<double>;

    using RaySelector3Df = RaySelector3D<float>;
    using RaySelector3Dd = RaySelector3D<double>;
}  // namespace erl::gp_sdf

template<>
struct YAML::convert<erl::gp_sdf::RaySelector3Df::Setting>
    : erl::gp_sdf::RaySelector3Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::gp_sdf::RaySelector3Dd::Setting>
    : erl::gp_sdf::RaySelector3Dd::Setting::YamlConvertImpl {};
