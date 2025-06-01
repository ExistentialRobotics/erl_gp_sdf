#pragma once

#include "surface_data_manager.hpp"

#include "erl_common/exception.hpp"
#include "erl_common/factory_pattern.hpp"
#include "erl_common/yaml.hpp"

namespace erl::sdf_mapping {

    class AbstractSurfaceMapping {
    protected:
        std::mutex m_mutex_;
        int m_map_dim_ = 3;
        bool m_is_double_ = false;

    public:
        using Factory = common::FactoryPattern<
            AbstractSurfaceMapping,
            false,
            false,
            const std::shared_ptr<common::YamlableBase> &>;
        virtual ~AbstractSurfaceMapping() = default;

        /**
         * Lock the mutex of the mapping.
         * @return the lock guard of the mutex.
         */
        [[nodiscard]] std::lock_guard<std::mutex>
        GetLockGuard() {
            return std::lock_guard<std::mutex>(m_mutex_);
        }

        [[nodiscard]] int
        GetMapDim() const {
            return m_map_dim_;
        }

        [[nodiscard]] bool
        IsDoublePrecision() const {
            return m_is_double_;
        }

        /**
         * Update the surface_indices mapping with the sensor observation.
         * @param rotation The rotation of the sensor. For 2D, it is a 2x2 matrix. For 3D, it is a
         * 3x3 matrix.
         * @param translation The translation of the sensor. For 2D, it is a 2x1 vector. For 3D, it
         * is a 3x1 vector.
         * @param scan The sensor observation, which can be a point cloud or a range array.
         * @param are_points If true, the scan is a point cloud. Otherwise, a range array.
         * @param are_local If true, the points are in the local frame.
         * @return true if the update is successful.
         */
        [[nodiscard]] virtual bool
        Update(
            const Eigen::Ref<const Eigen::MatrixXd> &rotation,
            const Eigen::Ref<const Eigen::VectorXd> &translation,
            const Eigen::Ref<const Eigen::MatrixXd> &scan,
            bool are_points,
            bool are_local) = 0;

        /**
         * Get the surface_indices data. When this method is called, the mutex should be locked temporarily
         * to copy the data, which blocks the Update method. If the mapping implementation is for
         * 2D, the z-axis should be set to 0.
         * @return A vector of surface points with normals, variances, etc.
         */
        [[nodiscard]] virtual std::vector<SurfaceData<double, 3>>
        GetSurfaceData() const = 0;

        // Comparison
        [[nodiscard]] virtual bool
        operator==(const AbstractSurfaceMapping & /*other*/) const {
            throw NotImplemented(__PRETTY_FUNCTION__);
        }

        [[nodiscard]] bool
        operator!=(const AbstractSurfaceMapping &other) const {
            return !(*this == other);
        }

        // IO

        [[nodiscard]] virtual bool
        Write(std::ostream &s) const = 0;

        [[nodiscard]] virtual bool
        Read(std::istream &s) = 0;

        // Factory pattern
        template<typename Derived = AbstractSurfaceMapping>
        static std::shared_ptr<Derived>
        Create(
            const std::string &mapping_type,
            const std::shared_ptr<common::YamlableBase> &setting) {
            return Factory::GetInstance().Create(mapping_type, setting);
        }

        template<typename Derived>
        static std::enable_if_t<std::is_base_of_v<AbstractSurfaceMapping, Derived>, bool>
        Register(std::string mapping_type = "") {
            return Factory::GetInstance().Register<Derived>(
                mapping_type,
                [](const std::shared_ptr<common::YamlableBase> &setting)
                    -> std::shared_ptr<AbstractSurfaceMapping> {
                    auto mapping_setting =
                        std::dynamic_pointer_cast<typename Derived::Setting>(setting);
                    if (setting == nullptr) {
                        mapping_setting = std::make_shared<typename Derived::Setting>();
                    }
                    if (mapping_setting == nullptr) {
                        ERL_WARN(
                            "Failed to cast setting to {}",
                            type_name<typename Derived::Setting>());
                        return nullptr;
                    }
                    return std::make_shared<Derived>(mapping_setting);
                });
        }
    };
}  // namespace erl::sdf_mapping
