#pragma once

#include "surface_data_manager.hpp"

#include "erl_common/exception.hpp"
#include "erl_common/factory_pattern.hpp"
#include "erl_common/yaml.hpp"
#include "erl_geometry/aabb.hpp"
#include "erl_geometry/octree_key.hpp"
#include "erl_geometry/quadtree_key.hpp"

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    class AbstractSurfaceMapping {
    public:
        using Factory = common::FactoryPattern<
            AbstractSurfaceMapping,
            false,
            false,
            const std::shared_ptr<common::YamlableBase> &>;
        using Key = std::conditional_t<Dim == 2, geometry::QuadtreeKey, geometry::OctreeKey>;
        using KeySet = std::conditional_t<  //
            Dim == 2,
            geometry::QuadtreeKeySet,
            geometry::OctreeKeySet>;
        using KeyVector = std::vector<Key>;
        using SurfDataManager = SurfaceDataManager<Dtype, Dim>;
        using SurfData = typename SurfDataManager::Data;
        using Aabb = geometry::Aabb<Dtype, Dim>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using Rotation = Eigen::Matrix<Dtype, Dim, Dim>;
        using Translation = Eigen::Vector<Dtype, Dim>;
        using Ranges = MatrixX;
        using Position = Eigen::Vector<Dtype, Dim>;
        using Positions = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;
        using Face = Eigen::Vector<int, Dim>;

    protected:
        std::mutex m_mutex_;
        // surface data manager to manage the surface data buffer
        SurfDataManager m_surf_data_manager_;

    public:
        virtual ~AbstractSurfaceMapping() = default;

        /**
         * Lock the mutex of the mapping.
         * @return the lock guard of the mutex.
         */
        [[nodiscard]] std::lock_guard<std::mutex>
        GetLockGuard();

        [[nodiscard]] const SurfDataManager &
        GetSurfaceDataManager() const;

        /**
         * Update the surface mapping with the sensor observation.
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
            const Eigen::Ref<const Rotation> &rotation,
            const Eigen::Ref<const Translation> &translation,
            const Eigen::Ref<const Ranges> &scan,
            bool are_points,
            bool are_local) = 0;

        // implement the methods required by GpSdfMapping

        /**
         * @return the scaling factor of the map.
         */
        [[nodiscard]] virtual Dtype
        GetScaling() const = 0;

        /**
         * Get the size of the cluster.
         * @return the size of the cluster.
         */
        [[nodiscard]] virtual Dtype
        GetClusterSize() const = 0;

        /**
         * Get the center of the cluster.
         * @param key the key of the cluster.
         * @return the center of the cluster.
         */
        [[nodiscard]] virtual Position
        GetClusterCenter(const Key &key) const = 0;

        /**
         * Get the keys of clusters that have been changed.
         * @return set of keys of clusters.
         */
        [[nodiscard]] virtual const KeySet &
        GetChangedClusters() const = 0;

        /**
         * Get clusters.
         * @return a collection of all cluster keys.
         */
        [[nodiscard]] virtual KeySet
        GetAllClusters() const = 0;

        [[nodiscard]] virtual Key
        GetClusterKey(const Eigen::Ref<const Position> &pos) const = 0;

        /**
         * Iterate over the clusters in the given axis-aligned bounding box.
         * @param aabb the axis-aligned bounding box to collect clusters.
         * @param callback the callback function to process the key of the cluster.
         */
        virtual void
        IterateClustersInAabb(const Aabb &aabb, std::function<void(const Key &)> callback)
            const = 0;

        /**
         * Get the surface data buffer.
         * @return vector of surface data.
         */
        [[nodiscard]] virtual const std::vector<SurfData> &
        GetSurfaceDataBuffer() const = 0;

        /**
         * Collect surface data in the given axis-aligned bounding box.
         * @param aabb the axis-aligned bounding box to collect surface data.
         * @param surface_data_indices vector of (distance to point, surface point index).
         */
        virtual void
        CollectSurfaceDataInAabb(
            const Aabb &aabb,
            std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const = 0;

        virtual void
        GetMesh(std::vector<Position> &vertices, std::vector<Face> &faces) const;

        /**
         * Get the boundary of the map.
         * @return the boundary of the map as an axis-aligned bounding box.
         */
        [[nodiscard]] virtual Aabb
        GetMapBoundary() const = 0;

        /**
         * Check if the given positions are in free space.
         * @param positions the positions to check.
         * @param in_free_space the vector to store the result. 1.0 if the position is in free
         * space, -1.0 otherwise.
         * @return true if this method is successful. false if the algorithm fails / is not
         * implemented.
         */
        [[nodiscard]] virtual bool
        IsInFreeSpace(const Positions &positions, VectorX &in_free_space) const = 0;

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
            return Factory::GetInstance().template Register<Derived>(
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

    using AbstractSurfaceMapping2Df = AbstractSurfaceMapping<float, 2>;
    using AbstractSurfaceMapping2Dd = AbstractSurfaceMapping<double, 2>;
    using AbstractSurfaceMapping3Df = AbstractSurfaceMapping<float, 3>;
    using AbstractSurfaceMapping3Dd = AbstractSurfaceMapping<double, 3>;

    extern template class AbstractSurfaceMapping<float, 2>;
    extern template class AbstractSurfaceMapping<double, 2>;
    extern template class AbstractSurfaceMapping<float, 3>;
    extern template class AbstractSurfaceMapping<double, 3>;
}  // namespace erl::gp_sdf
