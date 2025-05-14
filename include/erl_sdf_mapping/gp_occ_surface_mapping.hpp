#pragma once

#include "abstract_surface_mapping.hpp"
#include "surface_data_manager.hpp"

#include "erl_gaussian_process/lidar_gp_2d.hpp"
#include "erl_gaussian_process/range_sensor_gp_3d.hpp"
#include "erl_geometry/occupancy_octree.hpp"
#include "erl_geometry/occupancy_quadtree.hpp"

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    class GpOccSurfaceMapping : public AbstractSurfaceMapping {
        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

    public:
        // type definitions required by GpSdfMapping
        using Key = std::conditional_t<Dim == 2, geometry::QuadtreeKey, geometry::OctreeKey>;
        using KeySet = std::conditional_t<  //
            Dim == 2,
            geometry::QuadtreeKeySet,
            geometry::OctreeKeySet>;
        using KeyVector = std::conditional_t<  //
            Dim == 2,
            geometry::QuadtreeKeyVector,
            geometry::OctreeKeyVector>;
        using Tree = std::conditional_t<
            Dim == 2,
            geometry::OccupancyQuadtree<Dtype>,
            geometry::OccupancyOctree<Dtype>>;
        using TreeNode = std::conditional_t<  //
            Dim == 2,
            geometry::OccupancyQuadtreeNode,
            geometry::OccupancyOctreeNode>;
        using SurfDataManager = SurfaceDataManager<Dtype, Dim>;
        using SurfData = typename SurfDataManager::Data;

        // other types
        using SensorGp = std::conditional_t<
            Dim == 2,
            gaussian_process::LidarGaussianProcess2D<Dtype>,
            gaussian_process::RangeSensorGaussianProcess3D<Dtype>>;
        using SensorGpSetting = typename SensorGp::Setting;
        using TreeSetting = typename Tree::Setting;
        using Aabb = geometry::Aabb<Dtype, Dim>;
        using SurfIndices0Type = absl::flat_hash_map<Key, std::size_t>;
        using SurfIndices1Type = absl::flat_hash_map<Key, absl::flat_hash_map<int, std::size_t>>;

        // eigen types
        using Scalar = Eigen::Matrix<Dtype, 1, 1>;
        using MatrixX = Eigen::MatrixX<Dtype>;
        using VectorX = Eigen::VectorX<Dtype>;
        using Rotation = Eigen::Matrix<Dtype, Dim, Dim>;
        using Translation = Eigen::Vector<Dtype, Dim>;
        using Ranges = MatrixX;
        using Position = Eigen::Vector<Dtype, Dim>;
        using Gradient = Position;
        using Positions = Eigen::Matrix<Dtype, Dim, Eigen::Dynamic>;

        struct Setting : public common::Yamlable<Setting> {
            struct ComputeVariance {
                // position variance to set when the estimated gradient is almost zero.
                Dtype zero_gradient_position_var = 1.;
                // gradient variance to set when the estimated gradient is almost zero.
                Dtype zero_gradient_gradient_var = 1.;
                Dtype position_var_alpha = 0.01f;  // scaling number of the position variance.
                Dtype min_distance_var = 1.0f;     // allowed minimum distance variance.
                Dtype max_distance_var = 100.0f;   // allowed maximum distance variance.
                Dtype min_gradient_var = 0.01f;    // allowed minimum gradient variance.
                Dtype max_gradient_var = 1.0f;     // allowed maximum gradient variance.
            };

            struct UpdateMapPoints {
                int max_adjust_tries = 10;
                // points of OCC smaller than this value are considered unobservable.
                // i.e., inside the object.
                Dtype min_observable_occ = -0.1f;
                Dtype min_position_var = 0.001f;  // minimum position variance.
                Dtype min_gradient_var = 0.001f;  // minimum gradient variance.

                // maximum absolute value of surface points' OCC, which should be zero ideally.
                Dtype max_surface_abs_occ = 0.02f;
                // maximum valid gradient variance, above this threshold, it won't be used for the
                // Bayes Update.
                Dtype max_valid_gradient_var = 0.5f;
                // if the position variance by Bayes Update is above this threshold, it will be
                // discarded.
                Dtype max_bayes_position_var = 1.0f;
                // if the gradient variance by Bayes Update is above this threshold, it will be
                // discarded.
                Dtype max_bayes_gradient_var = 0.6f;
            };

            ComputeVariance compute_variance;
            UpdateMapPoints update_map_points;
            std::shared_ptr<SensorGpSetting> sensor_gp = std::make_shared<SensorGpSetting>();
            std::shared_ptr<TreeSetting> tree = std::make_shared<TreeSetting>();
            // resolution to track the surface points; when <= 0, each leaf node contains only one
            // surface point.
            Dtype surface_resolution = 0.01f;
            Dtype scaling = 1.0f;         // internal scaling factor.
            Dtype perturb_delta = 0.01f;  // perturbation delta for gradient estimation.

            // a gradient with norm below this threshold is considered zero.
            Dtype zero_gradient_threshold = 1.e-15f;
            bool update_occupancy = true;  // whether to update the occupancy of the occupancy tree.
            uint32_t cluster_depth = 14;

            struct YamlConvertImpl {
                static YAML::Node
                encode(const Setting &setting);

                static bool
                decode(const YAML::Node &node, Setting &setting);
            };
        };

        class SurfaceDataIterator {
            GpOccSurfaceMapping *m_mapping_;
            bool m_use0_ = true;
            typename SurfIndices0Type::iterator m_it0_;
            typename SurfIndices1Type::iterator m_it1_;
            absl::flat_hash_map<int, std::size_t>::iterator m_it2_;

        public:
            explicit SurfaceDataIterator(GpOccSurfaceMapping *mapping);

            SurfaceDataIterator(const SurfaceDataIterator &other) = default;
            SurfaceDataIterator &
            operator=(const SurfaceDataIterator &other) = default;
            SurfaceDataIterator(SurfaceDataIterator &&other) = default;
            SurfaceDataIterator &
            operator=(SurfaceDataIterator &&other) = default;

            [[nodiscard]] bool
            operator==(const SurfaceDataIterator &other) const;

            [[nodiscard]] bool
            operator!=(const SurfaceDataIterator &other) const;

            SurfData &
            operator*();

            SurfData *
            operator->();

            SurfaceDataIterator &
            operator++();

            SurfaceDataIterator
            operator++(int);
        };

    private:
        std::shared_ptr<Setting> m_setting_ = std::make_shared<Setting>();
        std::shared_ptr<SensorGp> m_sensor_gp_ = nullptr;
        std::shared_ptr<Tree> m_tree_ = nullptr;
        // strides for the grid indices
        Eigen::Vector<int, Dim> m_strides_ = Eigen::Vector<int, Dim>::Zero();
        // key -> surface data index (used when the surface resolution is <= 0)
        SurfIndices0Type m_surf_indices0_;
        // key -> [grid_min, (grid index -> surface data index)]
        SurfIndices1Type m_surf_indices1_;
        // surface data manager to manage the surface data buffer
        SurfDataManager m_surf_data_manager_;
        Eigen::Matrix<Dtype, Dim, 2 * Dim> m_pos_perturb_ = {};
        Dtype m_surface_resolution_inv_ = 0.0f;  // inverse of the tree resolution
        KeySet m_changed_keys_ = {};
        std::mutex m_mutex_;

    public:
        explicit GpOccSurfaceMapping(std::shared_ptr<Setting> setting);

        [[nodiscard]] std::shared_ptr<const Setting>
        GetSetting() const;

        [[nodiscard]] std::shared_ptr<const SensorGp>
        GetSensorGp() const;

        [[nodiscard]] std::shared_ptr<const Tree>
        GetTree() const;

        [[nodiscard]] const SurfDataManager &
        GetSurfaceDataManager() const;

        bool
        Update(
            const Eigen::Ref<const Rotation> &rotation,
            const Eigen::Ref<const Translation> &translation,
            const Eigen::Ref<const Ranges> &ranges);

        SurfaceDataIterator
        BeginSurfaceData();

        SurfaceDataIterator
        EndSurfaceData();

        // implement the methods required by GpSdfMapping

        /**
         * Lock the mutex of the mapping.
         * @return the lock guard of the mutex.
         */
        [[nodiscard]] std::lock_guard<std::mutex>
        GetLockGuard();

        /**
         * @return the scaling factor of the map.
         */
        [[nodiscard]] Dtype
        GetScaling() const;

        /**
         * Get the size of the cluster.
         * @return the size of the cluster.
         */
        [[nodiscard]] Dtype
        GetClusterSize() const;

        /**
         * Get the center of the cluster.
         * @param key the key of the cluster.
         * @return the center of the cluster.
         */
        [[nodiscard]] Position
        GetClusterCenter(const Key &key) const;

        /**
         * Get the keys of clusters that have been changed.
         * @return set of keys of clusters.
         */
        [[nodiscard]] const KeySet &
        GetChangedClusters() const;

        /**
         * Iterate over the clusters in the given axis-aligned bounding box.
         * @param aabb the axis-aligned bounding box to collect clusters.
         * @param callback the callback function to process the key of the cluster.
         */
        void
        IterateClustersInAabb(const Aabb &aabb, std::function<void(const Key &)> callback) const;

        /**
         * Get the surface data buffer.
         * @return vector of surface data.
         */
        [[nodiscard]] const std::vector<SurfData> &
        GetSurfaceDataBuffer() const;

        /**
         * Collect surface data in the given axis-aligned bounding box.
         * @param aabb the axis-aligned bounding box to collect surface data.
         * @param surface_data_indices vector of (distance to point, surface point index).
         */
        void
        CollectSurfaceDataInAabb(
            const Aabb &aabb,
            std::vector<std::pair<Dtype, std::size_t>> &surface_data_indices) const;

        /**
         * Get the boundary of the map.
         * @return the boundary of the map as an axis-aligned bounding box.
         */
        [[nodiscard]] Aabb
        GetMapBoundary() const;

        /**
         * Check if the given positions are in free space.
         * @param positions the positions to check.
         * @param in_free_space the vector to store the result. 1.0 if the position is in free
         * space, -1.0 otherwise.
         * @return true if this method is successful. false if the algorithm fails / is not
         * implemented.
         */
        [[nodiscard]] bool
        IsInFreeSpace(const Positions &positions, VectorX &in_free_space) const;

        // implement the methods required by AbstractSurfaceMapping for factory pattern

        [[nodiscard]] bool
        operator==(const AbstractSurfaceMapping &other) const override;

        [[nodiscard]] bool
        Write(std::ostream &s) const override;

        [[nodiscard]] bool
        Read(std::istream &s) override;

    private:
        static std::pair<Dtype, Dtype>
        Cartesian2Polar(Dtype x, Dtype y);

        void
        UpdateMapPoints0();

        void
        UpdateMapPoints1();

        void
        UpdateMapPoint(SurfData &surface_data, bool &updated, bool &to_remove);

        [[nodiscard]] std::pair<Key, int>
        ComputeSurfaceIndex1(const Position &pos_global) const;

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 2, bool>
        ComputeOcc(
            const Position &pos_local,
            Dtype &distance_local,
            Dtype &distance_pred,
            // Eigen::Ref<Scalar> distance_pred,
            // Eigen::Ref<Scalar> distance_pred_var,
            Dtype &occ) const;

        template<int D = Dim>
        [[nodiscard]] std::enable_if_t<D == 3, bool>
        ComputeOcc(
            const Position &pos_local,
            Dtype &distance_local,
            Dtype &distance_pred,
            // Eigen::Ref<Scalar> distance_pred,
            // Eigen::Ref<Scalar> distance_pred_var,
            Dtype &occ) const;

        template<int D = Dim>
        std::enable_if_t<D == 2>
        UpdateGradient(Dtype var_new, Dtype var_sum, const Gradient &grad_old, Gradient &grad_new);

        template<int D = Dim>
        std::enable_if_t<D == 3>
        UpdateGradient(Dtype var_new, Dtype var_sum, const Gradient &grad_old, Gradient &grad_new);

        void
        UpdateOccupancy();

        void
        AddNewMeasurement0();

        void
        AddNewMeasurement1();

        void
        RecordChangedKey(const Key &key);

        bool
        ComputeGradient1(
            const Position &pos_local,
            Gradient &gradient,
            Dtype &occ_mean,
            Dtype &distance_var);

        bool
        ComputeGradient2(
            const Eigen::Ref<const Position> &pos_local,
            Gradient &gradient,
            Dtype &occ_mean);

        void
        ComputeVariance(
            const Eigen::Ref<const Position> &pos_local,
            const Gradient &grad_local,
            const Dtype &distance,
            const Dtype &distance_var,
            const Dtype &occ_mean_abs,
            const Dtype &occ_abs,
            bool new_point,
            Dtype &var_position,
            Dtype &var_gradient) const;
    };

    using GpOccSurfaceMapping3Dd = GpOccSurfaceMapping<double, 3>;
    using GpOccSurfaceMapping3Df = GpOccSurfaceMapping<float, 3>;
    using GpOccSurfaceMapping2Dd = GpOccSurfaceMapping<double, 2>;
    using GpOccSurfaceMapping2Df = GpOccSurfaceMapping<float, 2>;

}  // namespace erl::sdf_mapping

#include "gp_occ_surface_mapping.tpp"

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping3Dd::Setting>
    : erl::sdf_mapping::GpOccSurfaceMapping3Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping3Df::Setting>
    : erl::sdf_mapping::GpOccSurfaceMapping3Df::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping2Dd::Setting>
    : erl::sdf_mapping::GpOccSurfaceMapping2Dd::Setting::YamlConvertImpl {};

template<>
struct YAML::convert<erl::sdf_mapping::GpOccSurfaceMapping2Df::Setting>
    : erl::sdf_mapping::GpOccSurfaceMapping2Df::Setting::YamlConvertImpl {};
