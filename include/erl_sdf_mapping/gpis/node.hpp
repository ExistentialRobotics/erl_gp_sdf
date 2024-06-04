#pragma once

#include "erl_geometry/node.hpp"
#include "erl_common/eigen.hpp"

namespace erl::sdf_mapping::gpis {

    /**
     * GpisData stores the distance, gradient and the corresponding variances of surface points for GPIS training.
     */
    template<int Dim>
    struct GpisData : public geometry::NodeData {

        double distance = 0.;
        Eigen::Vector<double, Dim> gradient = Eigen::Vector<double, Dim>::Zero();
        double var_position = 0.;  // input to GP
        double var_gradient = 0.;

        GpisData() = default;

        explicit GpisData(const double distance)
            : distance(distance) {}

        GpisData(const double distance, Eigen::Vector<double, Dim> gradient)
            : distance(distance),
              gradient(std::move(gradient)) {}

        GpisData(const double distance, Eigen::Vector<double, Dim> gradient, const double var_position, const double var_gradient)
            : distance(distance),
              gradient(std::move(gradient)),
              var_position(var_position),
              var_gradient(var_gradient) {}

        void
        UpdateData(const double new_distance, Eigen::Vector<double, Dim> new_gradient, const double new_var_position, const double new_var_gradient) {
            distance = new_distance;
            gradient = std::move(new_gradient);
            var_position = new_var_position;
            var_gradient = new_var_gradient;
        }

        void
        Print(std::ostream &os) const override {
            auto format = GetEigenTextFormat(common::EigenTextFormat::kNumpyFmt);
            os << "distance = " << distance << ", gradient = " << gradient.transpose().format(format) << ", var_position = " << var_position
               << ", var_gradient = " << var_gradient;
        }

        [[nodiscard]] bool
        operator==(const NodeData &other) const override {
            const auto *other_ptr = dynamic_cast<const GpisData<Dim> *>(&other);
            return other_ptr != nullptr && distance == other_ptr->distance && gradient == other_ptr->gradient && var_position == other_ptr->var_position &&
                   var_gradient == other_ptr->var_gradient;
        }
    };

    using GpisData2D = GpisData<2>;
    using GpisData3D = GpisData<3>;

    template<int Dim>
    struct GpisNode : public geometry::Node {

        explicit GpisNode(const Eigen::Ref<const Eigen::Vector<double, Dim>> &position)
            : Node(0, position, std::make_shared<GpisData<Dim>>()) {}

        [[nodiscard]] bool
        operator==(const Node &other) const override {
            return position == other.position && *node_data == *other.node_data;
        }
    };

    using GpisNode2D = GpisNode<2>;
    using GpisNode3D = GpisNode<3>;

}  // namespace erl::sdf_mapping::gpis
