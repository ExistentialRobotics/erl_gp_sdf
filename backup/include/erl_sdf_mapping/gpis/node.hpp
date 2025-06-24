#pragma once

#include "erl_common/eigen.hpp"

namespace erl::gp_sdf::gpis {

    /**
     * GpisData stores the distance, gradient and the corresponding variances of surface points for GPIS training.
     */
    template<int Dim>
    struct GpisData {

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
        Print(std::ostream &os) const {
            auto format = GetEigenTextFormat(common::EigenTextFormat::kNumpyFmt);
            os << "distance = " << distance << ", gradient = " << gradient.transpose().format(format) << ", var_position = " << var_position
               << ", var_gradient = " << var_gradient;
        }

        [[nodiscard]] bool
        operator==(const GpisData &other) const {
            const auto *other_ptr = dynamic_cast<const GpisData<Dim> *>(&other);
            return other_ptr != nullptr && distance == other_ptr->distance && gradient == other_ptr->gradient && var_position == other_ptr->var_position &&
                   var_gradient == other_ptr->var_gradient;
        }
    };

    using GpisData2D = GpisData<2>;
    using GpisData3D = GpisData<3>;

    template<int Dim>
    struct GpisNode {
        int type = 0;                                        // node type
        Eigen::VectorXd position;                            // node position to determine where to store in the tree
        std::shared_ptr<GpisData<Dim>> node_data = nullptr;  // attached data

        explicit GpisNode(const Eigen::Ref<const Eigen::Vector<double, Dim>> &position)
            : position(std::move(position)) {}

        [[nodiscard]] bool
        operator==(const GpisNode &other) const {
            return position == other.position && *node_data == *other.node_data;
        }

        bool
        operator!=(const GpisNode &other) const {
            return !(*this == other);
        }
    };

    using GpisNode2D = GpisNode<2>;
    using GpisNode3D = GpisNode<3>;

    template<int Dim>
    std::ostream &
    operator<<(std::ostream &os, const GpisData<Dim> &node_data) {
        node_data.Print(os);
        return os;
    }

}  // namespace erl::gp_sdf::gpis
