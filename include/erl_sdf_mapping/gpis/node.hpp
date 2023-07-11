#pragma once

#include "erl_common/eigen.hpp"
#include "erl_geometry/node.hpp"

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

        explicit GpisData(double distance)
            : distance(distance) {}

        GpisData(double distance, Eigen::Vector<double, Dim> gradient)
            : distance(distance),
              gradient(std::move(gradient)) {}

        GpisData(double distance, Eigen::Vector<double, Dim> gradient, double var_position, double var_gradient)
            : distance(distance),
              gradient(std::move(gradient)),
              var_position(var_position),
              var_gradient(var_gradient) {}

        inline void
        UpdateData(double new_distance, Eigen::Vector<double, Dim> new_gradient, double new_var_position, double new_var_gradient) {
            distance = new_distance;
            gradient = std::move(new_gradient);
            var_position = new_var_position;
            var_gradient = new_var_gradient;
        }

        void
        Print(std::ostream &os) const override {
            auto format = common::GetEigenTextFormat(common::EigenTextFormat::kNumpyFmt);
            os << "distance = " << distance << ", gradient = " << gradient.transpose().format(format) << ", var_position = " << var_position
               << ", var_gradient = " << var_gradient;
        }

        [[nodiscard]] bool
        operator==(const geometry::NodeData &other) const override {
            const auto *other_ptr = dynamic_cast<const GpisData<Dim> *>(&other);
            return other_ptr != nullptr && distance == other_ptr->distance && gradient == other_ptr->gradient && var_position == other_ptr->var_position &&
                   var_gradient == other_ptr->var_gradient;
        }
    };

    typedef GpisData<2> GpisData2D;
    typedef GpisData<3> GpisData3D;

    enum class GpisNodeType { kSurface = 0 };

    template<int Dim>
    struct GpisNode : public geometry::Node {

        // Eigen::Vector<double, Dim> position;

        explicit GpisNode(const Eigen::Ref<const Eigen::Vector<double, Dim>> &position)
            : Node(int(GpisNodeType::kSurface), position, std::make_shared<GpisData<Dim>>()) {}

        // [[nodiscard]] Eigen::VectorXd
        // GetPosition() const override {
        //     return position;
        // }

        [[nodiscard]] bool
        operator==(const geometry::Node &other) const override {
            return position == other.position && *node_data == *other.node_data;
        }
    };

    extern const std::vector<int> kGpiSdfNodeTypes;

    typedef GpisNode<2> GpisNode2D;
    typedef GpisNode<3> GpisNode3D;

}  // namespace erl::gp_sdf
