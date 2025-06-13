#include "erl_common/pybind11.hpp"
#include "erl_gp_sdf/gp_occ_surface_mapping.hpp"
#include "erl_gp_sdf/gp_sdf_mapping.hpp"

template<typename Dtype, int Dim>
void
BindGpSdfMappingImpl(const py::module &m, const char *name) {
    using namespace erl::common;
    using namespace erl::geometry;
    using namespace erl::sdf_mapping;
    using T = GpSdfMapping<Dtype, Dim>;
    using Setting = typename T::Setting;
    using Positions = typename T::Positions;

    py::class_<T, std::shared_ptr<T>> sdf_mapping(m, name);

    sdf_mapping
        .def(
            py::init<
                std::shared_ptr<Setting>,
                std::shared_ptr<AbstractSurfaceMapping<Dtype, Dim>>>(),
            py::arg("setting"),
            py::arg("surface_mapping"))
        .def_property_readonly("setting", &T::GetSetting)
        .def("update_gp_sdf", &T::UpdateGpSdf, py::arg("time_budget_us"))
        .def(
            "test",
            [](T &self, const Eigen::Ref<const Positions> &positions) -> std::optional<py::dict> {
                typename T::Distances distances;
                typename T::Gradients gradients;
                typename T::Variances variances;
                typename T::Covariances covariances;

                if (self.Test(positions, distances, gradients, variances, covariances)) {
                    py::dict result;
                    result["distances"] = distances;
                    result["gradients"] = gradients;
                    result["variances"] = variances;
                    result["covariances"] = covariances;
                    return result;
                }
                return std::nullopt;
            },
            py::arg("positions"))
        .def_property_readonly("used_gps", &T::GetUsedGps, "GPs used by the last test call")
        .def_property_readonly("gps", &T::GetGpMap);
}

void
BindGpSdfMapping(const py::module &m) {
    using namespace erl::sdf_mapping;
    BindGpSdfMappingImpl<double, 3>(m, "GpSdfMapping3Dd");
    BindGpSdfMappingImpl<float, 3>(m, "GpSdfMapping3Df");
    BindGpSdfMappingImpl<double, 2>(m, "GpSdfMapping2Dd");
    BindGpSdfMappingImpl<float, 2>(m, "GpSdfMapping2Df");
}
