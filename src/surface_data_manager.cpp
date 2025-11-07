#include "erl_gp_sdf/surface_data_manager.hpp"

namespace erl::gp_sdf {

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::operator==(const SurfaceData &other) const {
        return position == other.position && normal == other.normal &&
               var_position == other.var_position && var_normal == other.var_normal;
    }

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::operator!=(const SurfaceData &other) const {
        return !(*this == other);
    }

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::Write(std::ostream &stream) const {
        using namespace common;
        using namespace common::serialization;
        static const TokenWriteFunctionPairs<SurfaceData> token_function_pairs = {
            {
                "position",
                [](const SurfaceData *data, std::ostream &s) {
                    return SaveEigenMatrixToBinaryStream(s, data->position) && s.good();
                },
            },
            {
                "normal",
                [](const SurfaceData *data, std::ostream &s) {
                    return SaveEigenMatrixToBinaryStream(s, data->normal) && s.good();
                },
            },
            {
                "var_position",
                [](const SurfaceData *data, std::ostream &s) {
                    s.write(
                        reinterpret_cast<const char *>(&data->var_position),
                        sizeof(data->var_position));
                    return true;
                },
            },
            {
                "var_normal",
                [](const SurfaceData *data, std::ostream &s) {
                    s.write(
                        reinterpret_cast<const char *>(&data->var_normal),
                        sizeof(data->var_normal));
                    return true;
                },
            },
        };
        return WriteTokens(stream, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::Read(std::istream &stream) {
        using namespace common;
        using namespace common::serialization;
        static const TokenReadFunctionPairs<SurfaceData> token_function_pairs = {
            {
                "position",
                [](SurfaceData *data, std::istream &s) {
                    return LoadEigenMatrixFromBinaryStream(s, data->position) && s.good();
                },
            },
            {
                "normal",
                [](SurfaceData *data, std::istream &s) {
                    return LoadEigenMatrixFromBinaryStream(s, data->normal) && s.good();
                },
            },
            {
                "var_position",
                [](SurfaceData *data, std::istream &s) {
                    s.read(
                        reinterpret_cast<char *>(&data->var_position),
                        sizeof(data->var_position));
                    if (!s.good()) {
                        ERL_WARN("Failed to read var_position.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_normal",
                [](SurfaceData *data, std::istream &s) {
                    s.read(reinterpret_cast<char *>(&data->var_normal), sizeof(data->var_normal));
                    if (!s.good()) {
                        ERL_WARN("Failed to read var_normal.");
                        return false;
                    }
                    return true;
                },
            },
        };
        return ReadTokens(stream, this, token_function_pairs);
    }

    template class SurfaceData<double, 2>;
    template class SurfaceData<float, 2>;
    template class SurfaceData<double, 3>;
    template class SurfaceData<float, 3>;
    template class SurfaceDataManager<double, 2>;
    template class SurfaceDataManager<float, 2>;
    template class SurfaceDataManager<double, 3>;
    template class SurfaceDataManager<float, 3>;
}  // namespace erl::gp_sdf
