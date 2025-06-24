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
    SurfaceData<Dtype, Dim>::Write(std::ostream &s) const {
        using namespace common;
        static const TokenWriteFunctionPairs<SurfaceData> token_function_pairs = {
            {
                "position",
                [](const SurfaceData *data, std::ostream &stream) {
                    return SaveEigenMatrixToBinaryStream(stream, data->position) && stream.good();
                },
            },
            {
                "normal",
                [](const SurfaceData *data, std::ostream &stream) {
                    return SaveEigenMatrixToBinaryStream(stream, data->normal) && stream.good();
                },
            },
            {
                "var_position",
                [](const SurfaceData *data, std::ostream &stream) {
                    stream.write(
                        reinterpret_cast<const char *>(&data->var_position),
                        sizeof(data->var_position));
                    return true;
                },
            },
            {
                "var_normal",
                [](const SurfaceData *data, std::ostream &stream) {
                    stream.write(
                        reinterpret_cast<const char *>(&data->var_normal),
                        sizeof(data->var_normal));
                    return true;
                },
            },
        };
        return WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::Read(std::istream &s) {
        using namespace common;
        static const TokenReadFunctionPairs<SurfaceData> token_function_pairs = {
            {
                "position",
                [](SurfaceData *data, std::istream &stream) {
                    return LoadEigenMatrixFromBinaryStream(stream, data->position) && stream.good();
                },
            },
            {
                "normal",
                [](SurfaceData *data, std::istream &stream) {
                    return LoadEigenMatrixFromBinaryStream(stream, data->normal) && stream.good();
                },
            },
            {
                "var_position",
                [](SurfaceData *data, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&data->var_position),
                        sizeof(data->var_position));
                    if (!stream.good()) {
                        ERL_WARN("Failed to read var_position.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_normal",
                [](SurfaceData *data, std::istream &stream) {
                    stream.read(
                        reinterpret_cast<char *>(&data->var_normal),
                        sizeof(data->var_normal));
                    if (!stream.good()) {
                        ERL_WARN("Failed to read var_normal.");
                        return false;
                    }
                    return true;
                },
            },
        };
        return ReadTokens(s, this, token_function_pairs);
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
