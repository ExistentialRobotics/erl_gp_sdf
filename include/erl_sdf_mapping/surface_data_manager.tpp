#pragma once

namespace erl::sdf_mapping {

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::operator==(const SurfaceData &other) const {
        return position == other.position && normal == other.normal && var_position == other.var_position && var_normal == other.var_normal;
    }

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::operator!=(const SurfaceData &other) const {
        return !(*this == other);
    }

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::Write(std::ostream &s) const {
        static const std::vector<std::pair<const char *, std::function<bool(const SurfaceData *, std::ostream &)>>> token_function_pairs = {
            {
                "position",
                [](const SurfaceData *data, std::ostream &stream) {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, data->position)) {
                        ERL_WARN("Failed to write position.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "normal",
                [](const SurfaceData *data, std::ostream &stream) {
                    if (!common::SaveEigenMatrixToBinaryStream(stream, data->normal)) {
                        ERL_WARN("Failed to write normal.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_position",
                [](const SurfaceData *data, std::ostream &stream) {
                    stream.write(reinterpret_cast<const char *>(&data->var_position), sizeof(data->var_position));
                    return true;
                },
            },
            {
                "var_normal",
                [](const SurfaceData *data, std::ostream &stream) {
                    stream.write(reinterpret_cast<const char *>(&data->var_normal), sizeof(data->var_normal));
                    return true;
                },
            },
        };
        return common::WriteTokens(s, this, token_function_pairs);
    }

    template<typename Dtype, int Dim>
    bool
    SurfaceData<Dtype, Dim>::Read(std::istream &s) {
        static const std::vector<std::pair<const char *, std::function<bool(SurfaceData *, std::istream &)>>> token_function_pairs = {
            {
                "position",
                [](SurfaceData *data, std::istream &stream) {
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, data->position)) {
                        ERL_WARN("Failed to read position.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "normal",
                [](SurfaceData *data, std::istream &stream) {
                    if (!common::LoadEigenMatrixFromBinaryStream(stream, data->normal)) {
                        ERL_WARN("Failed to read normal.");
                        return false;
                    }
                    return true;
                },
            },
            {
                "var_position",
                [](SurfaceData *data, std::istream &stream) {
                    stream.read(reinterpret_cast<char *>(&data->var_position), sizeof(data->var_position));
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
                    stream.read(reinterpret_cast<char *>(&data->var_normal), sizeof(data->var_normal));
                    if (!stream.good()) {
                        ERL_WARN("Failed to read var_normal.");
                        return false;
                    }
                    return true;
                },
            },
        };
        return common::ReadTokens(s, this, token_function_pairs);
    }

}  // namespace erl::sdf_mapping
