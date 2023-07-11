#pragma once

#include <memory>

#include "erl_common/eigen.hpp"

namespace erl::sdf_mapping::gpis {
    struct TestBuffer {
        using InBuffer = Eigen::Ref<const Eigen::Matrix2Xd>;
        using OutVectorBuffer = Eigen::Ref<Eigen::VectorXd>;
        using OutMatrixBuffer = Eigen::Ref<Eigen::Matrix2Xd>;

        ssize_t n = 0;
        std::unique_ptr<InBuffer> positions;
        // output buffers
        std::unique_ptr<OutVectorBuffer> distances;
        std::unique_ptr<OutMatrixBuffer> gradients;
        std::unique_ptr<OutVectorBuffer> distance_variances;
        std::unique_ptr<OutMatrixBuffer> gradient_variances;

        [[nodiscard]] inline ssize_t
        Size() const {
            return n;
        }

        inline bool
        ConnectBuffers(
            const InBuffer &xy,
            OutVectorBuffer::PlainMatrix &distances_buf,
            OutMatrixBuffer::PlainMatrix &gradients_buf,
            OutVectorBuffer::PlainMatrix &distance_variances_buf,
            OutMatrixBuffer::PlainMatrix &gradient_variances_buf) {

            positions = std::make_unique<InBuffer>(xy);

            n = positions->cols();
            if (n == 0) { return false; }

            distances_buf.resize(n);
            gradients_buf.resize(2, n);
            distance_variances_buf.resize(n);
            gradient_variances_buf.resize(2, n);

            distances = std::make_unique<OutVectorBuffer>(distances_buf);
            gradients = std::make_unique<OutMatrixBuffer>(gradients_buf);
            distance_variances = std::make_unique<OutVectorBuffer>(distance_variances_buf);
            gradient_variances = std::make_unique<OutMatrixBuffer>(gradient_variances_buf);

            return true;
        }

        inline void
        DisconnectBuffers() {
            n = 0;
            positions = nullptr;
            distances = nullptr;
            gradients = nullptr;
            distance_variances = nullptr;
            gradient_variances = nullptr;
        }
    };

}  // namespace erl::sdf_mapping::gpis
