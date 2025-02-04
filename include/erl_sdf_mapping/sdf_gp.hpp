#pragma once
#include "log_edf_gp.hpp"

#include "erl_common/eigen.hpp"

#include <atomic>
#include <memory>

namespace erl::sdf_mapping {

    template<int Dim, typename SurfaceData>
    struct SdfGaussianProcess {
        static_assert(Dim == 2 || Dim == 3, "Dim must be 2 or 3.");

        bool active = false;
        std::atomic_bool locked_for_test = false;
        long num_edf_samples = 0;
        Eigen::Vector<double, Dim> position{};
        double half_size = 0;
        std::shared_ptr<LogEdfGaussianProcess> edf_gp = {};
        inline static const std::string kFileHeader = "# erl::sdf_mapping::SdfGaussianProcess";

        SdfGaussianProcess() = default;

        SdfGaussianProcess(const SdfGaussianProcess& other)
            : active(other.active),
              num_edf_samples(other.num_edf_samples),
              position(other.position),
              half_size(other.half_size) {
            if (other.edf_gp != nullptr) { edf_gp = std::make_shared<LogEdfGaussianProcess>(*other.edf_gp); }
        }

        SdfGaussianProcess(SdfGaussianProcess&& other) noexcept
            : active(other.active),
              num_edf_samples(other.num_edf_samples),
              position(std::move(other.position)),
              half_size(other.half_size),
              edf_gp(std::move(other.edf_gp)) {}

        SdfGaussianProcess&
        operator=(const SdfGaussianProcess& other) {
            if (this == &other) { return *this; }
            active = other.active;
            num_edf_samples = other.num_edf_samples;
            position = other.position;
            half_size = other.half_size;
            if (other.edf_gp != nullptr) { edf_gp = std::make_shared<LogEdfGaussianProcess>(*other.edf_gp); }
            return *this;
        }

        SdfGaussianProcess&
        operator=(SdfGaussianProcess&& other) noexcept {
            if (this == &other) { return *this; }
            active = other.active;
            num_edf_samples = other.num_edf_samples;
            position = other.position;
            half_size = other.half_size;
            edf_gp = std::move(other.edf_gp);
            return *this;
        }

        void
        Activate(const std::shared_ptr<LogEdfGaussianProcess::Setting>& edf_gp_setting) {
            if (edf_gp == nullptr) { edf_gp = std::make_shared<LogEdfGaussianProcess>(edf_gp_setting); }
            active = true;
        }

        void
        Deactivate() {
            active = false;
            num_edf_samples = 0;
        }

        [[nodiscard]] std::size_t
        GetMemoryUsage() const {
            std::size_t memory_usage = sizeof(SdfGaussianProcess);
            if (edf_gp != nullptr) { memory_usage += edf_gp->GetMemoryUsage(); }
            return memory_usage;
        }

        void
        LoadSurfaceData(
            std::vector<std::pair<double, std::shared_ptr<SurfaceData>>>& surface_data_vec,
            double offset_distance,
            double sensor_noise,
            double max_valid_gradient_var,
            double invalid_position_var) {
            num_edf_samples = edf_gp->LoadSurfaceData(surface_data_vec, position, offset_distance, sensor_noise, max_valid_gradient_var, invalid_position_var);
        }

        void
        Train() const {
            edf_gp->Train(num_edf_samples);
        }

        [[nodiscard]] bool
        Test(
            const Eigen::Vector<double, Dim>& test_position,
            Eigen::Ref<Eigen::Vector<double, Dim + 1>> f,
            Eigen::Ref<Eigen::Vector<double, Dim + 1>> var,
            Eigen::Ref<Eigen::Vector<double, Dim*(Dim + 1) / 2>> covariance,
            const double offset_distance,
            const double softmin_temperature,
            const bool use_gp_covariance,
            const bool compute_covariance) const {

            ERL_DEBUG_ASSERT(active, "SdfGaussianProcess is not active.");

            Eigen::VectorXd no_covariance;

            if (use_gp_covariance) {
                if (compute_covariance) {
                    edf_gp->Test(test_position, f, var, covariance);
                } else {
                    edf_gp->Test(test_position, f, var, no_covariance);
                }
                if (f.template tail<Dim>().norm() < 1.e-15) { return false; }  // invalid gradient, skip this GP
            } else {
                Eigen::VectorXd no_variance;
                edf_gp->Test(test_position, f, no_variance, no_covariance);
                Eigen::Vector<double, Dim> grad = f.template tail<Dim>();
                if (grad.norm() <= 1.e-15) { return false; }  // invalid gradient, skip this GP

                auto& mat_x = edf_gp->GetTrainInputSamplesBuffer();
                auto& vec_x_var = edf_gp->GetTrainInputSamplesVarianceBuffer();
                const long num_samples = edf_gp->GetNumTrainSamples();

                Eigen::VectorXd s(static_cast<int>(num_samples));
                double s_sum = 0;
                Eigen::VectorXd z_sdf(static_cast<int>(num_samples));
                Eigen::Matrix<double, Dim, Eigen::Dynamic> diff_z_sdf(Dim, num_samples);
                using SquareMatrix = Eigen::Matrix<double, Dim, Dim>;
                const SquareMatrix identity = SquareMatrix::Identity();
                for (long k = 0; k < num_samples; ++k) {
                    const Eigen::Vector<double, Dim> v = test_position - mat_x.col(k);
                    const double d = v.norm();  // distance to the training sample

                    z_sdf[k] = d;
                    s[k] = std::max(1.e-6, std::exp(-(d - f[0]) * softmin_temperature));
                    s_sum += s[k];

                    diff_z_sdf.col(k) = v / d;
                }

                // s /= s_sum;  // this line causes an extra for loop. replace it with the following line
                const double inv_s_sum = 1.0 / s_sum;

                const double sz_sdf = s.dot(z_sdf) * inv_s_sum;
                var[0] = 0.0;  // var_sdf
                SquareMatrix cov_grad = SquareMatrix::Zero();
                for (long k = 0; k < num_samples; ++k) {
                    double w = s[k] * (sz_sdf + 1.0 - z_sdf[k]) * inv_s_sum;
                    w = w * w * vec_x_var[k];
                    var[0] += w;
                    w = w / (z_sdf[k] * z_sdf[k]);
                    const SquareMatrix diff_grad = (identity - diff_z_sdf.col(k) * diff_z_sdf.col(k).transpose()) * w;
                    cov_grad += diff_grad;
                }
                var.template segment<Dim>(1) << cov_grad.diagonal();  // var_grad
                if (compute_covariance) {
                    if (Dim == 2) {
                        covariance << 0, 0, cov_grad(1, 0);
                    } else {
                        covariance << 0, 0, 0, cov_grad(1, 0), cov_grad(2, 0), cov_grad(2, 1);
                    }
                }
            }

            f[0] -= offset_distance;
            return true;
        }

        [[nodiscard]] bool
        operator==(const SdfGaussianProcess& other) const {
            if (active != other.active) { return false; }
            if (locked_for_test.load() != other.locked_for_test.load()) { return false; }
            if (num_edf_samples != other.num_edf_samples) { return false; }
            if (position != other.position) { return false; }
            if (half_size != other.half_size) { return false; }
            if (edf_gp == nullptr && other.edf_gp != nullptr) { return false; }
            if (edf_gp != nullptr && (other.edf_gp == nullptr || *edf_gp != *other.edf_gp)) { return false; }
            return true;
        }

        [[nodiscard]] bool
        operator!=(const SdfGaussianProcess& other) const {
            return !(*this == other);
        }

        [[nodiscard]] bool
        Write(std::ostream& s) const {
            s << kFileHeader << std::endl  //
              << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl;
            s << "active " << active << std::endl
              << "locked_for_test " << locked_for_test.load() << std::endl
              << "num_edf_samples " << num_edf_samples << std::endl;
            s << "position" << std::endl;
            if (!common::SaveEigenMatrixToBinaryStream(s, position)) { return false; }
            s << "half_size" << std::endl;
            s.write(reinterpret_cast<const char*>(&half_size), sizeof(half_size));
            s << "edf_gp " << (edf_gp != nullptr) << std::endl;
            if (edf_gp != nullptr && !edf_gp->Write(s)) { return false; }
            s << "end_of_SdfGaussianProcess" << std::endl;
            return s.good();
        }

        [[nodiscard]] bool
        Read(std::istream& s, const std::shared_ptr<LogEdfGaussianProcess::Setting>& edf_gp_setting) {

            if (!s.good()) {
                ERL_WARN("Input stream is not ready for reading");
                return false;
            }

            // check if the first line is valid
            std::string line;
            std::getline(s, line);
            if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
                ERL_WARN("Header does not start with \"{}\"", kFileHeader);
                return false;
            }

            auto skip_line = [&s]() {
                char c;
                do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
            };

            static const char* tokens[] = {
                "active",
                "locked_for_test",
                "num_edf_samples",
                "position",
                "half_size",
                "edf_gp",
                "end_of_SdfGaussianProcess",
            };

            // read data
            std::string token;
            int token_idx = 0;
            while (s.good()) {
                s >> token;
                if (token.compare(0, 1, "#") == 0) {
                    skip_line();  // comment line, skip forward until end of line
                    continue;
                }
                // non-comment line
                if (token != tokens[token_idx]) {
                    ERL_WARN("Expected token {}, got {}.", tokens[token_idx], token);  // check token
                    return false;
                }
                // reading state machine
                switch (token_idx) {
                    case 0: {  // active
                        s >> active;
                        break;
                    }
                    case 1: {  // locked_for_test
                        bool locked;
                        s >> locked;
                        locked_for_test.store(locked);
                        break;
                    }
                    case 2: {  // num_edf_samples
                        s >> num_edf_samples;
                        break;
                    }
                    case 3: {  // position
                        skip_line();
                        if (!common::LoadEigenMatrixFromBinaryStream(s, position)) {
                            ERL_WARN("Failed to read position.");
                            return false;
                        }
                        break;
                    }
                    case 4: {  // half_size
                        skip_line();
                        s.read(reinterpret_cast<char*>(&half_size), sizeof(half_size));
                        break;
                    }
                    case 5: {  // edf_gp
                        bool has_gp;
                        s >> has_gp;
                        if (has_gp) {
                            skip_line();
                            if (edf_gp == nullptr) { edf_gp = std::make_shared<LogEdfGaussianProcess>(edf_gp_setting); }
                            if (!edf_gp->Read(s)) { return false; }
                        }
                        break;
                    }
                    case 6: {  // end_of_SdfGaussianProcess
                        skip_line();
                        return true;
                    }
                    default: {  // should not reach here
                        ERL_FATAL("Internal error, should not reach here.");
                    }
                }
                ++token_idx;
            }
            ERL_WARN("Failed to read SdfGaussianProcess. Truncated file?");
            return false;  // should not reach here
        }
    };

}  // namespace erl::sdf_mapping
