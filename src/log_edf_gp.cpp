#include "erl_sdf_mapping/log_edf_gp.hpp"

#include "erl_covariance/reduced_rank_covariance.hpp"

namespace erl::sdf_mapping {

    std::size_t
    LogEdfGaussianProcess::GetMemoryUsage() const {
        std::size_t memory_usage = NoisyInputGaussianProcess::GetMemoryUsage();
        memory_usage += sizeof(*this) - sizeof(NoisyInputGaussianProcess);
        return memory_usage;
    }

    long
    LogEdfGaussianProcess::LoadSurfaceData(
        std::vector<std::pair<double, std::shared_ptr<SurfaceData2D>>> &surface_data_vec,
        const Eigen::Vector2d &coord_origin,
        const double offset_distance,
        const double sensor_noise,
        const double max_valid_gradient_var,
        const double invalid_position_var) {

        SetKernelCoordOrigin(coord_origin);

        const long max_num_samples = std::min(m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));
        Reset(max_num_samples, 2);

        std::sort(surface_data_vec.begin(), surface_data_vec.end(), [](const auto &a, const auto &b) { return a.first < b.first; });

        long count = 0;
        for (auto &[distance, surface_data]: surface_data_vec) {
            m_mat_x_train_.col(count) = surface_data->position - offset_distance * surface_data->normal;
            m_vec_var_h_[count] = sensor_noise;
            m_vec_var_x_[count] = surface_data->var_position;
            if ((surface_data->var_normal > max_valid_gradient_var) ||                      // invalid gradient
                (surface_data->normal.norm() < 0.9)) {                                      // invalid normal
                m_vec_var_x_[count] = std::max(m_vec_var_x_[count], invalid_position_var);  // position is unreliable
            }
            if (++count >= m_mat_x_train_.cols()) { break; }  // reached max_num_samples
        }
        m_num_train_samples_ = count;
        m_vec_y_train_.setOnes(m_num_train_samples_);
        if (m_reduced_rank_kernel_) { UpdateKtrain(m_num_train_samples_); }
        return count;
    }

    long
    LogEdfGaussianProcess::LoadSurfaceData(
        std::vector<std::pair<double, std::shared_ptr<SurfaceData3D>>> &surface_data_vec,
        const Eigen::Vector3d &coord_origin,
        const double offset_distance,
        const double sensor_noise,
        const double max_valid_gradient_var,
        const double invalid_position_var) {

        SetKernelCoordOrigin(coord_origin);

        const long max_num_samples = std::min(m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));
        Reset(max_num_samples, 3);

        std::sort(surface_data_vec.begin(), surface_data_vec.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
        long count = 0;
        for (auto &[distance, surface_data]: surface_data_vec) {
            m_mat_x_train_.col(count) = surface_data->position - offset_distance * surface_data->normal;
            m_vec_var_h_[count] = sensor_noise;
            m_vec_var_x_[count] = surface_data->var_position;
            if ((surface_data->var_normal > max_valid_gradient_var) ||                      // invalid gradient
                (surface_data->normal.norm() < 0.9)) {                                      // invalid normal
                m_vec_var_x_[count] = std::max(m_vec_var_x_[count], invalid_position_var);  // position is unreliable
            }
            if (++count >= m_mat_x_train_.cols()) { break; }  // reached max_num_samples
        }
        m_num_train_samples_ = count;
        m_vec_y_train_.setOnes(m_num_train_samples_);
        if (m_reduced_rank_kernel_) { UpdateKtrain(m_num_train_samples_); }
        return count;
    }

    bool
    LogEdfGaussianProcess::Test(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
        Eigen::Ref<Eigen::MatrixXd> mat_f_out,
        Eigen::Ref<Eigen::MatrixXd> mat_var_out,
        Eigen::Ref<Eigen::MatrixXd> mat_cov_out) const {

        if (!NoisyInputGaussianProcess::Test(mat_x_test, mat_f_out, mat_var_out, mat_cov_out)) { return false; }
        const long dim = mat_x_test.rows();
        const long n = mat_x_test.cols();
        const double log_lambda = m_setting_->log_lambda;
        for (long i = 0; i < n; ++i) {
            // edf
            double *f = mat_f_out.col(i).data();
            const double f_log_gpis = f[0];
            f[0] = std::log(std::abs(f_log_gpis)) / -log_lambda;
            // gradient
            double norm = 0;
            const double d = -1.0 / (log_lambda * std::abs(f_log_gpis));  // d = -ln(f)/lambda, grad_d = -1/(lambda*f)*grad_f
            for (long j = 1; j <= dim; ++j) {                             // gradient
                double &grad = f[j];                                      // grad_f
                grad *= d;                                                // grad_d
                norm += grad * grad;
            }
            norm = std::sqrt(norm);
            if (norm > 1.e-15) {                                   // avoid zero division
                for (long j = 1; j <= dim; ++j) { f[j] /= norm; }  // gradient norm is always 1. https://en.wikipedia.org/wiki/Eikonal_equation
            }
        }
        return true;
    }

    bool
    LogEdfGaussianProcess::operator==(const LogEdfGaussianProcess &other) const {
        if (!NoisyInputGaussianProcess::operator==(other)) { return false; }
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        return true;
    }

    bool
    LogEdfGaussianProcess::Write(const std::string &filename) const {
        ERL_INFO("Writing LogEdfGaussianProcess to file: {}", filename);
        std::filesystem::create_directories(std::filesystem::path(filename).parent_path());
        std::ofstream file(filename, std::ios_base::out | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename);
            return false;
        }

        const bool success = Write(file);
        file.close();
        return success;
    }

    static const std::string kFileHeader = "# erl::sdf_mapping::LogEdfGaussianProcess";

    bool
    LogEdfGaussianProcess::Write(std::ostream &s) const {
        if (!NoisyInputGaussianProcess::Write(s)) {
            ERL_WARN("Failed to write parent class NoisyInputGaussianProcess.");
            return false;
        }
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        s << "end_of_LogEdfGaussianProcess" << std::endl;
        return s.good();
    }

    bool
    LogEdfGaussianProcess::Read(const std::string &filename) {
        ERL_INFO("Reading LogEdfGaussianProcess from file: {}", std::filesystem::absolute(filename));
        std::ifstream file(filename.c_str(), std::ios_base::in | std::ios_base::binary);
        if (!file.is_open()) {
            ERL_WARN("Failed to open file: {}", filename.c_str());
            return false;
        }

        const bool success = Read(file);
        file.close();
        return success;
    }

    bool
    LogEdfGaussianProcess::Read(std::istream &s) {
        if (!NoisyInputGaussianProcess::Read(s)) {
            ERL_WARN("Failed to read parent class NoisyInputGaussianProcess.");
            return false;
        }

        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (line.compare(0, kFileHeader.length(), kFileHeader) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeader.c_str());
            return false;
        }

        auto skip_line = [&s]() {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "setting",
            "end_of_LogEdfGaussianProcess",
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
                case 0: {  // setting
                    skip_line();
                    if (!m_setting_->Read(s)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    break;
                }
                case 1: {  // end_of_LogEdfGaussianProcess
                    skip_line();
                    return true;
                }
                default: {
                    ERL_WARN("Unknown token: {}", token);
                    return false;
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read NoisyInputGaussianProcess. Truncated file?");
        return false;  // should not reach here
    }
}  // namespace erl::sdf_mapping
