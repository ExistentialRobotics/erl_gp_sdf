#include "erl_sdf_mapping/sign_gp.hpp"

namespace erl::sdf_mapping {

    std::size_t
    SignGaussianProcess::GetMemoryUsage() const {
        std::size_t memory_usage = NoisyInputGaussianProcess::GetMemoryUsage();
        memory_usage += sizeof(*this) - sizeof(NoisyInputGaussianProcess);
        return memory_usage;
    }

    long
    SignGaussianProcess::LoadSurfaceData(
        std::vector<std::pair<double, std::shared_ptr<SurfaceData2D>>> &surface_data_vec,
        const Eigen::Vector2d &coord_origin,
        const double sensor_noise,
        const double max_valid_gradient_var,
        const double invalid_position_var) {

        SetKernelCoordOrigin(coord_origin);

        const long max_num_samples = std::min(m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));
        Reset(max_num_samples, 2);

        const double offset_distance = m_setting_->offset_distance;
        long count = 0;
        for (auto &[distance, surface_data]: surface_data_vec) {
            m_mat_x_train_.col(count) = surface_data->position;
            m_vec_y_train_[count] = offset_distance;
            m_vec_var_h_[count] = sensor_noise;
            m_vec_var_grad_[count] = surface_data->var_normal;

            if ((surface_data->var_normal > max_valid_gradient_var) ||  // invalid gradient
                (surface_data->normal.norm() < 0.9)) {                  // invalid normal
                m_vec_var_x_[count] = invalid_position_var;             // position is unreliable
                m_vec_grad_flag_[count] = false;
                m_mat_grad_train_.col(count++).setZero();
            } else {
                m_vec_var_x_[count] = surface_data->var_position;
                m_vec_grad_flag_[count] = true;
                m_mat_grad_train_.col(count++) = surface_data->normal;
            }

            if (count >= m_mat_x_train_.cols()) { break; }  // reached max_num_samples
        }
        m_num_train_samples_ = count;
        if (m_reduced_rank_kernel_ && m_num_train_samples_ > 0) { UpdateKtrain(m_num_train_samples_); }
        return count;
    }

    long
    SignGaussianProcess::LoadSurfaceData(
        std::vector<std::pair<double, std::shared_ptr<SurfaceData3D>>> &surface_data_vec,
        const Eigen::Vector3d &coord_origin,
        double sensor_noise,
        double max_valid_gradient_var,
        double invalid_position_var) {

        SetKernelCoordOrigin(coord_origin);

        const long max_num_samples = std::min(m_setting_->max_num_samples, static_cast<long>(surface_data_vec.size()));
        Reset(max_num_samples, 3);
        const double offset_distance = m_setting_->offset_distance;

        long count = 0;
        for (auto &[distance, surface_data]: surface_data_vec) {
            const Eigen::Vector3d delta = offset_distance * surface_data->normal;

            m_mat_x_train_.col(count) = surface_data->position - delta;
            m_vec_y_train_[count] = -offset_distance;
            m_vec_var_h_[count] = sensor_noise;
            m_vec_var_grad_[count] = surface_data->var_normal;

            const bool invalid_normal = (surface_data->var_normal > max_valid_gradient_var) || (surface_data->normal.norm() < 0.9);

            if (invalid_normal) {
                m_vec_var_x_[count] = invalid_position_var;  // position is unreliable
                m_vec_grad_flag_[count] = false;
                m_mat_grad_train_.col(count++).setZero();
            } else {
                m_vec_var_x_[count] = surface_data->var_position;
                m_vec_grad_flag_[count] = true;
                m_mat_grad_train_.col(count++) = surface_data->normal;
            }

            if (count >= m_mat_x_train_.cols()) { break; }  // reached max_num_samples

            m_mat_x_train_.col(count) = surface_data->position + delta;
            m_vec_y_train_[count] = offset_distance;
            m_vec_var_h_[count] = sensor_noise;
            m_vec_var_grad_[count] = surface_data->var_normal;

            if (invalid_normal) {
                m_vec_var_x_[count] = invalid_position_var;  // position is unreliable
                m_vec_grad_flag_[count] = false;
                m_mat_grad_train_.col(count++).setZero();
            } else {
                m_vec_var_x_[count] = surface_data->var_position;
                m_vec_grad_flag_[count] = true;
                m_mat_grad_train_.col(count++) = -surface_data->normal;
            }

            if (count >= m_mat_x_train_.cols()) { break; }  // reached max_num_samples
        }
        m_num_train_samples_ = count;
        if (m_reduced_rank_kernel_ && m_num_train_samples_ > 0) { UpdateKtrain(m_num_train_samples_); }
        return count;
    }

    bool
    SignGaussianProcess::Test(
        const Eigen::Ref<const Eigen::MatrixXd> &mat_x_test,
        Eigen::Ref<Eigen::MatrixXd> mat_f_out,
        Eigen::Ref<Eigen::MatrixXd> /*mat_var_out*/,
        Eigen::Ref<Eigen::MatrixXd> /*mat_cov_out*/) const {

        if (!m_trained_) {
            ERL_WARN("The model has not been trained.");
            return false;
        }

        const long n = mat_x_test.cols();
        if (n == 0) { return false; }

        // compute mean and gradient of the test queries
        ERL_ASSERTM(mat_f_out.rows() >= 1, "mat_f_out.rows() = {}, it should be >= Dim + 1 = {}.", mat_f_out.rows(), 1);
        ERL_ASSERTM(mat_f_out.cols() >= n, "mat_f_out.cols() = {}, not enough for {} test queries.", mat_f_out.cols(), n);

        const auto [ktest_rows, ktest_cols] = m_kernel_->GetMinimumKtestSize(m_num_train_samples_, m_num_train_samples_with_grad_, m_x_dim_, n, false);
        Eigen::MatrixXd ktest(ktest_rows, ktest_cols);  // (dim of train samples, dim of test queries)
        const auto [output_rows, output_cols] =
            m_kernel_->ComputeKtestWithGradient(m_mat_x_train_, m_num_train_samples_, m_vec_grad_flag_, mat_x_test, n, false, ktest);
        (void) output_rows;
        (void) output_cols;
        ERL_DEBUG_ASSERT(
            output_rows == ktest_rows && output_cols == ktest_cols,
            "output_size = ({}, {}), it should be ({}, {}).",
            output_rows,
            output_cols,
            ktest_rows,
            ktest_cols);

        // compute value prediction
        /// ktest.T * m_vec_alpha_ = [h(x1),...,h(xn),dh(x1)/dx_1,...,dh(xn)/dx_1,...,dh(x1)/dx_dim,...,dh(xn)/dx_dim]
        const auto vec_alpha = m_vec_alpha_.head(ktest_rows);
        // const double offset_distance = m_setting_->offset_distance;
        for (long i = 0; i < n; ++i) {
            double &f = mat_f_out(0, i);
            f = ktest.col(i).dot(vec_alpha);  // + offset_distance;  // h(x)
            f = f > 0. ? 1. : -1.;            // sign(h(x))
        }
        return true;
    }

    bool
    SignGaussianProcess::operator==(const SignGaussianProcess &other) const {
        if (!NoisyInputGaussianProcess::operator==(other)) { return false; }
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        return true;
    }

    bool
    SignGaussianProcess::Write(const std::string &filename) const {
        ERL_INFO("Writing SignGaussianProcess to file: {}", filename);
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

    static const std::string kFileHeader = "# erl::sdf_mapping::SignGaussianProcess";

    bool
    SignGaussianProcess::Write(std::ostream &s) const {
        if (!NoisyInputGaussianProcess::Write(s)) {
            ERL_WARN("Failed to write parent class SignGaussianProcess.");
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
        s << "end_of_SignGaussianProcess" << std::endl;
        return s.good();
    }

    bool
    SignGaussianProcess::Read(const std::string &filename) {
        ERL_INFO("Reading SignGaussianProcess from file: {}", std::filesystem::absolute(filename));
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
    SignGaussianProcess::Read(std::istream &s) {
        if (!NoisyInputGaussianProcess::Read(s)) {
            ERL_WARN("Failed to read parent class SignGaussianProcess.");
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
            "end_of_SignGaussianProcess",
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
        ERL_WARN("Failed to read SignGaussianProcess. Truncated file?");
        return false;  // should not reach here
    }

}  // namespace erl::sdf_mapping
