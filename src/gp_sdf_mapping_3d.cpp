#include "erl_sdf_mapping/gp_sdf_mapping_3d.hpp"

#include "erl_common/angle_utils.hpp"
#include "erl_common/tracy.hpp"

#include <algorithm>
#include <chrono>
#include <thread>
#include <vector>

namespace erl::sdf_mapping {

    bool
    GpSdfMapping3D::Gp::operator==(const Gp &other) const {
        if (active != other.active) { return false; }
        if (locked_for_test.load() != other.locked_for_test.load()) { return false; }
        if (num_train_samples != other.num_train_samples) { return false; }
        if (position != other.position) { return false; }
        if (half_size != other.half_size) { return false; }
        if (gp == nullptr && other.gp != nullptr) { return false; }
        if (gp != nullptr && (other.gp == nullptr || *gp != *other.gp)) { return false; }
        return true;
    }

    static const std::string kFileHeaderGp = "# erl::sdf_mapping::GpSdfMapping3D::Gp";

    bool
    GpSdfMapping3D::Gp::Write(std::ostream &s) const {
        s << kFileHeaderGp << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl;
        s << "active " << active << std::endl
          << "locked_for_test " << locked_for_test.load() << std::endl
          << "num_train_samples " << num_train_samples << std::endl;
        s << "position" << std::endl;
        if (!common::SaveEigenMatrixToBinaryStream(s, position)) { return false; }
        s << "half_size" << std::endl;
        s.write(reinterpret_cast<const char *>(&half_size), sizeof(half_size));
        s << "gp " << (gp != nullptr) << std::endl;
        if (gp != nullptr && !gp->Write(s)) { return false; }
        s << "end_of_GpSdfMapping3D::Gp" << std::endl;
        return s.good();
    }

    bool
    GpSdfMapping3D::Gp::Read(std::istream &s, const std::shared_ptr<LogSdfGaussianProcess::Setting> &setting) {
        if (!s.good()) {
            ERL_WARN("Input stream is not ready for reading");
            return false;
        }

        // check if the first line is valid
        std::string line;
        std::getline(s, line);
        if (line.compare(0, kFileHeaderGp.length(), kFileHeaderGp) != 0) {  // check if the first line is valid
            ERL_WARN("Header does not start with \"{}\"", kFileHeaderGp.c_str());
            return false;
        }

        auto skip_line = [&s]() {
            char c;
            do { c = static_cast<char>(s.get()); } while (s.good() && c != '\n');
        };

        static const char *tokens[] = {
            "active",
            "locked_for_test",
            "num_train_samples",
            "position",
            "half_size",
            "gp",
            "end_of_GpSdfMapping3D::Gp",
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
                case 0: {
                    s >> active;
                    break;
                }
                case 1: {
                    bool locked;
                    s >> locked;
                    locked_for_test.store(locked);
                    break;
                }
                case 2: {
                    s >> num_train_samples;
                    break;
                }
                case 3: {
                    skip_line();
                    if (!common::LoadEigenMatrixFromBinaryStream(s, position)) {
                        ERL_WARN("Failed to read position.");
                        return false;
                    }
                    break;
                }
                case 4: {
                    skip_line();
                    s.read(reinterpret_cast<char *>(&half_size), sizeof(half_size));
                    break;
                }
                case 5: {
                    bool has_gp;
                    s >> has_gp;
                    if (has_gp) {
                        skip_line();
                        if (gp == nullptr) { gp = std::make_shared<LogSdfGaussianProcess>(setting); }
                        if (!gp->Read(s)) { return false; }
                    }
                    break;
                }
                case 6: {
                    skip_line();
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read GpSdfMapping3D::Gp. Truncated file?");
        return false;  // should not reach here
    }

    bool
    GpSdfMapping3D::TestBuffer::ConnectBuffers(
        const Eigen::Ref<const Eigen::Matrix3Xd> &positions_in,
        Eigen::VectorXd &distances_out,
        Eigen::Matrix3Xd &gradients_out,
        Eigen::Matrix4Xd &variances_out,
        Eigen::Matrix6Xd &covariances_out,
        const bool compute_covariance) {

        positions = nullptr;
        distances = nullptr;
        gradients = nullptr;
        variances = nullptr;
        covariances = nullptr;
        const long n = positions_in.cols();
        if (n == 0) return false;

        distances_out.resize(n);
        gradients_out.resize(3, n);
        variances_out.resize(4, n);
        if (compute_covariance) { covariances_out.resize(6, n); }
        this->positions = std::make_unique<Eigen::Ref<const Eigen::Matrix3Xd>>(positions_in);
        this->distances = std::make_unique<Eigen::Ref<Eigen::VectorXd>>(distances_out);
        this->gradients = std::make_unique<Eigen::Ref<Eigen::Matrix3Xd>>(gradients_out);
        this->variances = std::make_unique<Eigen::Ref<Eigen::Matrix4Xd>>(variances_out);
        this->covariances = std::make_unique<Eigen::Ref<Eigen::Matrix6Xd>>(covariances_out);
        return true;
    }

    void
    GpSdfMapping3D::TestBuffer::DisconnectBuffers() {
        positions = nullptr;
        distances = nullptr;
        gradients = nullptr;
        variances = nullptr;
        covariances = nullptr;
    }

    GpSdfMapping3D::GpSdfMapping3D(std::shared_ptr<Setting> setting)
        : m_setting_(NotNull(std::move(setting), "setting is nullptr.")),
          m_surface_mapping_(geometry::AbstractSurfaceMapping::CreateSurfaceMapping<geometry::AbstractSurfaceMapping3D>(
              m_setting_->surface_mapping_type,
              m_setting_->surface_mapping)) {
        ERL_ASSERTM(m_surface_mapping_ != nullptr, "surface_mapping is nullptr.");

        // get log dir from env
        if (m_setting_->log_timing) {
            char *log_dir_env = std::getenv("LOG_DIR");
            const std::filesystem::path log_dir = log_dir_env == nullptr ? std::filesystem::current_path() : std::filesystem::path(log_dir_env);
            const std::filesystem::path train_log_file_name = log_dir / "gp_sdf_mapping_3d_train.csv";
            const std::filesystem::path test_log_file_name = log_dir / "gp_sdf_mapping_3d_test.csv";
            if (std::filesystem::exists(train_log_file_name)) { std::filesystem::remove(train_log_file_name); }
            if (std::filesystem::exists(test_log_file_name)) { std::filesystem::remove(test_log_file_name); }
            m_train_log_file_.open(train_log_file_name);
            m_test_log_file_.open(test_log_file_name);
            ERL_WARN_COND(!m_train_log_file_.is_open(), ("Failed to open " + train_log_file_name.string()).c_str());
            ERL_WARN_COND(!m_test_log_file_.is_open(), ("Failed to open " + test_log_file_name.string()).c_str());
            m_train_log_file_ << "travel_distance,surf_mapping_time(us),gp_data_update_time(us),gp_delay_cnt,"
                              << "gp_train_time(us),total_gp_update_time(ms),total_update_time(ms)" << std::endl
                              << std::flush;
            m_test_log_file_ << "travel_distance,gp_search_time(us),gp_train_time(us),gp_test_time(us),total_test_time(ms)" << std::endl << std::flush;
        }
    }

    bool
    GpSdfMapping3D::Update(
        const Eigen::Ref<const Eigen::Matrix3d> &rotation,
        const Eigen::Ref<const Eigen::Vector3d> &translation,
        const Eigen::Ref<const Eigen::MatrixXd> &ranges) {

        ERL_TRACY_FRAME_MARK_START();

        if (m_setting_->log_timing) {
            std::lock_guard lock(m_log_mutex_);
            if (m_last_position_.has_value()) {
                const Eigen::Vector3d delta = translation - m_last_position_.value();
                m_travel_distance_ += delta.norm();
            } else {
                m_travel_distance_ = 0;
            }
            m_last_position_ = translation;
            m_train_log_file_ << m_travel_distance_;
        }

        const std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

        double time_budget = 1e6 / m_setting_->update_hz;  // us
        bool success;
        std::chrono::high_resolution_clock::time_point t0, t1;
        double dt;
        {
            std::lock_guard lock(m_mutex_);
            t0 = std::chrono::high_resolution_clock::now();
            success = m_surface_mapping_->Update(rotation, translation, ranges);
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (m_setting_->log_timing) { m_train_log_file_ << "," << dt; }  // surface_mapping_time
            ERL_INFO("Surface mapping update time: {:f} ms.", dt);
            ERL_TRACY_PLOT("surface_mapping_update_time (ms)", dt);
        }
        time_budget -= dt * 1e3;  // us

        t0 = std::chrono::high_resolution_clock::now();
        if (success) {
            std::lock_guard lock(m_mutex_);
            UpdateGps(time_budget);
        }
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("GP update time: {:f} ms.", dt);
        ERL_TRACY_PLOT("gp_update_time (ms)", dt);

        ERL_TRACY_PLOT("m_gp_map_.size()", static_cast<long>(m_gp_map_.size()));
        ERL_TRACY_PLOT("m_new_gp_keys_.size()", static_cast<long>(m_new_gp_keys_.size()));
        ERL_TRACY_PLOT("m_new_gp_queue_.size()", static_cast<long>(m_new_gp_queue_.size()));
        ERL_TRACY_PLOT("m_gps_to_train_.size()", static_cast<long>(m_gps_to_train_.size()));
        ERL_TRACY_PLOT_CONFIG("m_gp_map_.memory_usage", tracy::PlotFormatType::Memory, true, true, 0);
        ERL_TRACY_PLOT("m_gp_map_.memory_usage", static_cast<long>([&]() {
                           std::size_t gps_memory_usage = 0;
                           for (const auto &[key, gp]: m_gp_map_) {
                               gps_memory_usage += sizeof(key);
                               gps_memory_usage += sizeof(gp);
                               if (gp != nullptr) { gps_memory_usage += gp->GetMemoryUsage(); }
                           }
                           return gps_memory_usage;
                       }()));

        if (m_setting_->log_timing) {
            m_train_log_file_ << "," << dt;  // total_gp_update_time
            dt = std::chrono::duration<double, std::milli>(t1 - t_start).count();
            m_train_log_file_ << "," << dt << std::endl << std::flush;  // total_update_time
        }

        ERL_TRACY_FRAME_MARK_END();

        return success;
    }

    bool
    GpSdfMapping3D::Test(
        const Eigen::Ref<const Eigen::Matrix3Xd> &positions_in,
        Eigen::VectorXd &distances_out,
        Eigen::Matrix3Xd &gradients_out,
        Eigen::Matrix4Xd &variances_out,
        Eigen::Matrix6Xd &covariances_out) {

        {
            std::lock_guard lock(m_mutex_);
            if (m_gp_map_.empty()) {
                ERL_WARN("No GPs available for testing.");
                return false;
            }
        }

        if (positions_in.cols() == 0) {
            ERL_WARN("No query positions provided.");
            return false;
        }
        if (m_surface_mapping_ == nullptr) {
            ERL_WARN("Surface mapping is not initialized.");
            return false;
        }
        if (m_surface_mapping_->GetOctree() == nullptr) {
            ERL_WARN("Octree is not initialized.");
            return false;
        }
        if (!m_test_buffer_.ConnectBuffers(  // allocate memory for test results
                positions_in,
                distances_out,
                gradients_out,
                variances_out,
                covariances_out,
                m_setting_->test_query->compute_covariance)) {
            ERL_WARN("Failed to connect test buffers.");
            return false;
        }

        const std::chrono::high_resolution_clock::time_point t_start = std::chrono::high_resolution_clock::now();

        const uint32_t num_queries = positions_in.cols();
        const uint32_t num_threads = std::min(std::min(m_setting_->num_threads, std::thread::hardware_concurrency()), num_queries);
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = num_queries / num_threads;
        const std::size_t leftover = num_queries - batch_size * num_threads;
        std::size_t start_idx, end_idx;

        if (m_setting_->log_timing) {
            std::lock_guard lock(m_log_mutex_);
            m_test_log_file_ << m_travel_distance_;
        }

        std::chrono::high_resolution_clock::time_point t0, t1;
        double dt;
        {
            std::lock_guard lock(m_mutex_);  // CRITICAL SECTION
            {
                double x, y, z;
                m_surface_mapping_->GetOctree()->GetMetricMin(x, y, z);  // trigger the octree to update its metric min/max
            }

            // If we iterate through the octree for each query position separately, it takes too much CPU time.
            // Instead, we collect all GPs in the area of all query positions and then assign them to the query positions.
            // Some query positions may not have any GPs from m_candidate_gps_. We need to search for them separately.
            // We are sure that m_candidate_gps_ are not empty because m_gp_map_ is not empty.
            // Experiments show that this can reduce the search time by at most 50%.
            // For 15k crowded query positions, the search time is reduced from ~60ms to ~30ms.
            // Another important trick is to use a KdTree to search for candidate GPs for each query position.
            // Knn search is much faster and the result is sorted by distance.
            // Experiments show that this knn search can reduce the search time further to 2ms.
            SearchCandidateGps(positions_in);

            // Search GPs for each query position
            t0 = std::chrono::high_resolution_clock::now();
            m_query_to_gps_.clear();
            m_query_to_gps_.resize(num_queries);  // allocate memory for n threads, collected GPs will be locked for testing
            if (num_queries == 1) {
                SearchGpThread(0, 0, 1);  // save time on thread creation
            } else {
                start_idx = 0;
                for (uint32_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                    end_idx = start_idx + batch_size;
                    if (thread_idx < leftover) { end_idx++; }
                    threads.emplace_back(&GpSdfMapping3D::SearchGpThread, this, thread_idx, start_idx, end_idx);
                    start_idx = end_idx;
                }
                for (auto &thread: threads) { thread.join(); }
                threads.clear();
            }
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (m_setting_->log_timing) { m_test_log_file_ << "," << dt; }  // gp_search_time
            ERL_INFO("Search GPs: {:f} ms", dt);

            // Train any updated GPs
            if (!m_new_gp_queue_.empty()) {
                t0 = std::chrono::high_resolution_clock::now();
                std::unordered_set<std::shared_ptr<Gp>> new_gps;
                for (auto &gps: m_query_to_gps_) {
                    for (auto &[distance, gp]: gps) {
                        if (gp->active && !gp->gp->IsTrained()) { new_gps.insert(gp); }
                    }
                }
                m_gps_to_train_.clear();
                m_gps_to_train_.insert(m_gps_to_train_.end(), new_gps.begin(), new_gps.end());
                TrainGps();
                t1 = std::chrono::high_resolution_clock::now();
                dt = std::chrono::duration<double, std::micro>(t1 - t0).count();
                if (m_setting_->log_timing) { m_test_log_file_ << "," << dt; }  // gp_train_time
                ERL_INFO("Train GPs: {:f} us", dt);
            } else {
                if (m_setting_->log_timing) { m_test_log_file_ << ",0"; }
            }
        }

        // Compute the inference result for each query position
        m_query_used_gps_.clear();
        m_query_used_gps_.resize(num_queries);
        t0 = std::chrono::high_resolution_clock::now();
        if (num_queries == 1) {
            TestGpThread(0, 0, 1);  // save time on thread creation
        } else {
            start_idx = 0;
            for (uint32_t thread_idx = 0; thread_idx < num_threads; thread_idx++) {
                end_idx = start_idx + batch_size;
                if (thread_idx < leftover) { ++end_idx; }
                threads.emplace_back(&GpSdfMapping3D::TestGpThread, this, thread_idx, start_idx, end_idx);
                start_idx = end_idx;
            }
            for (auto &thread: threads) { thread.join(); }
            threads.clear();
        }
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (m_setting_->log_timing) { m_test_log_file_ << "," << dt; }  // gp_test_time
        ERL_INFO("Test GPs: {:f} ms", dt);

        m_test_buffer_.DisconnectBuffers();

        for (const auto &gps: m_query_to_gps_) {
            for (const auto &[distance, gp]: gps) { gp->locked_for_test = false; }  // unlock GPs
        }

        if (m_setting_->log_timing) {
            const std::chrono::high_resolution_clock::time_point t_end = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t_end - t_start).count();
            m_test_log_file_ << "," << dt << std::endl << std::flush;  // total_test_time
        }

        return true;
    }

    bool
    GpSdfMapping3D::operator==(const GpSdfMapping3D &other) const {
        if (m_setting_ == nullptr && other.m_setting_ != nullptr) { return false; }
        if (m_setting_ != nullptr && (other.m_setting_ == nullptr || *m_setting_ != *other.m_setting_)) { return false; }
        if (m_surface_mapping_ == nullptr && other.m_surface_mapping_ != nullptr) { return false; }
        if (m_surface_mapping_ != nullptr && (other.m_surface_mapping_ == nullptr || *m_surface_mapping_ != *other.m_surface_mapping_)) { return false; }
        if (m_gp_map_.size() != other.m_gp_map_.size()) { return false; }
        for (const auto &[key, gp]: m_gp_map_) {
            auto it = other.m_gp_map_.find(key);
            if (it == other.m_gp_map_.end()) { return false; }
            const auto &[other_key, other_gp] = *it;
            if (gp == nullptr && other_gp != nullptr) { return false; }
            if (gp != nullptr && (other_gp == nullptr || *gp != *other_gp)) { return false; }
        }
        if (m_new_gp_keys_.size() != other.m_new_gp_keys_.size()) { return false; }
        for (const auto &[key, handle]: m_new_gp_keys_) {
            const auto it = other.m_new_gp_keys_.find(key);
            if (it == other.m_new_gp_keys_.end()) { return false; }
            if ((*handle).time_stamp != (*it->second).time_stamp) { return false; }
        }
        if (m_train_gp_time_ != other.m_train_gp_time_) { return false; }
        if (m_travel_distance_ != other.m_travel_distance_) { return false; }
        if (m_last_position_ != other.m_last_position_) { return false; }
        return true;
    }

    bool
    GpSdfMapping3D::Write(const std::string &filename) const {
        ERL_INFO("Writing GpSdfMapping3D to file: {}", filename);
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

    static const std::string kFileHeader = "# erl::sdf_mapping::GpSdfMapping3D";

    bool
    GpSdfMapping3D::Write(std::ostream &s) const {
        s << kFileHeader << std::endl  //
          << "# (feel free to add / change comments, but leave the first line as it is!)" << std::endl
          << "setting" << std::endl;
        // write setting
        if (!m_setting_->Write(s)) {
            ERL_WARN("Failed to write setting.");
            return false;
        }
        // write data
        s << "surface_mapping " << (m_surface_mapping_ != nullptr) << std::endl;
        if (m_surface_mapping_ != nullptr && !m_surface_mapping_->Write(s)) {
            ERL_WARN("Failed to write surface mapping.");
            return false;
        }
        // m_clusters_to_update_ is temporary data.
        s << "gp_map " << m_gp_map_.size() << std::endl;
        for (const auto &[key, gp]: m_gp_map_) {
            s.write(reinterpret_cast<const char *>(&key[0]), sizeof(key[0]));
            s.write(reinterpret_cast<const char *>(&key[1]), sizeof(key[1]));
            s.write(reinterpret_cast<const char *>(&key[2]), sizeof(key[2]));
            const bool gp_exists = gp != nullptr;
            s.write(reinterpret_cast<const char *>(&gp_exists), sizeof(gp_exists));
            if (gp_exists && !gp->Write(s)) {
                ERL_WARN("Failed to write GP.");
                return false;
            }
        }
        // m_candidate_gps_ is temporary data.
        // m_kd_tree_candidate_gps_ is temporary data.
        // m_query_to_gps_ is temporary data.
        // m_query_used_gps_ is temporary data.
        s << "new_gp_keys " << m_new_gp_keys_.size() << std::endl;
        for (const auto &[key, handle]: m_new_gp_keys_) {
            s.write(reinterpret_cast<const char *>(&key[0]), sizeof(key[0]));
            s.write(reinterpret_cast<const char *>(&key[1]), sizeof(key[1]));
            s.write(reinterpret_cast<const char *>(&key[2]), sizeof(key[2]));
            s.write(reinterpret_cast<const char *>(&(*handle).time_stamp), sizeof((*handle).time_stamp));
        }
        // m_new_gp_queue_ can be reconstructed from m_new_gp_keys_.
        // m_gps_to_train_ is temporary data.
        s << "train_gp_time" << std::endl;
        s.write(reinterpret_cast<const char *>(&m_train_gp_time_), sizeof(m_train_gp_time_));
        s << "travel_distance" << std::endl;
        s.write(reinterpret_cast<const char *>(&m_travel_distance_), sizeof(m_travel_distance_));
        s << "last_position " << m_last_position_.has_value() << std::endl;
        if (m_last_position_.has_value()) {
            if (!common::SaveEigenMatrixToBinaryStream(s, m_last_position_.value())) {
                ERL_WARN("Failed to write last_position.");
                return false;
            }
        }
        // m_train_log_file_ is temporary data.
        // m_test_log_file_ is temporary data.
        s << "end_of_GpSdfMapping3D" << std::endl;
        return s.good();
    }

    bool
    GpSdfMapping3D::Read(const std::string &filename) {
        ERL_INFO("Reading GpSdfMapping3D from file: {}", std::filesystem::absolute(filename));
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
    GpSdfMapping3D::Read(std::istream &s) {
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
            "surface_mapping",
            "gp_map",
            "new_gp_keys",
            "train_gp_time",
            "travel_distance",
            "last_position",
            "end_of_GpSdfMapping3D",
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
                case 0: {         // setting
                    skip_line();  // skip the line to read the bindary data section
                    if (!m_setting_->Read(s)) {
                        ERL_WARN("Failed to read setting.");
                        return false;
                    }
                    break;
                }
                case 1: {  // surface_mapping
                    bool has_surface_mapping;
                    s >> has_surface_mapping;
                    skip_line();
                    if (has_surface_mapping) {
                        if (m_surface_mapping_ == nullptr) {
                            m_surface_mapping_ = geometry::AbstractSurfaceMapping::CreateSurfaceMapping<geometry::AbstractSurfaceMapping3D>(  //
                                m_setting_->surface_mapping_type,
                                m_setting_->surface_mapping);
                        }
                        if (!m_surface_mapping_->Read(s)) {
                            ERL_WARN("Failed to read surface mapping.");
                            return false;
                        }
                    }
                    break;
                }
                case 2: {  // gp_map
                    uint32_t num_gps;
                    s >> num_gps;
                    skip_line();
                    for (uint32_t i = 0; i < num_gps; ++i) {
                        geometry::OctreeKey key;
                        s.read(reinterpret_cast<char *>(&key[0]), sizeof(key[0]));
                        s.read(reinterpret_cast<char *>(&key[1]), sizeof(key[1]));
                        s.read(reinterpret_cast<char *>(&key[2]), sizeof(key[2]));
                        auto [it, inserted] = m_gp_map_.try_emplace(key, nullptr);
                        if (!inserted) {
                            ERL_WARN("Duplicate GP key: ({}, {}).", key[0], key[1]);
                            return false;
                        }
                        bool has_gp;
                        s.read(reinterpret_cast<char *>(&has_gp), sizeof(has_gp));
                        if (has_gp) {
                            it->second = std::make_shared<Gp>();
                            if (!it->second->Read(s, m_setting_->gp_sdf)) {
                                ERL_WARN("Failed to read GP.");
                                return false;
                            }
                        }
                    }
                    break;
                }
                case 3: {  // new_gp_keys
                    uint32_t num_new_gp_keys;
                    s >> num_new_gp_keys;
                    skip_line();
                    for (uint32_t i = 0; i < num_new_gp_keys; ++i) {
                        geometry::OctreeKey key;
                        s.read(reinterpret_cast<char *>(&key[0]), sizeof(key[0]));
                        s.read(reinterpret_cast<char *>(&key[1]), sizeof(key[1]));
                        s.read(reinterpret_cast<char *>(&key[2]), sizeof(key[2]));
                        long time_stamp;
                        s.read(reinterpret_cast<char *>(&time_stamp), sizeof(time_stamp));
                        m_new_gp_keys_.insert({key, m_new_gp_queue_.push({time_stamp, key})});
                    }
                    break;
                }
                case 4: {  // train_gp_time
                    skip_line();
                    s.read(reinterpret_cast<char *>(&m_train_gp_time_), sizeof(m_train_gp_time_));
                    break;
                }
                case 5: {  // travel_distance
                    skip_line();
                    s.read(reinterpret_cast<char *>(&m_travel_distance_), sizeof(m_travel_distance_));
                    break;
                }
                case 6: {  // last_position
                    bool has_last_position;
                    s >> has_last_position;
                    skip_line();
                    if (has_last_position) {
                        if (!common::LoadEigenMatrixFromBinaryStream(s, m_last_position_.emplace())) {
                            ERL_WARN("Failed to read last_position.");
                            return false;
                        }
                    }
                    break;
                }
                case 7: {  // end_of_GpSdfMapping3D
                    skip_line();
                    return true;
                }
                default: {  // should not reach here
                    ERL_FATAL("Internal error, should not reach here.");
                }
            }
            ++token_idx;
        }
        ERL_WARN("Failed to read GpSdfMapping3D. Truncated file?");
        return false;  // should not reach here
    }

    void
    GpSdfMapping3D::UpdateGps(double time_budget) {
        ERL_DEBUG_ASSERT(m_setting_->gp_sdf_area_scale > 1, "GP area scale must be greater than 1");

        // add affected clusters
        auto t0 = std::chrono::high_resolution_clock::now();
        auto t_start = t0;
        const geometry::OctreeKeySet changed_clusters = m_surface_mapping_->GetChangedClusters();
        const uint32_t cluster_level = m_surface_mapping_->GetClusterLevel();
        const std::shared_ptr<geometry::SurfaceMappingOctree> octree = m_surface_mapping_->GetOctree();
        const uint32_t cluster_depth = octree->GetTreeDepth() - cluster_level;
        const double cluster_size = octree->GetNodeSize(cluster_depth);
        const double area_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2;
        geometry::OctreeKeySet affected_clusters(changed_clusters);
        for (const auto &cluster_key: changed_clusters) {
            for (auto it = octree->BeginTreeInAabb(geometry::Aabb3D(octree->KeyToCoord(cluster_key, cluster_depth), area_half_size), cluster_depth),
                      end = octree->EndTreeInAabb();
                 it != end;
                 ++it) {
                if (it->GetDepth() != cluster_depth) { continue; }
                affected_clusters.insert(octree->AdjustKeyToDepth(it.GetKey(), cluster_depth));
            }
        }
        m_clusters_to_update_.clear();
        m_clusters_to_update_.insert(m_clusters_to_update_.end(), affected_clusters.begin(), affected_clusters.end());
        auto t1 = std::chrono::high_resolution_clock::now();
        auto dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Collecte {} -> {} affected clusters: {} ms.", changed_clusters.size(), affected_clusters.size(), dt);

        // update GPs in affected clusters
        /// create GPs for new clusters, compute AABB of all clusters to be updated
        t0 = std::chrono::high_resolution_clock::now();
        long cnt_new_gps = 0;
        for (auto &cluster_key: m_clusters_to_update_) {
            auto [it, inserted] = m_gp_map_.try_emplace(cluster_key, nullptr);
            if (!inserted) { continue; }
            ++cnt_new_gps;
            it->second = std::make_shared<Gp>();
            octree->KeyToCoord(cluster_key, cluster_depth, it->second->position);
            it->second->half_size = area_half_size;
        }
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("{} new GP(s) created, {} ms.", cnt_new_gps, dt);

        /// create threads to update GPs
        t0 = std::chrono::high_resolution_clock::now();
        uint32_t num_threads = std::min(m_setting_->num_threads, std::thread::hardware_concurrency());
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        const std::size_t batch_size = m_clusters_to_update_.size() / num_threads;
        const std::size_t left_over = m_clusters_to_update_.size() - batch_size * num_threads;
        std::size_t end_idx = 0;
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) {
            std::size_t start_idx = end_idx;
            end_idx = start_idx + batch_size;
            if (thread_idx < left_over) { end_idx++; }
            threads.emplace_back(&GpSdfMapping3D::UpdateGpThread, this, thread_idx, start_idx, end_idx);
        }
        for (uint32_t thread_idx = 0; thread_idx < num_threads; ++thread_idx) { threads[thread_idx].join(); }
        threads.clear();
        t1 = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("Update GPs' training data: {:f} ms, {} GPs.", dt, m_clusters_to_update_.size());

        if (!m_setting_->train_gp_immediately) {
            t0 = std::chrono::high_resolution_clock::now();
            for (auto &cluster_key: m_clusters_to_update_) {
                auto it = m_gp_map_.find(cluster_key);
                if (it == m_gp_map_.end() || !it->second->active) { continue; }  // GP does not exist or deactivated (e.g. due to no training data)
                if (it->second->gp != nullptr && !it->second->gp->IsTrained()) {
                    auto time_stamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
                    if (m_new_gp_keys_.find(cluster_key) == m_new_gp_keys_.end()) {
                        m_new_gp_keys_.insert({cluster_key, m_new_gp_queue_.push({time_stamp, cluster_key})});
                    } else {
                        auto &heap_key = m_new_gp_keys_.at(cluster_key);
                        (*heap_key).time_stamp = time_stamp;
                        m_new_gp_queue_.increase(heap_key);
                    }
                }
            }
            t1 = std::chrono::high_resolution_clock::now();
            dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
            ERL_INFO("Collect GPs to train: {} ms.", dt);

            dt = std::chrono::duration<double, std::milli>(t1 - t_start).count();
            time_budget -= dt * 1e3;  // us

            // train as many new GPs as possible within the time limit
            if (time_budget > m_train_gp_time_) {
                auto max_num_gps_to_train = static_cast<std::size_t>(std::floor(time_budget / m_train_gp_time_));
                max_num_gps_to_train = std::min(max_num_gps_to_train, m_new_gp_queue_.size());
                m_gps_to_train_.clear();
                while (!m_new_gp_queue_.empty() && m_gps_to_train_.size() < max_num_gps_to_train) {
                    geometry::OctreeKey cluster_key = m_new_gp_queue_.top().key;
                    m_new_gp_queue_.pop();
                    m_new_gp_keys_.erase(cluster_key);
                    auto it = m_gp_map_.find(cluster_key);
                    auto gp = it->second;
                    if (it == m_gp_map_.end() || !gp->active) { continue; }  // GP does not exist or deactivated (e.g. due to no training data)
                    if (gp->gp != nullptr && !gp->gp->IsTrained()) { m_gps_to_train_.push_back(gp); }
                }
                TrainGps();
            }
            ERL_INFO("{} GP(s) not trained yet due to time limit.", m_new_gp_queue_.size());
        }

        dt = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() - t_start).count();
        m_train_log_file_ << "," << dt << "," << m_new_gp_queue_.size() << "," << m_train_gp_time_;
    }

    void
    GpSdfMapping3D::UpdateGpThread(const uint32_t thread_idx, const std::size_t start_idx, const std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        const auto octree = m_surface_mapping_->GetOctree();
        if (octree == nullptr) { return; }
        const double sensor_noise = m_surface_mapping_->GetSensorNoise();
        const uint32_t cluster_depth = octree->GetTreeDepth() - m_surface_mapping_->GetClusterLevel();
        const double cluster_size = octree->GetNodeSize(cluster_depth);
        const double aabb_half_size = cluster_size * m_setting_->gp_sdf_area_scale / 2.;

        std::vector<std::pair<double, std::shared_ptr<SurfaceData>>> surface_data_vec;
        const auto max_num_samples = static_cast<std::size_t>(m_setting_->gp_sdf->max_num_samples);
        surface_data_vec.reserve(max_num_samples);
        for (uint32_t i = start_idx; i < end_idx; ++i) {
            auto &cluster_key = m_clusters_to_update_[i];
            // get the GP of the cluster
            std::shared_ptr<Gp> &gp = m_gp_map_.at(cluster_key);
            // testing thread may unlock the GP, but it is impossible to lock it here due to the mutex
            if (gp->locked_for_test) {  // create a new GP if the old one is locked for testing
                const auto new_gp = std::make_shared<Gp>();
                new_gp->position = gp->position;
                new_gp->half_size = gp->half_size;
                gp = new_gp;
            }
            gp->active = true;  // activate the GP
            if (gp->gp == nullptr) { gp->gp = std::make_shared<LogSdfGaussianProcess>(m_setting_->gp_sdf); }

            // collect surface data in the area
            surface_data_vec.clear();
            for (auto it = octree->BeginLeafInAabb(geometry::Aabb3D(octree->KeyToCoord(cluster_key, cluster_depth), aabb_half_size)),
                      end = octree->EndLeafInAabb();
                 it != end;
                 ++it) {
                auto surface_data = it->GetSurfaceData();
                if (surface_data == nullptr) { continue; }  // no surface data in the node
                surface_data_vec.emplace_back((gp->position - surface_data->position).norm(), surface_data);
            }
            if (surface_data_vec.empty()) {  // no surface data in the area
                gp->active = false;          // deactivate the GP if there is no training data
                gp->num_train_samples = 0;
                continue;
            }
            std::sort(surface_data_vec.begin(), surface_data_vec.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
            if (surface_data_vec.size() > static_cast<std::size_t>(max_num_samples)) { surface_data_vec.resize(max_num_samples); }

            // prepare data for GP training
            gp->gp->Reset(static_cast<long>(surface_data_vec.size()), 3);
            Eigen::MatrixXd &mat_x = gp->gp->GetTrainInputSamplesBuffer();
            Eigen::VectorXd &vec_y = gp->gp->GetTrainOutputSamplesBuffer();
            Eigen::MatrixXd &mat_grad = gp->gp->GetTrainOutputGradientSamplesBuffer();
            Eigen::VectorXd &vec_var_x = gp->gp->GetTrainInputSamplesVarianceBuffer();
            Eigen::VectorXd &vec_var_y = gp->gp->GetTrainOutputValueSamplesVarianceBuffer();
            Eigen::VectorXd &vec_var_grad = gp->gp->GetTrainOutputGradientSamplesVarianceBuffer();
            Eigen::VectorXl &vec_grad_flag = gp->gp->GetTrainGradientFlagsBuffer();
            long count = 0;
            for (auto &[distance, surface_data]: surface_data_vec) {
                mat_x.col(count) = surface_data->position;
                vec_y[count] = m_setting_->offset_distance;
                vec_var_y[count] = sensor_noise;
                vec_var_grad[count] = surface_data->var_normal;
                if ((surface_data->var_normal > m_setting_->max_valid_gradient_var) ||  // invalid gradient
                    (surface_data->normal.norm() < 0.9)) {                              // invalid normal
                    vec_var_x[count] = m_setting_->invalid_position_var;                // position is unreliable
                    vec_grad_flag[count] = false;
                    mat_grad.col(count++).setZero();
                    continue;
                }
                vec_var_x[count] = surface_data->var_position;
                vec_grad_flag[count] = true;
                mat_grad.col(count++) = surface_data->normal;
                if (count >= mat_x.cols()) { break; }  // reached max_num_samples
            }
            gp->num_train_samples = count;
            if (m_setting_->train_gp_immediately) { gp->Train(); }
        }
    }

    void
    GpSdfMapping3D::TrainGps() {
        const uint32_t n = m_gps_to_train_.size();
        if (n == 0) { return; }

        ERL_INFO("Training {} GPs ...", n);
        const auto t0 = std::chrono::high_resolution_clock::now();
#pragma omp parallel for default(none) shared(n)
        for (uint32_t i = 0; i < n; ++i) { m_gps_to_train_[i]->Train(); }
        m_gps_to_train_.clear();
        const auto t1 = std::chrono::high_resolution_clock::now();
        double time = std::chrono::duration<double, std::micro>(t1 - t0).count();
        ERL_INFO("Training GPs time: {} ms.", time / 1e3);
        time /= static_cast<double>(n);
        m_train_gp_time_ = m_train_gp_time_ * 0.1 + time * 0.9;
        ERL_INFO("Per GP training time: {:f} us.", m_train_gp_time_);
    }

    void
    GpSdfMapping3D::SearchCandidateGps(const Eigen::Ref<const Eigen::Matrix3Xd> &positions_in) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const double search_area_half_size = m_setting_->test_query->search_area_half_size;
        const auto octree = m_surface_mapping_->GetOctree();
        const uint32_t cluster_depth = octree->GetTreeDepth() - m_surface_mapping_->GetClusterLevel();
        const double cluster_half_size = octree->GetNodeSize(cluster_depth) / 2;
        const geometry::Aabb3D octree_aabb = octree->GetMetricAabb();
        geometry::Aabb3D query_aabb(
            positions_in.rowwise().minCoeff().array() - search_area_half_size,
            positions_in.rowwise().maxCoeff().array() + search_area_half_size);
        geometry::Aabb3D intersection = query_aabb.Intersection(octree_aabb);
        m_candidate_gps_.clear();
        while (intersection.sizes().prod() > 0) {
            // search the octree for clusters in the search area
            for (auto it = octree->BeginTreeInAabb(intersection, cluster_depth), end = octree->EndTreeInAabb(); it != end; ++it) {
                if (it->GetDepth() != cluster_depth) { continue; }  // not a cluster node
                auto it_gp = m_gp_map_.find(it.GetIndexKey());      // get the GP of the cluster
                if (it_gp == m_gp_map_.end()) { continue; }         // no gp for this cluster
                const auto gp = it_gp->second;                      // get the GP of the cluster
                if (!gp->active) { continue; }                      // gp is inactive (e.g. due to no training data)
                gp->locked_for_test = true;                         // lock the GP for testing
                m_candidate_gps_.emplace_back(geometry::Aabb3D(gp->position, cluster_half_size), gp);
            }
            if (!m_candidate_gps_.empty()) { break; }  // found at least one GP
            // double the size of query_aabb
            query_aabb = geometry::Aabb3D(query_aabb.center, 2.0 * query_aabb.half_sizes);
            geometry::Aabb3D new_intersection = query_aabb.Intersection(octree_aabb);
            if ((intersection.min() == new_intersection.min()) && (intersection.max() == new_intersection.max())) { break; }  // intersection did not change
            intersection = std::move(new_intersection);
        }
        // build kdtree of candidate GPs
        if (!m_candidate_gps_.empty()) {
            Eigen::Matrix3Xd positions(3, m_candidate_gps_.size());
            for (std::size_t i = 0; i < m_candidate_gps_.size(); ++i) { positions.col(static_cast<long>(i)) = m_candidate_gps_[i].second->position; }
            m_kd_tree_candidate_gps_ = std::make_shared<geometry::KdTree3d>(std::move(positions));
        }
        const auto t1 = std::chrono::high_resolution_clock::now();
        const double dt = std::chrono::duration<double, std::milli>(t1 - t0).count();
        ERL_INFO("{} candidate GPs found.", m_candidate_gps_.size());
        ERL_INFO("Search candidate GPs: {:f} ms", dt);
    }

    void
    GpSdfMapping3D::SearchGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        if (m_surface_mapping_ == nullptr) { return; }
        const auto octree = m_surface_mapping_->GetOctree();
        const uint32_t cluster_level = m_surface_mapping_->GetClusterLevel();
        const uint32_t cluster_depth = octree->GetTreeDepth() - cluster_level;
        const geometry::Aabb3D octree_aabb = octree->GetMetricAabb();

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            const Eigen::Vector3d test_position = m_test_buffer_.positions->col(i);
            std::vector<std::pair<double, std::shared_ptr<Gp>>> &gps = m_query_to_gps_[i];
            gps.clear();
            gps.reserve(16);

            double search_area_half_size = m_setting_->test_query->search_area_half_size;
            geometry::Aabb3D search_aabb(test_position, search_area_half_size);
            geometry::Aabb3D intersection = octree_aabb.Intersection(search_aabb);

            // search gps from the common buffer at first to save time
            // use kdtree to search for 32 nearest GPs
            if (m_kd_tree_candidate_gps_ != nullptr) {
                Eigen::VectorXl indicies = Eigen::VectorXl::Constant(32, -1);
                Eigen::VectorXd squared_distances(32);
                m_kd_tree_candidate_gps_->Knn(32, test_position, indicies, squared_distances);
                for (long j = 0; j < 32; ++j) {
                    const long &index = indicies[j];
                    if (index < 0) { break; }  // no more GPs
                    const auto &[cluster_aabb, gp] = m_candidate_gps_[index];
                    if (!cluster_aabb.intersects(intersection)) { continue; }
                    gps.emplace_back(std::sqrt(squared_distances[j]), gp);
                    if (gps.size() >= 16) { break; }  // found enough GPs
                }
            } else {
                // no kdtree, search all GPs in the common buffer, which is slow
                gps.reserve(m_candidate_gps_.size());  // request more memory
                for (const auto &[cluster_aabb, gp]: m_candidate_gps_) {
                    if (!cluster_aabb.intersects(intersection)) { continue; }
                    gps.emplace_back((gp->position - test_position).norm(), gp);
                }
            }

            if (gps.empty()) {                    // no gp found
                if (!m_candidate_gps_.empty()) {  // common buffer is not empty
                    search_area_half_size *= 2;   // double search area size
                    search_aabb = geometry::Aabb3D(test_position, search_area_half_size);
                    geometry::Aabb3D new_intersection = octree_aabb.Intersection(search_aabb);
                    // intersection did not change, no need to search again
                    if ((intersection.min() == new_intersection.min()) && (intersection.max() == new_intersection.max())) { continue; }
                    // update intersection
                    intersection = std::move(new_intersection);
                }
            } else {
                continue;  // found at least one gp
            }

            while (gps.empty() && intersection.sizes().prod() > 0) {
                // search the octree for clusters in the search area
                for (auto it = octree->BeginTreeInAabb(intersection, cluster_depth), end = octree->EndTreeInAabb(); it != end; ++it) {
                    if (it->GetDepth() != cluster_depth) { continue; }  // not a cluster node
                    auto it_gp = m_gp_map_.find(it.GetIndexKey());
                    if (it_gp == m_gp_map_.end()) { continue; }  // no gp for this cluster
                    const auto gp = it_gp->second;
                    if (!gp->active) { continue; }  // gp is inactive (e.g. due to no training data)
                    gp->locked_for_test = true;     // lock the GP for testing
                    gps.emplace_back((gp->position - test_position).norm(), gp);
                }
                if (!gps.empty()) { break; }  // found at least one gp

                search_area_half_size *= 2;  // double search area size
                search_aabb = geometry::Aabb3D(test_position, search_area_half_size);
                geometry::Aabb3D new_intersection = octree_aabb.Intersection(search_aabb);
                // intersection did not change, no need to search again
                if ((intersection.min() == new_intersection.min()) && (intersection.max() == new_intersection.max())) { break; }
                // update intersection
                intersection = std::move(new_intersection);
            }
        }
    }

    void
    GpSdfMapping3D::TestGpThread(uint32_t thread_idx, std::size_t start_idx, std::size_t end_idx) {
        ERL_TRACY_SET_THREAD_NAME(fmt::format("{}:{}", __PRETTY_FUNCTION__, thread_idx).c_str());
        (void) thread_idx;

        if (m_surface_mapping_ == nullptr) { return; }
        std::vector<std::size_t> gp_indices;
        constexpr int max_tries = 8;                // we ask at most 4 GPs for each query position
        Eigen::Matrix4Xd fs(4, max_tries);          // f, fGrad1, fGrad2, fGrad3
        Eigen::Matrix4Xd variances(4, max_tries);   // variances of f, fGrad1, fGrad2, fGrad3
        Eigen::Matrix6Xd covariance(6, max_tries);  // covariances of (fGrad1,f), (fGrad2,f), (fGrad2, fGrad1), (fGrad3, f), (fGrad3, fGrad1), (fGrad3, fGrad2)
        Eigen::MatrixXd no_variance;
        Eigen::MatrixXd no_covariance;
        std::vector<std::pair<long, long>> tested_idx;
        tested_idx.reserve(max_tries);

        const bool recompute_variance = m_setting_->test_query->recompute_variance;
        const bool compute_covariance = m_setting_->test_query->compute_covariance;
        const bool use_nearest_only = m_setting_->test_query->use_nearest_only;
        const double max_test_valid_distance_var = m_setting_->test_query->max_test_valid_distance_var;
        const double softmax_temperature = m_setting_->test_query->softmax_temperature;
        const double offset_distance = m_setting_->offset_distance;

        for (uint32_t i = start_idx; i < end_idx; ++i) {
            double &distance_out = (*m_test_buffer_.distances)[i];
            auto gradient_out = m_test_buffer_.gradients->col(i);
            auto variance_out = m_test_buffer_.variances->col(i);

            distance_out = 0.;
            gradient_out.setZero();

            variances.setConstant(1e6);
            if (compute_covariance) { covariance.setConstant(1e6); }

            auto &gps = m_query_to_gps_[i];
            if (gps.empty()) { continue; }

            const Eigen::Vector3d test_position = m_test_buffer_.positions->col(i);
            gp_indices.resize(gps.size());
            std::iota(gp_indices.begin(), gp_indices.end(), 0);
            if (gps.size() > 1) {
                std::stable_sort(gp_indices.begin(), gp_indices.end(), [&gps](const size_t i1, const size_t i2) { return gps[i1].first < gps[i2].first; });
            }

            tested_idx.clear();
            bool need_weighted_sum = false;
            long cnt = 0;
            for (std::size_t &j: gp_indices) {
                // call selected GPs for inference
                Eigen::Ref<Eigen::Vector4d> f = fs.col(cnt);           // distance, gradient_x, gradient_y, gradient_z
                Eigen::Ref<Eigen::VectorXd> var = variances.col(cnt);  // var_distance, var_gradient_x, var_gradient_y, var_gradient_z
                auto &gp = gps[j].second->gp;

                if (recompute_variance) {
                    if (compute_covariance) {
                        gp->Test(test_position, f, no_variance, covariance.col(cnt));
                    } else {
                        gp->Test(test_position, f, no_variance, no_covariance);
                    }
                    Eigen::Vector3d grad(f[1], f[2], f[3]);
                    if (grad.norm() < 1.e-15) { continue; }  // invalid gradient, skip this GP
                    grad.normalize();
                    double grad_azimuth, grad_elevation;
                    common::DirectionToAzimuthElevation(grad, grad_azimuth, grad_elevation);

                    auto &mat_x = gp->GetTrainInputSamplesBuffer();
                    auto &vec_x_var = gp->GetTrainInputSamplesVarianceBuffer();
                    const Eigen::Vector3d predicted_surf_position = test_position - grad * f[0];
                    long num_samples = gp->GetNumTrainSamples();
                    Eigen::VectorXd weight(num_samples);
                    double weight_sum = 0;
                    double &var_sdf = var[0];
                    double &var_grad_x = var[1];
                    double &var_grad_y = var[2];
                    double &var_grad_z = var[3];
                    var_sdf = 0;
                    var_grad_x = 0;
                    var_grad_y = 0;
                    var_grad_z = 0;
                    double var_azimuth = 0;
                    double var_elevation = 0;
                    for (long k = 0; k < num_samples; ++k) {
                        // difference between training surface position and predicted surface position
                        const double d = (mat_x.col(k) - predicted_surf_position).norm();
                        weight[k] = std::max(1.e-6, std::exp(-d * softmax_temperature));
                        weight_sum += weight[k];

                        var_sdf += weight[k] * vec_x_var[k];

                        Eigen::Vector3d v = test_position - mat_x.col(k);
                        v.normalize();                             // warning: unchanged if v's norm is very small
                        if (v.squaredNorm() < 0.81) { continue; }  // invalid gradient, skip this sample
                        double v_azimuth, v_elevation;
                        common::DirectionToAzimuthElevation(v, v_azimuth, v_elevation);
                        const double diff_azimuth = std::fabs(v_azimuth - grad_azimuth);
                        const double diff_elevation = std::fabs(v_elevation - grad_elevation);
                        var_azimuth += weight[k] * diff_azimuth * diff_azimuth;
                        var_elevation += weight[k] * diff_elevation * diff_elevation;
                    }

                    var_sdf /= weight_sum;
                    var_azimuth /= weight_sum;
                    var_elevation /= weight_sum;

                    const double sin_azimuth = std::sin(grad_azimuth);
                    const double cos_azimuth = std::cos(grad_azimuth);
                    const double sin_elevation = std::sin(grad_elevation);
                    const double cos_elevation = std::cos(grad_elevation);
                    const double std_azimuth = std::sqrt(var_azimuth);
                    const double std_elevation = std::sqrt(var_elevation);
                    var_grad_x = sin_azimuth * cos_elevation * std_azimuth + cos_azimuth * sin_elevation * std_elevation;
                    var_grad_x *= var_grad_x;
                    var_grad_y = cos_azimuth * cos_elevation * std_azimuth - sin_azimuth * sin_elevation * std_elevation;
                    var_grad_y *= var_grad_y;
                    var_grad_z = cos_elevation * std_elevation;
                    var_grad_z *= var_grad_z;
                } else {
                    if (compute_covariance) {
                        gp->Test(test_position, f, var, covariance.col(cnt));
                    } else {
                        gp->Test(test_position, f, var, no_covariance);
                    }
                    if (f.tail<3>().norm() < 1.e-15) { continue; }  // invalid gradient, skip this GP
                }

                tested_idx.emplace_back(cnt++, j);
                if (use_nearest_only) { break; }
                if ((!need_weighted_sum) && (gp_indices.size() > 1) && (var[0] > max_test_valid_distance_var)) { need_weighted_sum = true; }
                if ((!need_weighted_sum) || (cnt >= max_tries)) { break; }
            }

            // sort the results by distance variance
            if (tested_idx.size() > 1 && need_weighted_sum) {
                std::stable_sort(tested_idx.begin(), tested_idx.end(), [&](auto a, auto b) -> bool { return variances(0, a.first) < variances(0, b.first); });
                // the first two results have different signs, pick the one with smaller variance
                if (fs(0, tested_idx[0].first) * fs(0, tested_idx[1].first) < 0) { need_weighted_sum = false; }
            }

            // store the result
            if (need_weighted_sum) {
                if (m_test_buffer_.Size() == 1) { ERL_DEBUG("SDF1: {:f}, SDF2: {:f}", fs(0, tested_idx[0].first), fs(0, tested_idx[1].first)); }

                if (variances(0, tested_idx[0].first) < max_test_valid_distance_var) {
                    auto j = tested_idx[0].first;
                    // column j is the result
                    distance_out = fs(0, j);
                    gradient_out << fs.col(j).tail<3>();
                    variance_out << variances.col(j);
                    if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariance.col(j); }
                    m_query_used_gps_[i][0] = gps[tested_idx[0].second].second;
                    m_query_used_gps_[i][1] = nullptr;
                } else {
                    // pick the best <= 4 results to do weighted sum
                    const std::size_t m = std::min(tested_idx.size(), 4ul);
                    double w_sum = 0;
                    Eigen::Vector4d f = Eigen::Vector4d::Zero();
                    Eigen::Vector4d variance_f = Eigen::Vector4d::Zero();
                    Eigen::Vector6d covariance_f = Eigen::Vector6d::Zero();
                    for (std::size_t k = 0; k < m; ++k) {
                        const long jk = tested_idx[k].first;
                        const double w = 1.0 / (variances(0, jk) - max_test_valid_distance_var);
                        w_sum += w;
                        f += fs.col(jk) * w;
                        variance_f += variances.col(jk) * w;
                        m_query_used_gps_[i][k] = gps[tested_idx[k].second].second;
                        if (compute_covariance) { covariance_f += covariance.col(jk) * w; }
                    }
                    f /= w_sum;
                    distance_out = f[0];
                    gradient_out << f.tail<3>();
                    variance_out = variance_f / w_sum;
                    if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariance_f / w_sum; }
                }
            } else {
                if (m_test_buffer_.Size() == 1) { ERL_DEBUG("SDF1: {}", fs(0, tested_idx[0].first)); }

                // the first column is the result
                distance_out = fs(0, 0);
                gradient_out << fs.col(0).tail<3>();
                variance_out << variances.col(0);
                if (compute_covariance) { m_test_buffer_.covariances->col(i) = covariance.col(0); }
                m_query_used_gps_[i][0] = gps[tested_idx[0].second].second;
                m_query_used_gps_[i][1] = nullptr;
            }

            distance_out -= offset_distance;
            gradient_out.normalize();
        }
    }

}  // namespace erl::sdf_mapping
