#include "erl_common/eigen.hpp"
#include "erl_common/opencv.hpp"

#include <open3d/geometry/VoxelGrid.h>

template<typename Dtype>
cv::Mat
ConvertMatrixToImage(const Eigen::MatrixX<Dtype> &mat, bool colorize) {
    cv::Mat img;
    Eigen::MatrixX<Dtype> mat_clean = Eigen::MatrixX<Dtype>::Zero(mat.rows(), mat.cols());

    const Dtype *src_ptr = mat.data();
    Dtype *dst_ptr = mat_clean.data();
    for (long i = 0; i < mat.size(); ++i) {
        if (!std::isfinite(src_ptr[i])) { continue; }
        dst_ptr[i] = src_ptr[i];
    }

    cv::eigen2cv(mat_clean, img);
    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    img.convertTo(img, CV_8UC1);

    if (colorize) {
        cv::applyColorMap(img, img, cv::COLORMAP_JET);
    } else {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    return img;
}

template<typename Dtype, int Order = Eigen::ColMajor>
cv::Mat
ConvertVectorToImage(int xs, int ys, const Eigen::VectorX<Dtype> &vec, bool colorize) {
    cv::Mat img;  // x: down, y: right
    if (Order == Eigen::ColMajor) {
        img = cv::Mat(
            ys,  // rows
            xs,  // cols
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            const_cast<Dtype *>(vec.data()));
        img = img.t();
    } else {
        img = cv::Mat(
            xs,
            ys,
            sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
            const_cast<Dtype *>(vec.data()));
    }

    cv::normalize(img, img, 0, 255, cv::NORM_MINMAX);
    img.convertTo(img, CV_8UC1);
    if (colorize) {
        cv::applyColorMap(img, img, cv::COLORMAP_JET);
    } else {
        cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    }
    return img;
}

template<typename Dtype, int Order = Eigen::ColMajor>
void
ConvertToVoxelGrid(
    const cv::Mat &color_img,
    const Eigen::Matrix3X<Dtype> &positions,
    const std::shared_ptr<open3d::geometry::VoxelGrid> &voxel_grid) {
    voxel_grid->voxels_.clear();
    for (int j = 0; j < color_img.cols; ++j) {  // column major
        for (int i = 0; i < color_img.rows; ++i) {
            Eigen::Vector3d position;
            if (Order == Eigen::ColMajor) {
                position = positions.col(i + j * color_img.rows).template cast<double>();
            } else {
                position = positions.col(j + i * color_img.cols).template cast<double>();
            }
            const auto &color = color_img.at<cv::Vec3b>(i, j);
            voxel_grid->AddVoxel(
                {voxel_grid->GetVoxel(position),
                 Eigen::Vector3d(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0)});
        }
    }
}

template<typename Dtype>
std::pair<cv::Mat, cv::Mat>
ConvertSdfToImage(Eigen::VectorX<Dtype> &distances, const int xs, const int ys) {
    cv::Mat img_sdf(ys, xs, sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1, distances.data());
    img_sdf = img_sdf.t();
    cv::normalize(img_sdf, img_sdf, 0, 255, cv::NORM_MINMAX, CV_8UC1);
    cv::applyColorMap(img_sdf, img_sdf, cv::COLORMAP_JET);

    Eigen::MatrixX8U sdf_sign = (distances.array() >= 0.0f).template cast<uint8_t>() * 255;
    cv::Mat img_sdf_sign(ys, xs, CV_8UC1, sdf_sign.data());
    img_sdf_sign = img_sdf_sign.t();

    // // for a zero pixel in img_sdf_sign fill the pixel in img_sdf with zero
    // const cv::Mat mask = img_sdf_sign == 0;
    // img_sdf.setTo(0, mask);
    return {std::move(img_sdf), std::move(img_sdf_sign)};
}

// template<typename Dtype>
// cv::Mat
// ConvertToImage(const Eigen::VectorX<Dtype> &sign, int xs, int ys, bool colorize) {
//     cv::Mat sign_image(
//         ys,
//         xs,
//         sizeof(Dtype) == 4 ? CV_32FC1 : CV_64FC1,
//         const_cast<Dtype *>(sign.data()));
//     sign_image = sign_image.t();
//     cv::normalize(sign_image, sign_image, 0, 255, cv::NORM_MINMAX);
//     sign_image.convertTo(sign_image, CV_8UC1);
//     if (colorize) { cv::applyColorMap(sign_image, sign_image, cv::COLORMAP_JET); }
//     return sign_image;
// }

// template<typename Dtype>
// cv::Mat
// ConvertToImage(Eigen::MatrixX<Dtype> ranges, bool is_lidar) {
//     for (long i = 0; i < ranges.size(); ++i) {
//         Dtype &range = ranges.data()[i];
//         if (range < 0.0 || range > 1000.0) { range = 0.0; }
//     }
//     cv::Mat ranges_img;
//     if (is_lidar) {
//         cv::eigen2cv(Eigen::MatrixX<Dtype>(ranges.transpose()), ranges_img);
//         cv::flip(ranges_img, ranges_img, 0);
//         cv::resize(ranges_img, ranges_img, {0, 0}, 2, 2);
//     } else {
//         cv::eigen2cv(ranges, ranges_img);
//     }
//     cv::normalize(ranges_img, ranges_img, 0, 255, cv::NORM_MINMAX);
//     ranges_img.convertTo(ranges_img, CV_8UC1);
//     cv::applyColorMap(ranges_img, ranges_img, cv::COLORMAP_JET);
//     return ranges_img;
// }

// template<typename Dtype>
// void
// ConvertSdfToVoxelGrid(
//     const cv::Mat &img_sdf,
//     const Eigen::Matrix3X<Dtype> &positions,
//     const std::shared_ptr<open3d::geometry::VoxelGrid> &voxel_grid_sdf) {
//     voxel_grid_sdf->voxels_.clear();
//     for (int j = 0; j < img_sdf.cols; ++j) {  // column major
//         for (int i = 0; i < img_sdf.rows; ++i) {
//             Eigen::Vector3d position = positions.col(i + j * img_sdf.rows).template
//             cast<double>(); const auto &color = img_sdf.at<cv::Vec3b>(i, j);
//             voxel_grid_sdf->AddVoxel(
//                 {voxel_grid_sdf->GetVoxel(position),
//                  Eigen::Vector3d(color[2] / 255.0, color[1] / 255.0, color[0] / 255.0)});
//         }
//     }
// }
