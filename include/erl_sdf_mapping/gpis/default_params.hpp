#pragma once

#include <cmath>
#include "erl_gaussian_process/mapping.hpp"

namespace erl::sdf_mapping::gpis {
    struct DefaultParams {
        struct ComputeVariance {
            inline static const double kZeroGradientPositionVar = 1.;
            inline static const double kZeroGradientGradientVar = 1.;
            inline static const double kMinDistanceVar = 1.;
            inline static const double kMaxDistanceVar = 100.;
            inline static const double kPositionVarAlpha = 0.01;
            inline static const double kMinGradientVar = 0.01;
            inline static const double kMaxGradientVar = 1.;
        };

        struct GpTheta {
            inline static const int kGroupSize = 26;
            inline static const int kOverlapSize = 6;
            inline static const double kBoundaryMargin = 0.0175;
            inline static const double kInitVariance = 1e6;
            inline static const double kSensorRangeVar = 0.01;
            inline static const double kMaxValidDistanceVar = 0.1;
            inline static const double kOccTestTemperature = 30;

            struct TrainBuffer {
                inline static const double kValidRangeMin = 0.2;
                inline static const double kValidRangeMax = 30;
                inline static const double kValidAngleMin = -135. / 180. * M_PI;
                inline static const double kValidAngleMax = 135. / 180. * M_PI;
                struct Mapping {
                    inline static const gaussian_process::Mapping::Type kType = gaussian_process::Mapping::Type::kExp;
                    inline static const double scale = 1.0;
                };
            };
        };

        struct GpSdf {
            struct Kernel {
                inline static const covariance::Covariance::Type kType = covariance::Covariance::Type::kMatern32;
                inline static const double kAlpha = 1.0;
                inline static const double kScale = 1.2;
            };
        };

        struct NodeContainer {
            inline static const int kCapacity = 1;
            inline static const double kMinSquaredDistance = 0.04;
        };

        struct Quadtree {
            inline static const bool kNodesInLeavesOnly = true;
            inline static const double kClusterHalfAreaSize = 0.8;
            inline static const double kMaxHalfAreaSize = 150;
            inline static const double kMinHalfAreaSize = 0.2;
        };

        struct UpdateMapPoints {
            inline static const double kMinObservableOcc = -0.1;
            inline static const double kMaxSurfaceAbsOcc = 0.02;
            inline static const double kMaxValidGradientVar = 0.5;
            inline static const int kMaxAdjustTries = 10;
            inline static const double kMaxBayesPositionVar = 1.;
            inline static const double kMaxBayesGradientVar = 0.6;
        };

        struct UpdateGpSdf {
            inline static const bool kAddOffsetPoints = false;
            inline static const double kOffsetDistance = 0.02;
            inline static const double kSearchAreaScale = 4;
            inline static const double kZeroGradientThreshold = 1.e-6;
            inline static const double kMaxValidGradientVar = 0.1;
            inline static const double kInvalidPositionVar = 2.;
        };

        struct TestQuery {
            inline static const double kMaxTestValidDistanceVar = 0.4;
            inline static const double kSearchAreaHalfSize = 4.8;
            inline static const bool kUseNearestOnly = false;
        };

        inline static const unsigned int num_threads = -1;
        inline static const double kInitTreeHalfSize = 12.8;
        inline static const double kPerturbDelta = 0.01;
    };
}  // namespace erl::gp_sdf
