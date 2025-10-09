#include "erl_common/test_helper.hpp"
#include "erl_geometry/marching_cubes.hpp"
#include "erl_gp_sdf/bayesian_hilbert_surface_mapping.hpp"
#include "erl_gp_sdf/gp_occ_surface_mapping.hpp"
#include "erl_gp_sdf/gp_sdf_mapping.hpp"

#include <boost/program_options.hpp>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/io/TriangleMeshIO.h>
#include <open3d/visualization/utility/DrawGeometry.h>

using namespace erl::common;

const std::filesystem::path kProjectRootDir = ERL_GP_SDF_ROOT_DIR;
const std::filesystem::path kConfigDir = kProjectRootDir / "config";
int g_argc = 0;
char **g_argv = nullptr;

template<typename Dtype>
struct GridDef : public Yamlable<GridDef<Dtype>> {
    Eigen::Vector3<Dtype> size = Eigen::Vector3<Dtype>::Zero();
    Eigen::Matrix3<Dtype> rotation = Eigen::Matrix3<Dtype>::Identity();
    Eigen::Vector3<Dtype> translation = Eigen::Vector3<Dtype>::Zero();

    GridDef() = default;

    struct YamlConvertImpl {
        static YAML::Node
        encode(const GridDef &grid_def) {
            YAML::Node node;
            ERL_YAML_SAVE_ATTR(node, grid_def, size);
            ERL_YAML_SAVE_ATTR(node, grid_def, rotation);
            ERL_YAML_SAVE_ATTR(node, grid_def, translation);
            return node;
        }

        static bool
        decode(const YAML::Node &node, GridDef &grid_def) {
            if (!node.IsMap()) { return false; }
            ERL_YAML_LOAD_ATTR(node, grid_def, size);
            ERL_YAML_LOAD_ATTR(node, grid_def, rotation);
            ERL_YAML_LOAD_ATTR(node, grid_def, translation);
            return true;
        }
    };
};

template<typename Dtype>
struct ToMeshImpl {

    using AbstractSurfaceMapping = erl::gp_sdf::AbstractSurfaceMapping<Dtype, 3>;
    using GpOccSurfaceMapping = erl::gp_sdf::GpOccSurfaceMapping<Dtype, 3>;
    using BayesianHilbertSurfaceMapping = erl::gp_sdf::BayesianHilbertSurfaceMapping<Dtype, 3>;
    using GpSdfMapping = erl::gp_sdf::GpSdfMapping<Dtype, 3>;

    using Aabb = erl::geometry::Aabb<Dtype, 3>;
    using Vector3 = Eigen::Vector3<Dtype>;
    using Matrix3 = Eigen::Matrix3<Dtype>;
    using Matrix3X = Eigen::Matrix3X<Dtype>;

    struct Options {
        std::string surface_mapping_type = type_name<GpOccSurfaceMapping>();
        std::string surface_mapping_setting_type =
            type_name<typename GpOccSurfaceMapping::Setting>();
        std::string sdf_mapping_bin = "sdf_mapping.bin";
        std::string output_mesh_file = "output_mesh.ply";
        std::string grid_file = "";
        float x_min = 0.0f;
        float x_max = 0.0f;
        float y_min = 0.0f;
        float y_max = 0.0f;
        float z_min = 0.0f;
        float z_max = 0.0f;
        float grid_resolution = 0.05f;
        float iso_value = 0.0f;
        bool near_surface_only = false;
    };

    Options options;
    std::shared_ptr<AbstractSurfaceMapping> surface_mapping = nullptr;
    std::shared_ptr<GpSdfMapping> sdf_mapping = nullptr;
    Matrix3 grid_rotation = Matrix3::Identity();
    Vector3 grid_translation = Vector3::Zero();

    ToMeshImpl() {
        ParseOptions();
        LoadMapping();
        GenerateMesh();
    }

    void
    ParseOptions() {
        ERL_INFO(
            "Surface mapping setting types: {}, {}",
            type_name<typename GpOccSurfaceMapping::Setting>(),
            type_name<typename BayesianHilbertSurfaceMapping::Setting>());
        ERL_INFO("SDF mapping setting types: {}", type_name<typename GpSdfMapping::Setting>());

        bool options_parsed = false;
        namespace po = boost::program_options;
        po::options_description desc;
        try {
            // clang-format off
            desc.add_options()
                ("help", "produce help message")
                (
                    "surface-mapping-type",
                    po::value<std::string>(&options.surface_mapping_type)->default_value(options.surface_mapping_type),
                    "Surface mapping type"
                )
                (
                    "surface-mapping-setting-type",
                    po::value<std::string>(&options.surface_mapping_setting_type)->default_value(options.surface_mapping_setting_type),
                    fmt::format("Surface mapping setting type: {}, {}",
                                type_name<typename GpOccSurfaceMapping::Setting>(),
                                type_name<typename BayesianHilbertSurfaceMapping::Setting>()).c_str()
                )
                (
                    "sdf-mapping-bin",
                    po::value<std::string>(&options.sdf_mapping_bin)->default_value(options.sdf_mapping_bin),
                    "SDF mapping binary file"
                )
                (
                    "output-mesh-file",
                    po::value<std::string>(&options.output_mesh_file)->default_value(options.output_mesh_file),
                    "Output mesh file"
                )
                (
                    "grid-file",
                    po::value<std::string>(&options.grid_file)->default_value(options.grid_file),
                    "Grid file"
                )
                (
                    "x-min",
                    po::value<float>(&options.x_min)->default_value(options.x_min),
                    "X min coordinate"
                )
                (
                    "x-max",
                    po::value<float>(&options.x_max)->default_value(options.x_max),
                    "X max coordinate"
                )
                (
                    "y-min",
                    po::value<float>(&options.y_min)->default_value(options.y_min),
                    "Y min coordinate"
                )
                (
                    "y-max",
                    po::value<float>(&options.y_max)->default_value(options.y_max),
                    "Y max coordinate"
                )
                (
                    "z-min",
                    po::value<float>(&options.z_min)->default_value(options.z_min),
                    "Z min coordinate"
                )
                (
                    "z-max",
                    po::value<float>(&options.z_max)->default_value(options.z_max),
                    "Z max coordinate"
                )
                (
                    "grid-resolution",
                    po::value<float>(&options.grid_resolution)->default_value(options.grid_resolution),
                    "Grid resolution for mesh extraction (in meters)"
                )
                (
                    "iso-value",
                    po::value<float>(&options.iso_value)->default_value(options.iso_value),
                    "Isosurface value for mesh extraction"
                )
                (
                    "near-surface-only",
                    po::bool_switch(&options.near_surface_only),
                    "Whether to generate mesh only near the surface"
                );
            // clang-format on

            po::variables_map vm;
            po::store(po::command_line_parser(g_argc, g_argv).options(desc).run(), vm);
            if (vm.count("help")) {
                std::cout << "Usage: " << g_argv[0] << " [options]" << std::endl
                          << desc << std::endl;
                return;
            }
            po::notify(vm);
            options_parsed = true;
        } catch (const std::exception &e) {
            std::cerr << e.what() << std::endl;
            std::cout << "Usage: " << g_argv[0] << " [options]" << std::endl << desc << std::endl;
        }
        ASSERT_TRUE(options_parsed);

        if (!options.grid_file.empty()) {
            ASSERT_TRUE(std::filesystem::exists(options.grid_file));
            ERL_INFO("Load grid definition from {}.", options.grid_file);
            GridDef<Dtype> grid_def;
            ASSERT_TRUE(grid_def.FromYamlFile(options.grid_file));
            options.x_min = -grid_def.size[0] / 2;
            options.x_max = grid_def.size[0] / 2;
            options.y_min = -grid_def.size[1] / 2;
            options.y_max = grid_def.size[1] / 2;
            options.z_min = -grid_def.size[2] / 2;
            options.z_max = grid_def.size[2] / 2;
            grid_rotation = grid_def.rotation;
            grid_translation = grid_def.translation;
        }
    }

    void
    LoadMapping() {
        auto &setting_factory = YamlableBase::Factory::GetInstance();

        auto surface_mapping_setting = setting_factory.Create(options.surface_mapping_setting_type);
        surface_mapping =
            AbstractSurfaceMapping::Create(options.surface_mapping_type, surface_mapping_setting);
        auto sdf_mapping_setting = std::make_shared<typename GpSdfMapping::Setting>();
        sdf_mapping = std::make_shared<GpSdfMapping>(sdf_mapping_setting, surface_mapping);
        using Serializer = Serialization<GpSdfMapping>;
        ASSERT_TRUE(Serializer::Read(options.sdf_mapping_bin, sdf_mapping.get()));
    }

    void
    GenerateMesh() {
        if (options.x_min == options.x_max || options.y_min == options.y_max ||
            options.z_min == options.z_max) {
            ERL_WARN("Map boundary is not fully defined, using surface mapping boundary.");
            const Aabb map_boundary = surface_mapping->GetMapBoundary();
            const Dtype scaling = 1.0f / surface_mapping->GetScaling();

            options.x_min = map_boundary.min()[0] * scaling;
            options.x_max = map_boundary.max()[0] * scaling;
            options.y_min = map_boundary.min()[1] * scaling;
            options.y_max = map_boundary.max()[1] * scaling;
            options.z_min = map_boundary.min()[2] * scaling;
            options.z_max = map_boundary.max()[2] * scaling;
        }

        if (options.near_surface_only) {
            Eigen::Vector3<Dtype> boundary_size;
            boundary_size[0] = options.x_max - options.x_min;
            boundary_size[1] = options.y_max - options.y_min;
            boundary_size[2] = options.z_max - options.z_min;
            std::vector<Eigen::Vector3<Dtype>> surface_points;
            std::vector<Eigen::Vector3<Dtype>> triangle_normals;
            auto extracted_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
            sdf_mapping->GetMesh(
                boundary_size,
                grid_rotation,
                grid_translation,
                options.grid_resolution,
                options.iso_value,
                surface_points,
                extracted_mesh->triangles_,
                triangle_normals);
            for (const auto &point: surface_points) {
                extracted_mesh->vertices_.push_back(point.template cast<double>());
            }
            for (const auto &normal: triangle_normals) {
                extracted_mesh->triangle_normals_.push_back(normal.template cast<double>());
            }
            extracted_mesh->ComputeVertexNormals();
            open3d::io::WriteTriangleMesh(options.output_mesh_file, *extracted_mesh, true);
            open3d::visualization::DrawGeometries({extracted_mesh});
            return;
        }

        GridMapInfo3D<Dtype> grid_map_info(
            Vector3(options.x_min, options.y_min, options.z_min),
            Vector3(options.x_max, options.y_max, options.z_max),
            Vector3(options.grid_resolution, options.grid_resolution, options.grid_resolution),
            Eigen::Vector3i::Zero());
        ERL_INFO(
            "Using boundary min {} and max {}.",
            grid_map_info.Min().transpose(),
            grid_map_info.Max().transpose());
        constexpr bool row_major = false;
        Matrix3X positions = grid_map_info.GenerateMeterCoordinates(row_major);
        ERL_INFO("{} positions are created.", positions.cols());

        // transform positions using grid rotation and translation
        positions = grid_rotation * positions;
        positions.colwise() += grid_translation;

        typename GpSdfMapping::Distances distances(positions.cols());
        typename GpSdfMapping::Gradients gradients(3, positions.cols());
        typename GpSdfMapping::Variances variances(4, positions.cols());
        typename GpSdfMapping::Covariances covariances(6, positions.cols());
        ASSERT_TRUE(sdf_mapping->Test(positions, distances, gradients, variances, covariances));
        ERL_INFO("SDF mapping prediction done, {} points.", positions.cols());

        auto extracted_mesh = std::make_shared<open3d::geometry::TriangleMesh>();
        constexpr bool parallel = true;
        erl::geometry::MarchingCubes::Run(
            grid_map_info.Min().template cast<double>(),
            grid_map_info.Resolution().template cast<double>(),
            grid_map_info.Shape(),
            distances.template cast<double>(),
            options.iso_value,
            row_major,
            parallel,
            extracted_mesh->vertices_,
            extracted_mesh->triangles_,
            extracted_mesh->triangle_normals_);
        ASSERT_TRUE(!extracted_mesh->vertices_.empty());
        ASSERT_TRUE(!extracted_mesh->triangles_.empty());

        // transform the vertices using grid rotation and translation
        Eigen::Matrix3d rotation = grid_rotation.template cast<double>();
        Eigen::Vector3d translation = grid_translation.template cast<double>();
        for (Eigen::Vector3d &vertex: extracted_mesh->vertices_) {
            vertex = rotation * vertex + translation;
        }

        open3d::io::WriteTriangleMesh(options.output_mesh_file, *extracted_mesh, true);
        open3d::visualization::DrawGeometries({extracted_mesh});
    }
};

template<>
struct YAML::convert<GridDef<double>> : GridDef<double>::YamlConvertImpl {};

template<>
struct YAML::convert<GridDef<float>> : GridDef<float>::YamlConvertImpl {};

TEST(GpSdfMapping, ToMeshD) { ToMeshImpl<double> test; }

TEST(GpSdfMapping, ToMeshF) { ToMeshImpl<float> test; }

int
main(int argc, char *argv[]) {
    testing::InitGoogleTest(&argc, argv);
    g_argc = argc;
    g_argv = argv;
    return RUN_ALL_TESTS();
}
