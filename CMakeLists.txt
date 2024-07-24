cmake_minimum_required(VERSION 3.28.3)
project(ModelGenerator VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Add source files
file(GLOB SOURCES "src/*.cpp")

# Create the executable
add_executable(model_generator ${SOURCES})

# Define asset and data directories
add_compile_definitions(ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")

# Find and include LibTorch
find_package(Torch REQUIRED)

# Link third-party libraries (if any)
target_link_libraries(model_generator "${TORCH_LIBRARIES}")

# Add tests
enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
