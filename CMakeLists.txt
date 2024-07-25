cmake_minimum_required(VERSION 3.28.3)
project(ModelGenerator VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)

# Add source files
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Create the executable
add_executable(model_generator ${SOURCES})

# Define asset and data directories
#add_compile_definitions(ASSET_DIR="${CMAKE_SOURCE_DIR}/assets")

# Specify the CUDA installation directory
#set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")

# Find and include LibTorch
#set(Torch_DIR "${CMAKE_SOURCE_DIR}/third_party/libtorch-cxx11-abi-shared-with-deps-2.3.1+cu121/libtorch/share/cmake/Torch")
#find_package(Torch REQUIRED)

# Find CUDA
#find_package(CUDA REQUIRED)



# Link third-party libraries (if any)
#target_link_libraries(model_generator "${TORCH_LIBRARIES}")

# Add tests
#enable_testing()
#file(GLOB TEST_SOURCES "tests/*.cpp")
#foreach(TEST_SOURCE ${TEST_SOURCES})
#    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
#    add_executable(${TEST_NAME} ${TEST_SOURCE})
#    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
#endforeach()
