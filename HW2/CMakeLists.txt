cmake_minimum_required(VERSION 4.4)

# Set the default before CUDA compiler detection. Keep explicit cache/CLI
# settings and CUDAARCHS environment overrides for cross-compilation and CI.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND "$ENV{CUDAARCHS}" STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA target architectures")
endif()

project(HW2 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 23)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(hw2 STATIC
    src/storage.cpp
    src/tensor.cpp
    src/activation.cpp
    src/activation_cpu.cpp
    src/activation_gpu.cu
)

target_include_directories(hw2
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"    
)

target_link_libraries(hw2
    PRIVATE
        CUDA::cudart
)

add_executable(hw2_demo
    examples/demo.cpp
)

target_link_libraries(hw2_demo PRIVATE hw2)

include(CTest)
if(BUILD_TESTING)
    add_executable(test_activation tests/test_activation.cpp)
    target_link_libraries(test_activation PRIVATE hw2 CUDA::cudart)

    foreach(dtype IN ITEMS float double)
        add_test(NAME activation_cpu_${dtype}
            COMMAND test_activation cpu ${dtype})
        add_test(NAME activation_gpu_${dtype}
            COMMAND test_activation gpu ${dtype})
        set_tests_properties(activation_gpu_${dtype} PROPERTIES
            SKIP_RETURN_CODE 77 LABELS GPU)
    endforeach()

    option(HW2_TEST_PYTORCH "Enable optional PyTorch reference tests" ON)
    if(HW2_TEST_PYTORCH)
        find_package(Python3 COMPONENTS Interpreter QUIET)
        if(Python3_Interpreter_FOUND)
            add_executable(activation_reference_driver tests/activation_reference_driver.cpp)
            target_link_libraries(activation_reference_driver PRIVATE hw2 CUDA::cudart)
            foreach(backend IN ITEMS cpu gpu)
                foreach(dtype IN ITEMS float double)
                    set(test_name pytorch_${backend}_${dtype})
                    add_test(NAME ${test_name} COMMAND ${Python3_EXECUTABLE}
                        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_pytorch.py
                        --driver $<TARGET_FILE:activation_reference_driver>
                        --backend ${backend} --dtype ${dtype})
                    set_tests_properties(${test_name} PROPERTIES
                        SKIP_RETURN_CODE 77 LABELS PyTorch TIMEOUT 300)
                    if(backend STREQUAL "gpu")
                        set_property(TEST ${test_name} APPEND PROPERTY LABELS GPU)
                    endif()
                endforeach()
            endforeach()
        else()
            message(STATUS "Python3 not found; PyTorch reference tests are not registered")
        endif()
    endif()
endif()
