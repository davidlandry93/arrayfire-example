cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(ArrayfireExample)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeModules)

find_package(ArrayFire REQUIRED)
find_package(CUDA REQUIRED)
find_package(NVVM REQUIRED)

include_directories(${ArrayFire_INCLUDE_DIRS})

set(EXTRA_LIBS ${CMAKE_THREAD_LIBS_INIT} ${CUDA_LIBRARIES} ${NVVM_LIB})

add_executable(pcpca src/pcpca.cpp)
target_link_libraries(pcpca ${ArrayFire_LIBRARIES} ${EXTRA_LIBS})
