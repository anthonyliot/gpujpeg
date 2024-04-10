cmake_minimum_required(VERSION 3.8.0 FATAL_ERROR)
# change version also in configure.ac
project(gpujpeg VERSION 0.22.0 LANGUAGES C CXX)
include(CheckLanguage)

# options
set(BUILD_OPENGL OFF CACHE STRING "Build with OpenGL support, options are: AUTO ON OFF")
set(BUILD_OPENCL AUTO CACHE STRING "Build with OpenCL support, options are: AUTO ON OFF")
set(BUILD_CUDA AUTO CACHE STRING "Build with Cuda support, options are: AUTO ON OFF")
set_property(CACHE BUILD_OPENGL PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE BUILD_OPENCL PROPERTY STRINGS AUTO ON OFF)
set_property(CACHE BUILD_CUDA PROPERTY STRINGS AUTO ON OFF)

option(WITH_CL_WRAPPER "Use OpenCL Loader library instead to use system one" OFF)
option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(CMAKE_CUDA_ARCHITECTURES "CUDA architectures to build for (needs CMake 3.18)" OFF)
option(HUFFMAN_CONST_TABLES "Enable Huffman GPU decoder quick const tables" OFF)
set(OPENGL_CONTEXT AUTO CACHE STRING "Use specifiedn OpenGL context creation, options are: AUTO GLX GLFW NONE")
set_property(CACHE OPENGL_CONTEXT PROPERTY STRINGS AUTO GLX GLFW NONE)

if (BUILD_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "Cuda Found: Include: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
        add_definitions(-DGPUJPEG_USE_CUDA)
        set(CMAKE_CUDA_STANDARD 11)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    else()
        message(STATUS "No CUDA support")
    endif()
else()
    message(STATUS "Build without CUDA support")
endif()

if (BUILD_OPENCL)
    if(NOT WITH_CL_WRAPPER)
        # check support for OpenCL 1.2
        find_package(OpenCL 1.2 REQUIRED QUIET)

        if(OpenCL_FOUND)
            message(STATUS "OpenCL Found: Include: ${OpenCL_INCLUDE_DIR} | Lib/Framework: ${OpenCL_LIBRARY}")
            add_definitions(-DGPUJPEG_USE_OPENCL)
        else()
            message(STATUS "OpenCL Not found.")
        endif()

        add_library(opencl::opencl SHARED IMPORTED GLOBAL)
        set_property(TARGET opencl::opencl PROPERTY IMPORTED_LOCATION ${OpenCL_LIBRARY})
        set_property(TARGET opencl::opencl PROPERTY IMPORTED_IMPLIB ${OpenCL_LIBRARY})
        target_include_directories(opencl::opencl INTERFACE ${OpenCL_LIBRARY})
    else()
        # TODO:
        # use source code from
        # https://github.com/robertwgh/libOpenCL-loader/tree/master
        add_definitions(-DGPUJPEG_USE_OPENCL_WRAPPER)
        add_definitions(-DGPUJPEG_USE_OPENCL)

        set(OpenCL_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/inc/loader")
        file(GLOB OpenCL_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/loader/*)

        file(GLOB_RECURSE SRCS ${OpenCL_SRC_FILES})
        file(GLOB_RECURSE INCS ${OpenCL_INCLUDE_DIR})

        add_library(opencl_opencl STATIC ${SRCS} ${INCS})
        add_library(opencl::opencl ALIAS opencl_opencl)
    endif()

    # OpenGL executables are enabled
    set(GPUJPEG_OPENCL_ENABLED TRUE)
else()
    message(STATUS "Build without OPENCL support")
endif()

if(WIN32)
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/${CMAKE_PROJECT_NAME}")
    endif()
    string(REPLACE "\\" "/" CMAKE_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

cmake_policy(SET CMP0010 NEW) # error on undefined variable

set(CPACK_PACKAGE_INSTALL_DIRECTORY "gpujpeg") # may be removed when version will be explicitly defined
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/COPYING")
set(CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_SOURCE_DIR}/README.md")
include(CPack)

include(CTest)

execute_process(COMMAND git rev-parse --short HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        OUTPUT_VARIABLE GIT_REV
        OUTPUT_STRIP_TRAILING_WHITESPACE)
add_definitions(-DGIT_REV=\"${GIT_REV}\")

if(NOT BUILD_SHARED_LIBS)
    add_definitions("-DGPUJPEG_STATIC")
    set(PC_CFLAGS "${PC_CFLAGS} -DGPUJPEG_STATIC")
    set(PC_LIBS "${PC_LIBS} -lcudart")
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libgpujpeg/gpujpeg_version.h.cmakein
               ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg/gpujpeg_version.h @ONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/libgpujpeg.pc.cmakein
               ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg.pc @ONLY)
add_definitions("-DHAVE_GPUJPEG_VERSION_H")

# Include cmake modules in "cmake" directory (FindGLFW.cmake)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/")

# Common settings
include_directories(${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/src)

set(NEEDED_COMPILER_FEATURES c_std_11)

set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_EXTENSIONS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(COMPILED_OPTIONS)

set(CMAKE_VERBOSE_MAKEFILE ON)

# allow passing <PackageName>_ROOT to find_package()
if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

# add warnings
if(POLICY CMP0092)
    cmake_policy(SET CMP0092 NEW)
endif()

# Fix behavior of CMAKE_CXX_STANDARD when targeting macOS.
if (POLICY CMP0025)
    cmake_policy(SET CMP0025 NEW)
endif ()

# Find OpenGL, GLEW, GLUT and GLFW
if(POLICY CMP0072)
    cmake_policy(SET CMP0072 NEW)
endif()

if (MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /W4")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /W4")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -pedantic")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wall -Xcompiler -Wextra")
endif()

# brew install libomp
# export OpenMP_ROOT=$(brew --prefix)/opt/libomp
find_package(OpenMP)
if(TARGET OpenMP::OpenMP_C)
    message(STATUS "OpenMP: ${OpenMP_C_LIB_NAMES} (${OpenMP_C_INCLUDE_DIR})")
    set(GPUJPEG_OPENMP_ENABLED TRUE)
    add_definitions(-DGPUJPEG_USE_OPENMP)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()

find_package(OpenGL)
find_package(GLEW)
find_package(GLUT)
find_package(GLFW)
find_package(X11)
if(NOT OPENGL_FOUND)
    set(OPENGL_ERROR "OpenGL library not found")
elseif(NOT GLEW_FOUND)
    set(OPENGL_ERROR "GLEW library not found")
endif()
if(NOT BUILD_OPENGL STREQUAL "OFF" AND NOT OPENGL_ERROR)
    message(STATUS "OpenGL: ${OPENGL_LIBRARIES} (${OPENGL_INCLUDE_DIR})")
    message(STATUS "GLEW: ${GLEW_LIBRARIES} (${GLEW_INCLUDE_DIRS})")
    message(STATUS "GLUT: ${GLUT_glut_LIBRARY} (${GLUT_INCLUDE_DIR})")
    message(STATUS "GLFW: ${GLFW_LIBRARIES} (${GLFW_INCLUDE_DIR})")
    message(STATUS "X11: ${X11_LIBRARIES} (${X11_INCLUDE_DIR})")

    # OpenGL executables are enabled
    set(GPUJPEG_OPENGL_ENABLED TRUE)

    # Build GPUJPEG library with OpenGL support
    add_definitions("-DGPUJPEG_USE_OPENGL")
    add_definitions("-DGL_SILENCE_DEPRECATION")

    set(GPUJPEG_OPENGL_LIBRARIES)
    include_directories(${OPENGL_INCLUDE_DIR} ${GLEW_INCLUDE_DIRS})
    list(APPEND GPUJPEG_OPENGL_LIBRARIES GLEW::GLEW)
    list(APPEND GPUJPEG_OPENGL_LIBRARIES ${OPENGL_LIBRARIES})
    if(GLUT_FOUND)
        include_directories(${GLUT_INCLUDE_DIR})
        list(APPEND GPUJPEG_OPENGL_LIBRARIES ${GLUT_glut_LIBRARY})
    endif()
    if(NOT APPLE AND X11_FOUND AND (OPENGL_CONTEXT STREQUAL "AUTO" OR OPENGL_CONTEXT STREQUAL "GLX"))
        include_directories(${X11_INCLUDE_DIR})
        list(APPEND GPUJPEG_OPENGL_LIBRARIES ${X11_LIBRARIES})
        add_definitions("-DGPUJPEG_USE_GLX")
        list(APPEND COMPILED_OPTIONS "OpenGL(GLX)")
    elseif(GLFW_INCLUDE_DIR AND GLFW_FOUND AND NOT OPENGL_CONTEXT STREQUAL "NONE")
        include_directories(${GLFW_INCLUDE_DIR})
        list(APPEND GPUJPEG_OPENGL_LIBRARIES ${GLFW_LIBRARIES})
        add_definitions("-DGPUJPEG_USE_GLFW")
        list(APPEND COMPILED_OPTIONS "OpenGL(GLFW)")
    else()
        list(APPEND COMPILED_OPTIONS "OpenGL(nocontext)")
    endif()
elseif(BUILD_OPENGL STREQUAL "ON")
    message(FATAL_ERROR "OpenGL not found: ${OPENGL_ERROR}")
endif()

if(HUFFMAN_CONST_TABLES)
    add_definitions("-DHUFFMAN_CONST_TABLES")
    set(COMPILED_OPTIONS "${COMPILED_OPTIONS} Huffman_const_tables")
endif()

if(DEFINED GLEW_USE_STATIC_LIBS)
    add_definitions("-DGLEW_STATIC")
endif()

message(STATUS "Configured options: ${COMPILED_OPTIONS}")

# GPUJPEG library
file(GLOB H_FILES libgpujpeg/*.h ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg/gpujpeg_version.h)
file(GLOB_RECURSE C_FILES src/*.c src/*.cu src/*.cpp)
list(REMOVE_ITEM C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.c")

if (NOT BUILD_OPENCL)
    list(REMOVE_ITEM C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/accel/cl/gpujpeg_accel_cl.c")
    list(REMOVE_ITEM C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/accel/cl/gpujpeg_accel_cl.h")
endif()

if (NOT BUILD_CUDA)
    list(REMOVE_ITEM C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/accel/cl/gpujpeg_accel_cuda.cu")
    list(REMOVE_ITEM C_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/accel/cl/gpujpeg_accel_cuda.h")
endif()

add_library(gpujpeg ${H_FILES} ${C_FILES})
target_include_directories(${PROJECT_NAME}
    PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_compile_features(${PROJECT_NAME} PRIVATE ${NEEDED_COMPILER_FEATURES})
set_property(TARGET ${PROJECT_NAME} PROPERTY C_STANDARD 99)
set_target_properties(${PROJECT_NAME} PROPERTIES SOVERSION 0)
set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${H_FILES}")
if(UNIX)
    target_link_libraries(${PROJECT_NAME} m)
endif()
if(GPUJPEG_OPENGL_ENABLED)
    target_link_libraries(${PROJECT_NAME} ${GPUJPEG_OPENGL_LIBRARIES})
endif()
if(GPUJPEG_OPENCL_ENABLED)
    target_link_libraries(${PROJECT_NAME} opencl::opencl)
endif()
if(GPUJPEG_OPENMP_ENABLED)
    target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenMP_C_INCLUDE_DIR}
    )
    target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_C)
endif()

# GPUJPEG commmand-line application
add_executable(gpujpeg-bin src/main.c)
target_compile_features(gpujpeg-bin PRIVATE ${NEEDED_COMPILER_FEATURES})
set_property(TARGET gpujpeg-bin PROPERTY C_STANDARD 99)
target_link_libraries(gpujpeg-bin gpujpeg)
set(GPUJPEG_BIN_OUTPUT_NAME "gpujpegtool")
set_target_properties(gpujpeg-bin PROPERTIES OUTPUT_NAME ${GPUJPEG_BIN_OUTPUT_NAME})

install(TARGETS ${PROJECT_NAME} gpujpeg-bin
        EXPORT GPUJPEGTargets
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        PUBLIC_HEADER DESTINATION include/libgpujpeg
        RUNTIME DESTINATION bin)

install(FILES AUTHORS CONTRIBUTING.md COPYING FAQ.md INSTALL NEWS.md README.md
        DESTINATION doc)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libgpujpeg.pc DESTINATION share/pkgconfig)

# When OpenGL was found, include OpenGL executables
if(GPUJPEG_OPENGL_ENABLED AND GLUT_FOUND)

    # GPUJPEG decoder to OpenGL texture
    file(GLOB FILES test/decoder_gltex/*)
    add_executable(decoder_gltex ${FILES})
    set_property(TARGET decoder_gltex PROPERTY C_STANDARD 99)
    target_link_libraries(decoder_gltex ${PROJECT_NAME} ${GPUJPEG_OPENGL_LIBRARIES})

    # OpenGL interoperability example (currently not working)
    #file(GLOB FILES test/opengl_interop/*.c test/opengl_interop/*.h test/opengl_interop/*.cu)
    #cuda_add_executable(opengl_interop ${FILES})
    #target_link_libraries(opengl_interop src)

endif()

# CMake config
include(CMakePackageConfigHelpers)
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfigVersion.cmake"
    VERSION ${Upstream_VERSION}
    COMPATIBILITY AnyNewerVersion)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfig.cmake"
    INSTALL_DESTINATION "lib/cmake/GPUJPEG"
    NO_CHECK_REQUIRED_COMPONENTS_MACRO
    NO_SET_AND_CHECK_MACRO)
install(EXPORT GPUJPEGTargets
    DESTINATION lib/cmake/GPUJPEG)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/GPUJPEGConfigVersion.cmake
    DESTINATION lib/cmake/GPUJPEG)

file(GLOB_RECURSE ALL_SOURCES *.c *.cu *.cuh *.h *.hpp)
add_custom_target(cscope cscope -bv ${ALL_SOURCES})
add_custom_target(ctags ctags ${ALL_SOURCES})
add_custom_target(universal-ctags ctags-universal ${ALL_SOURCES})

add_test(NAME unittests COMMAND make -C test/unit LIBDIR=${CMAKE_BINARY_DIR} CPATH=${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include LIBRARY_PATH=${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/lib64
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
add_test(NAME test-colors COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/colors/run_tests.sh ${CMAKE_CURRENT_BINARY_DIR}/${GPUJPEG_BIN_OUTPUT_NAME})
add_test(NAME regression COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/regression/run_tests.sh ${CMAKE_CURRENT_BINARY_DIR}/${GPUJPEG_BIN_OUTPUT_NAME})

# vim: sw=4
