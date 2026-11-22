# lc_native: one shared library with miniaudio capture, whisper.cpp and its Silero VAD, loaded by livecaptions/native.py.
#
#   cmake -S native/lc_native -B build/native -DCMAKE_BUILD_TYPE=Release [-DLC_GPU=vulkan|metal|cuda|cpu]
#   cmake --build build/native --config Release
#
# whisper.cpp comes from WHISPER_DIR if it exists (e.g. a sibling checkout), otherwise it is fetched at the pinned commit.

cmake_minimum_required(VERSION 3.21)
# macOS: microphone capture works from 13.0; the process tap is used only on 14.2+ (checked at run time)
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0" CACHE STRING "oldest macOS to run on")
project(lc_native C CXX)
if (APPLE)
    enable_language(OBJCXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)   # static whisper/ggml go into a shared library

set(LC_WHISPER_COMMIT "b1535f5fb163f4443804888f7be980621226cb15" CACHE STRING "pinned whisper.cpp commit")
set(WHISPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../whisper.cpp" CACHE PATH "whisper.cpp source folder (fetched if missing)")

if (APPLE)
    set(LC_GPU_DEFAULT metal)
elseif (WIN32)
    set(LC_GPU_DEFAULT vulkan)
else()
    set(LC_GPU_DEFAULT cpu)
endif()
set(LC_GPU ${LC_GPU_DEFAULT} CACHE STRING "GPU backend: vulkan, metal, cuda or cpu")
option(LC_NATIVE_CPU "optimise for this machine's CPU (off for builds to share)" ON)

# whisper.cpp and ggml as static libraries inside lc_native
set(BUILD_SHARED_LIBS      OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE ${LC_NATIVE_CPU} CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_CUDA   OFF CACHE BOOL "" FORCE)
set(GGML_METAL  OFF CACHE BOOL "" FORCE)
if (LC_GPU STREQUAL "vulkan")
    set(GGML_VULKAN ON CACHE BOOL "" FORCE)
elseif (LC_GPU STREQUAL "cuda")
    set(GGML_CUDA ON CACHE BOOL "" FORCE)
elseif (LC_GPU STREQUAL "metal")
    set(GGML_METAL ON CACHE BOOL "" FORCE)
    set(GGML_METAL_EMBED_LIBRARY ON CACHE BOOL "" FORCE)
endif()

if (EXISTS "${WHISPER_DIR}/CMakeLists.txt")
    get_filename_component(WHISPER_DIR "${WHISPER_DIR}" ABSOLUTE)
    add_subdirectory(${WHISPER_DIR} w EXCLUDE_FROM_ALL)   # short folder name: MSBuild paths must stay under 260 characters
    set(LC_WHISPER_SRC ${WHISPER_DIR})
else()
    cmake_minimum_required(VERSION 3.28)   # EXCLUDE_FROM_ALL in FetchContent_Declare
    include(FetchContent)
    FetchContent_Declare(whisper GIT_REPOSITORY https://github.com/ggml-org/whisper.cpp.git GIT_TAG ${LC_WHISPER_COMMIT} EXCLUDE_FROM_ALL)
    FetchContent_MakeAvailable(whisper)
    set(LC_WHISPER_SRC ${whisper_SOURCE_DIR})
endif()

add_library(lc_native SHARED lc_native.cpp lc_native.h lc_internal.h)
if (APPLE)
    target_sources(lc_native PRIVATE lc_tap_macos.mm)
    set_source_files_properties(lc_tap_macos.mm PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
endif()
target_include_directories(lc_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${LC_WHISPER_SRC}/examples)   # miniaudio.h
target_link_libraries(lc_native PRIVATE whisper ggml)
if (MSVC)
    target_compile_options(lc_native PRIVATE /utf-8)
    target_compile_definitions(lc_native PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if (APPLE)
    target_link_libraries(lc_native PRIVATE "-framework CoreAudio" "-framework AudioToolbox" "-framework CoreFoundation"
                          "-framework Foundation")
endif()
if (UNIX AND NOT APPLE)
    target_link_libraries(lc_native PRIVATE dl pthread m)
endif()
set_target_properties(lc_native PROPERTIES PREFIX "" OUTPUT_NAME "lc_native")

# the library lands in livecaptions/bin, where native.py looks first
install(TARGETS lc_native RUNTIME DESTINATION . LIBRARY DESTINATION .)
