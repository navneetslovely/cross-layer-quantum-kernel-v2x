# qkv2x — ns-3.42 contrib module: SUMO+ns-3 cross-layer V2X misbehaviour testbed.
#
# Requires SUMO's libtraci (set SUMO_HOME to a SUMO install/build that contains
# src/libsumo/libtraci.h and bin/libtracicpp). The model library itself does NOT
# depend on libtraci; only the qkv2x-cosim example links it.

if(NOT DEFINED ENV{SUMO_HOME})
  message(FATAL_ERROR
    "qkv2x: environment variable SUMO_HOME is not set. Point it at a SUMO source "
    "build (the directory containing src/ and bin/). See README.md, Section 4.1.")
endif()

include_directories($ENV{SUMO_HOME}/src)

find_library(SUMO_TRACI_LIB
  NAMES tracicpp libtracicpp
  PATHS $ENV{SUMO_HOME}/bin $ENV{SUMO_HOME}/lib $ENV{SUMO_HOME}/build/bin
  NO_DEFAULT_PATH)
if(NOT SUMO_TRACI_LIB)
  message(FATAL_ERROR
    "qkv2x: libtracicpp not found under $ENV{SUMO_HOME}. Build SUMO with libtraci "
    "(see scripts/install_linux.sh) so that bin/libtracicpp(.a/.so) exists.")
endif()
message(STATUS "qkv2x: using libtraci at ${SUMO_TRACI_LIB}")

build_lib(
  LIBNAME qkv2x
  SOURCE_FILES
    model/bsm-header.cc
    model/attack-engine.cc
    model/correlated-shadowing-loss-model.cc
    model/cross-layer-logger.cc
    model/bsm-application.cc
  HEADER_FILES
    model/bsm-header.h
    model/attack-engine.h
    model/correlated-shadowing-loss-model.h
    model/cross-layer-logger.h
    model/bsm-application.h
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libmobility}
    ${libwifi}
    ${libpropagation}
    ${libapplications}
)
