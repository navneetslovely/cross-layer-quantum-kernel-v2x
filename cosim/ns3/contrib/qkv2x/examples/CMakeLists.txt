# The co-sim orchestrator links SUMO's libtraci (SUMO_TRACI_LIB is set by the
# parent CMakeLists). Built only when ns-3 examples are enabled.
build_lib_example(
  NAME qkv2x-cosim
  SOURCE_FILES qkv2x-cosim.cc
  LIBRARIES_TO_LINK
    ${libqkv2x}
    ${libcore}
    ${libnetwork}
    ${libmobility}
    ${libwifi}
    ${libpropagation}
    ${libapplications}
    ${SUMO_TRACI_LIB}
)
