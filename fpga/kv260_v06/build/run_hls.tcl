set script_dir [file normalize [file dirname [info script]]]
set source_dir [file normalize [file join $script_dir .. src]]
set work_root [file normalize [file join $script_dir work hls]]
file mkdir $work_root
cd $work_root

open_project -reset hls
set_top snn_qp_v06
add_files [file join $source_dir snn_qp_v06_kernel.cpp] \
    -cflags "-std=c++11 -I$source_dir"
open_solution -reset solution_200mhz
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5.0 -name default

# Optional csim uses the native driver as a compact command/ABI smoke test.
if {[info exists ::env(V06_CSIM_PROBLEM)] &&
    [info exists ::env(V06_CSIM_OUTPUT)]} {
    add_files -tb [file join $source_dir native_fixed_v06.cpp] \
        -cflags "-std=c++11 -I$source_dir"
    csim_design -argv "$::env(V06_CSIM_PROBLEM) $::env(V06_CSIM_OUTPUT)"
}

csynth_design
exit
