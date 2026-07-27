set script_dir [file normalize [file dirname [info script]]]
set source_dir [file normalize [file join $script_dir .. src]]
set work_root [file normalize [file join $script_dir work]]
file mkdir $work_root
cd $work_root

# Vitis HLS 2022.1 treats the open_project argument as a project name and
# rejects an absolute path.  Enter the isolated work root first.
open_project -reset hls
set_top snn_qp_v05
add_files [file join $source_dir snn_qp_v05_kernel.cpp] \
    -cflags "-std=c++11 -I$source_dir"
open_solution -reset solution_200mhz
set_part {xck26-sfvc784-2LV-c}
create_clock -period 5.0 -name default

if {[info exists ::env(MSRP_CSIM_BUNDLE)] &&
    [info exists ::env(MSRP_CSIM_OUTPUT)]} {
    add_files -tb [file join $source_dir native_fixed_v05.cpp] \
        -cflags "-std=c++11 -I$source_dir"
    csim_design -argv "$::env(MSRP_CSIM_BUNDLE) $::env(MSRP_CSIM_OUTPUT)"
}

csynth_design
exit
