# Apply this property while constructing the ZynqMP platform used by v06.
# Four-byte AXI-Lite decode is required because the ABI has many 32-bit
# registers interleaved with 64-bit scalar pairs.
set_property CONFIG.PSU__MAXIGP0__DATA_WIDTH {32} [get_bd_cells zynq_ultra_ps_e_0]
