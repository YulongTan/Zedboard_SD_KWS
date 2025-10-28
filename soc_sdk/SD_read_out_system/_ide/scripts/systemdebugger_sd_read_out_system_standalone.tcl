# Usage with Vitis IDE:
# In Vitis IDE create a Single Application Debug launch configuration,
# change the debug type to 'Attach to running target' and provide this 
# tcl script in 'Execute Script' option.
# Path of this script: D:\Vitis\USERS\10_Zedboard_audio_in\SD_read\soc_sdk\SD_read_out_system\_ide\scripts\systemdebugger_sd_read_out_system_standalone.tcl
# 
# 
# Usage with xsct:
# To debug using xsct, launch xsct and run below command
# source D:\Vitis\USERS\10_Zedboard_audio_in\SD_read\soc_sdk\SD_read_out_system\_ide\scripts\systemdebugger_sd_read_out_system_standalone.tcl
# 
connect -url tcp:127.0.0.1:3121
targets -set -nocase -filter {name =~"APU*"}
rst -system
after 3000
targets -set -filter {jtag_cable_name =~ "Digilent Zed 210248B77C6C" && level==0 && jtag_device_ctx=="jsn-Zed-210248B77C6C-23727093-0"}
fpga -file D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/soc_sdk/SD_read_out/_ide/bitstream/design_1_wrapper.bit
targets -set -nocase -filter {name =~"APU*"}
loadhw -hw D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/soc_sdk/SD_read/export/SD_read/hw/design_1_wrapper.xsa -mem-ranges [list {0x40000000 0xbfffffff}] -regs
configparams force-mem-access 1
targets -set -nocase -filter {name =~"APU*"}
source D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/soc_sdk/SD_read_out/_ide/psinit/ps7_init.tcl
ps7_init
ps7_post_config
targets -set -nocase -filter {name =~ "*A9*#0"}
dow D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/soc_sdk/SD_read_out/Debug/SD_read_out.elf
configparams force-mem-access 0
bpadd -addr &main
