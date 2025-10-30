# 
# Usage: To re-create this platform project launch xsct with below options.
# xsct D:\Vitis\USERS\10_Zedboard_audio_in\SD_read\soc_sdk\SD_read\platform.tcl
# 
# OR launch xsct and run below command.
# source D:\Vitis\USERS\10_Zedboard_audio_in\SD_read\soc_sdk\SD_read\platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

platform create -name {SD_read}\
-hw {D:\Vitis\USERS\10_Zedboard_audio_in\SD_read\hw\design_1_wrapper.xsa}\
-proc {ps7_cortexa9_0} -os {standalone} -out {D:/Vitis/USERS/10_Zedboard_audio_in/SD_read/soc_sdk}

platform write
platform generate -domains 
platform active {SD_read}
bsp reload
bsp setlib -name xilffs -ver 4.7
bsp write
bsp reload
catch {bsp regenerate}
bsp config use_lfn "0"
bsp config use_lfn "1"
bsp write
bsp reload
catch {bsp regenerate}
platform generate
platform active {SD_read}
bsp reload
bsp reload
platform generate -domains 
