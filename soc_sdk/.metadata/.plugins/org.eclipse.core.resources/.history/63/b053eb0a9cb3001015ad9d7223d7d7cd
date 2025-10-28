/******************************************************************************
* Copyright (c) 2013 - 2020 Xilinx, Inc.  All rights reserved.
* SPDX-License-Identifier: MIT
******************************************************************************/

/***************************** Include Files *********************************/
#include <stdio.h>
#include "sleep.h"
#include "xparameters.h"	/* SDK generated parameters */
#include "xplatform_info.h"
#include "xil_printf.h"
#include "xil_cache.h"
#include "xsdps.h"			/* SD device driver */
#include "ff.h"

/************************** Constant Definitions *****************************/

/**************************** Type Definitions *******************************/

/***************** Macros (Inline Functions) Definitions *********************/

/************************** Function Prototypes ******************************/
int SD_Transfer_read(char *filename, u32 FileSize);
int SD_Transfer_read2(char *filename, u32 FileSize);
int SD_Transfer_read3(char *filename, u32 FileSize);
int SD_Init(void);
/************************** Variable Definitions *****************************/
static FIL fil;		/* File object */
static FATFS fatfs;

static char FileName[32] = "mapdata4.bin";
static char FileName2[32] = "xhex2.bin";
static char FileName3[32] = "yhex2.bin";
static char *SD_File;


/************************** Buffer Setting *****************************/
u8 DestinationAddress[256*1024] __attribute__ ((aligned(32)));
int lidar_points_x[100];
int	lidar_points_y[100];
int lidar_points[200];

int main(void)
{
	int Status;

	Status = SD_Init();
	if(Status != XST_SUCCESS) {
		print("Error when config SD Card!!!\r\n");
		return XST_FAILURE;
	}
	/*************    ȡSD   ļ        *************/
	//读map.bin文件，写入map
	Status = SD_Transfer_read(FileName, 512*512);
	usleep(10);
	if (Status != XST_SUCCESS) {
		xil_printf("Read SD data file failed!!!\r\n");
		return XST_FAILURE;
	}

	//读激光点bin文件，写入lidar_points_x，lidar_points_y
	//将激光点的X和Y拼接在一起（大概五行代码）
	Status = SD_Transfer_read2(FileName2, 89*4);//读激光点X数据
	usleep(10);
	if (Status != XST_SUCCESS) {
		xil_printf("Read SD data file failed!!!\r\n");
		return XST_FAILURE;
	}
	Status = SD_Transfer_read3(FileName3, 89*4);//读激光点Y数据
	usleep(10);
	if (Status != XST_SUCCESS) {
		xil_printf("Read SD data file failed!!!\r\n");
		return XST_FAILURE;
	}

	int i;
	//inverse x and y points
	for(i=0;i<=99;i++){
		lidar_points_x[i] = 0 - lidar_points_x[i];
		lidar_points_y[i] = 0 - lidar_points_y[i];
	}

	for(i=0;i<=99;i++){  //注意for内部的分号
		lidar_points[2*i] = lidar_points_x[i];
		lidar_points[2*i+1] = lidar_points_y[i];
		}

	return 0;
}





int SD_Init(void)
{
	FRESULT Res;
	/* To test logical drive 0, Path should be "0:/"
	 * * For logical drive 1, Path should be "1:/" */
    TCHAR *Path = "0:/";
    /*Register volume work area, initialize device*/
    Res = f_mount(&fatfs, Path, 0);

    if (Res != FR_OK) {
    	return XST_FAILURE;
    }
    return XST_SUCCESS;
}


/****************************************************************************
* File system example using SD driver to write to and read from an SD card
* in polled mode. This example creates a new file on an
* SD card (which is previously formatted with FATFS), write data to the file
* and reads the same data back to verify.
* @param	None
* @return	XST_SUCCESS if successful, otherwise XST_FAILURE.
******************************************************************************/
//int SD_Transfer_read(void)
//{
//	FRESULT Res;
//	UINT NumBytesRead;
//	u32 FileSize = (256*256);
//
//	/* Open file with required permissions. */
//	SD_File = (char *)FileName;
//
//	Res = f_open(&fil, SD_File, FA_READ);
//	if (Res) {
//		return XST_FAILURE;
//	}
//
//	/* Pointer to beginning of file. */
//	Res = f_lseek(&fil, 0);
//	if (Res) {
//		return XST_FAILURE;
//	}
//
//	/* Read data from file. */
//	Res = f_read(&fil, (void*)DestinationAddress, FileSize,
//			&NumBytesRead);
//	if (Res) {
//		return XST_FAILURE;
//	}
//
//	/* Close file. */
//	Res = f_close(&fil);
//	if (Res) {
//		return XST_FAILURE;
//	}
//
//	return XST_SUCCESS;
//}
int SD_Transfer_read(char *filename, u32 FileSize)
{
	FRESULT Res;
	UINT NumBytesRead;
//	u32 FileSize = (256*256);

	/* Open file with required permissions. */
	SD_File = (char *)filename;

	Res = f_open(&fil, SD_File, FA_READ);
	if (Res) {
		return XST_FAILURE;
	}

	/* Pointer to beginning of file. */
	Res = f_lseek(&fil, 0);
	if (Res) {
		return XST_FAILURE;
	}

	/* Read data from file. */
	Res = f_read(&fil, (void*)DestinationAddress, FileSize,
			&NumBytesRead);
	if (Res) {
		return XST_FAILURE;
	}

	/* Close file. */
	Res = f_close(&fil);
	if (Res) {
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}


int SD_Transfer_read2(char *filename, u32 FileSize)
{
	FRESULT Res;
	UINT NumBytesRead;
//	u32 FileSize = (256*256);

	/* Open file with required permissions. */
	SD_File = (char *)filename;

	Res = f_open(&fil, SD_File, FA_READ);
	if (Res) {
		return XST_FAILURE;
	}

	/* Pointer to beginning of file. */
	Res = f_lseek(&fil, 0);
	if (Res) {
		return XST_FAILURE;
	}

	/* Read data from file. */
	Res = f_read(&fil, (void*)lidar_points_x, FileSize,
			&NumBytesRead);
	if (Res) {
		return XST_FAILURE;
	}

	/* Close file. */
	Res = f_close(&fil);
	if (Res) {
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}

int SD_Transfer_read3(char *filename, u32 FileSize)
{
	FRESULT Res;
	UINT NumBytesRead;
//	u32 FileSize = (256*256);

	/* Open file with required permissions. */
	SD_File = (char *)filename;

	Res = f_open(&fil, SD_File, FA_READ);
	if (Res) {
		return XST_FAILURE;
	}

	/* Pointer to beginning of file. */
	Res = f_lseek(&fil, 0);
	if (Res) {
		return XST_FAILURE;
	}

	/* Read data from file. */
	Res = f_read(&fil, (void*)lidar_points_y, FileSize,
			&NumBytesRead);
	if (Res) {
		return XST_FAILURE;
	}

	/* Close file. */
	Res = f_close(&fil);
	if (Res) {
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}
