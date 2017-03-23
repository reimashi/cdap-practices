#pragma once

#define HEAD_LENGTH 15
#define N_COLS 512
#define N_ROWS 512
#define BYTES_PER_PIXEL 3
#define DATA_LENGTH N_COLS * N_ROWS * BYTES_PER_PIXEL

typedef struct {
	long width = N_COLS;
	long height = N_ROWS;
	int bytesPerPixel = BYTES_PER_PIXEL;
	long headerSize = HEAD_LENGTH;
	unsigned char header[HEAD_LENGTH];
	long dataSize = DATA_LENGTH;
	unsigned char data[DATA_LENGTH];
} PPMImage;

// Load a PPM image from file to RAM
void loadPpmImage(char* path, PPMImage* image);

// Save a PPM image form RAM to file
void savePpmImage(char* path, PPMImage* image);