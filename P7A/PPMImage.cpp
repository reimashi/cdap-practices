#include "PPMImage.h"
#include <stdio.h>
#include <stdlib.h>

void loadPpmImage(char* path, PPMImage* image) {
	FILE *fIn;
	fIn = fopen(path, "rb");
	fread(image->header, 1, image->headerSize, fIn);
	fread(image->data, 1, image->dataSize, fIn);
	fclose(fIn);
}

void savePpmImage(char* path, PPMImage* image) {
	FILE *fOut;
	fOut = fopen(path, "wb");
	fwrite(image->header, 1, image->headerSize, fOut);
	fwrite(image->data, 1, image->dataSize, fOut);
	fclose(fOut);
}
