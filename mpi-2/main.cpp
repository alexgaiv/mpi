#include <Windows.h>
#include <mpi.h>
#include <gdiplus.h> 
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <time.h>
#include <algorithm>
#include <math.h>
#include "stopwatch.h"

#pragma comment(lib, "gdiplus.lib")

int g_size;
int g_rank;

class Image
{
public:
	std::vector<BYTE> pixels;

	Image() { width = height = 0; }
	Image(int width, int height, bool allocData = true);

	int GetWidth() const { return width; }
	int GetHeight() const { return height; }

	BYTE &Pixel(int x, int y) { return pixels[y * width + x]; }
	const BYTE &Pixel(int x, int y) const { return pixels[y * width + x]; }

	bool LoadFromFile(const wchar_t *filename);
	bool SaveToFile(const wchar_t *filename) const;
	void Smooth(int radius, Image &result) const;
	void SmoothParallel(int radius, Image &result) const;
	bool Compare(const Image &image) const;
private:
	int width, height;
};

Image::Image(int width, int height, bool allocData) :
	pixels(allocData ? width * height : 0),
	width(width),
	height(height)
{ }

bool Image::LoadFromFile(const wchar_t *filename)
{
	Gdiplus::Bitmap image(filename);
	if (image.GetLastStatus() != Gdiplus::Ok)
		return false;

	width = image.GetWidth();
	height = image.GetHeight();
	pixels = std::vector<BYTE>(width * height);

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			Gdiplus::Color c;
			image.GetPixel(x, y, &c);
			pixels[y * width + x] = (c.GetR() + c.GetG() + c.GetB()) / 3;
		}

	return true;
}

bool Image::SaveToFile(const wchar_t *filename) const
{
	Gdiplus::Bitmap image(width, height);

	for (int y = 0; y < height; y++)
		for (int x = 0; x < width; x++)
		{
			int val = pixels[y * width + x];
			Gdiplus::Color c(val, val, val);
			image.SetPixel(x, y, c);
		}

	const CLSID pngEncoderClsId = {0x557cf406, 0x1a04, 0x11d3, {0x9a, 0x73, 0x00, 0x00, 0xf8, 0x1e, 0xf3, 0x2e}};
	return image.Save(filename, &pngEncoderClsId) == Gdiplus::Ok;
}

void Image::Smooth(int radius, Image &result) const
{
	Image ret(width, height);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			int sum = 0, n = 0;
			for (int kx = -radius; kx <= radius; kx++)
				for (int ky = -radius; ky <= radius; ky++)
				{
					int x2 = x + kx, y2 = y + ky;

					if (x2 >= 0 && x2 < width &&
						y2 >= 0 && y2 < height)
					{
						sum += Pixel(x2, y2);
						n++;
					}
				}
			ret.Pixel(x, y) = BYTE((double)sum / n);
		}
	}

	result = ret;
}

void Image::SmoothParallel(int radius, Image &result) const
{
	if (g_size >= height) {
		Smooth(radius, result);
		return;
	}

	int wh = width * height;
	int margin = width*radius;

	int *send_counts = new int[g_size];
	int *disps = new int[g_size];
	int ppp = (height / g_size) * width; // pixels per process

	for (int i = 0; i < g_size - 1; i++) {
		send_counts[i] = ppp + (i == 0 ? margin : margin * 2);
		disps[i] = max(0, i * ppp - margin);
	}

	send_counts[g_size - 1] = min(wh, ppp + margin) + (height % g_size) * width;
	disps[g_size - 1] = max(0, (g_size - 1) * ppp - margin);

	int recvCount = send_counts[g_rank];
	BYTE *recv = new BYTE[recvCount];

	MPI_Scatterv(pixels.data(), send_counts, disps, MPI_CHAR, recv, recvCount, MPI_CHAR, 0, MPI_COMM_WORLD);

	Image imgLocal(width, recvCount / width);
	imgLocal.pixels.assign(recv, recv + recvCount);
	imgLocal.Smooth(radius, imgLocal);
	
	if (g_rank == 0) {
		delete[] recv;
		recv = new BYTE[width * height];
	}
	for (int i = 0; i < g_size - 1; i++) {
		send_counts[i] = ppp;
		disps[i] = i * ppp;
	}
	send_counts[g_size - 1] = ppp + (height % g_size) * width;
	disps[g_size - 1] = (g_size - 1) * ppp;
	int data_shift = g_rank == 0 ? 0 : margin;
	
	MPI_Gatherv(imgLocal.pixels.data() + data_shift, send_counts[g_rank], MPI_CHAR, recv,
		send_counts, disps, MPI_CHAR, 0, MPI_COMM_WORLD);

	if (g_rank == 0)
	{
		Image ret(width, height);
		ret.pixels.assign(recv, recv + wh);
		result = ret;
	}
	
	delete[] send_counts;
	delete[] disps;
	delete[] recv;
}

bool Image::Compare(const Image &image) const
{
	int s = (int)pixels.size();
	if (s != image.pixels.size()) return false;

	for (int i = 0; i < s; i++)
		if (pixels[i] != image.pixels[i]) return false;

	return true;
}

int main(int argc, char *argv[])
{
	MpiStopwatch timer;

	Gdiplus::GdiplusStartupInput gdiplusStartupInput;
	ULONG_PTR gdiplusToken;
	Gdiplus::GdiplusStartup(&gdiplusToken, &gdiplusStartupInput, NULL);

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &g_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

	if (argc < 2) {
		MPI_Finalize();
		return 0;
	}

	int radius = atoi(argv[1]);
	if (!radius) radius = 1;

	int width = 0;
	int height = 0;
	Image image;

	if (g_rank == 0)
	{
		bool s = image.LoadFromFile(L"test.png");
		if (!s) return 1;
		width = image.GetWidth();
		height = image.GetHeight();
	}

	MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);
	if (g_rank != 0)
		image = Image(width, height, false);

	Image result1;
	timer.Start();
	image.SmoothParallel(radius, result1);
	double dt1 = timer.Stop();

	if (g_rank == 0)
	{
		result1.SaveToFile(L"result.png");

		Image result2;
		timer.Start();
		image.Smooth(radius, result2);
		double dt2 = timer.Stop();

		printf("Image size: (%d, %d)\n", width, height);
		printf("Radius: %d\n", radius);
		printf("parallel version: %f ms\n", dt1);
		printf("non-parallel version: %f ms\n", dt2);
		printf("acceleration: %f\n", dt2 / dt1);
		printf(result1.Compare(result2) ? "results are the same\n" : "results are differ\n");
	}

	MPI_Finalize();
	Gdiplus::GdiplusShutdown(gdiplusToken);
	return 0;
}