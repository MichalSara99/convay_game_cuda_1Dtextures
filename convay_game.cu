#include"convay_game.h"

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<curand.h>
#include<curand_kernel.h>
#include<stdlib.h>
#include<time.h>


#define THREADS_PER_BLOCK 10

struct neighbours {
	int leftIdx;
	int topIdx;
	int topLeftIdx;
	int topRightIdx;
	int rightIdx;
	int bottomIdx;
	int bottomLeftIdx;
	int bottomRightIdx;
};

// global texture references:

texture<int> boardTex;
texture<int> roBoardTex;

__device__
int getCount(bool readOnly,int tid, int width, long size, neighbours n) {

	int cnt{ 0 };
	if (readOnly) {
		// check the corners and boundaries:
		if (tid == 0) {
			cnt += tex1Dfetch(roBoardTex, n.rightIdx) + tex1Dfetch(roBoardTex,n.bottomIdx) + tex1Dfetch(roBoardTex,n.bottomRightIdx);
			return cnt;
		}

		// bottom left corner:
		if (tid == (size - width)) {
			cnt += tex1Dfetch(roBoardTex,n.topIdx) + tex1Dfetch(roBoardTex,n.topRightIdx) + tex1Dfetch(roBoardTex,n.rightIdx);
			return cnt;
		}
		//	top right corner:
		if (tid == (width - 1)) {
			cnt += tex1Dfetch(roBoardTex,n.leftIdx) + tex1Dfetch(roBoardTex,n.bottomLeftIdx) + tex1Dfetch(roBoardTex,n.bottomIdx);
			return cnt;
		}
		//	bottom right corner:
		if (tid == (size - 1)) {
			cnt += tex1Dfetch(roBoardTex,n.topIdx) + tex1Dfetch(roBoardTex,n.topLeftIdx) + tex1Dfetch(roBoardTex,n.leftIdx);
			return cnt;
		}
		// left boundary of the board:
		if ((tid % width) == 0) {
			cnt += tex1Dfetch(roBoardTex,n.topIdx) + tex1Dfetch(roBoardTex,n.topRightIdx)
				+ tex1Dfetch(roBoardTex,n.rightIdx) + tex1Dfetch(roBoardTex,n.bottomRightIdx) + tex1Dfetch(roBoardTex,n.bottomIdx);
			return cnt;
		}
		// top boundary of the board:
		if ((tid >= 0) && (tid <= (width - 1))) {
			cnt += tex1Dfetch(roBoardTex, n.leftIdx) + tex1Dfetch(roBoardTex, n.bottomLeftIdx)
				+ tex1Dfetch(roBoardTex, n.bottomIdx) + tex1Dfetch(roBoardTex, n.bottomRightIdx) + tex1Dfetch(roBoardTex, n.rightIdx);
			return cnt;
		}
		// right boundary of the board:
		if (((tid + 1) % width) == 0) {
			cnt += tex1Dfetch(roBoardTex,n.topIdx) + tex1Dfetch(roBoardTex,n.topLeftIdx)
				+ tex1Dfetch(roBoardTex,n.leftIdx) + tex1Dfetch(roBoardTex,n.bottomLeftIdx) + tex1Dfetch(roBoardTex,n.bottomIdx);
			return cnt;
		}
		// bottom boundary of the board:
		if ((tid >= (size - width)) && (tid <= (size - 1))) {
			cnt += tex1Dfetch(roBoardTex,n.leftIdx) + tex1Dfetch(roBoardTex,n.topLeftIdx)
				+ tex1Dfetch(roBoardTex,n.topIdx) + tex1Dfetch(roBoardTex,n.topRightIdx) + tex1Dfetch(roBoardTex,n.rightIdx);
			return cnt;
		}

		cnt += tex1Dfetch(roBoardTex,n.leftIdx) + tex1Dfetch(roBoardTex,n.topLeftIdx) + tex1Dfetch(roBoardTex,n.topIdx) + 
			tex1Dfetch(roBoardTex,n.topRightIdx) + tex1Dfetch(roBoardTex,n.rightIdx) + tex1Dfetch(roBoardTex,n.bottomRightIdx) + 
			tex1Dfetch(roBoardTex,n.bottomIdx) + tex1Dfetch(roBoardTex,n.bottomLeftIdx);
		return cnt;
	}
	else {

		// check the corners and boundaries:
		if (tid == 0) {
			cnt += tex1Dfetch(boardTex,n.rightIdx) + tex1Dfetch(boardTex,n.bottomIdx) + tex1Dfetch(boardTex,n.bottomRightIdx);
			return cnt;
		}

		// bottom left corner:
		if (tid == (size - width)) {
			cnt += tex1Dfetch(boardTex,n.topIdx) + tex1Dfetch(boardTex,n.topRightIdx) + tex1Dfetch(boardTex,n.rightIdx);
			return cnt;
		}
		//	top right corner:
		if (tid == (width - 1)) {
			cnt += tex1Dfetch(boardTex,n.leftIdx) + tex1Dfetch(boardTex,n.bottomLeftIdx) + tex1Dfetch(boardTex,n.bottomIdx);
			return cnt;
		}
		//	bottom right corner:
		if (tid == (size - 1)) {
			cnt += tex1Dfetch(boardTex,n.topIdx) + tex1Dfetch(boardTex,n.topLeftIdx) + tex1Dfetch(boardTex,n.leftIdx);
			return cnt;
		}
		// left boundary of the board:
		if ((tid % width) == 0) {
			cnt += tex1Dfetch(boardTex,n.topIdx) + tex1Dfetch(boardTex,n.topRightIdx)
				+ tex1Dfetch(boardTex,n.rightIdx) + tex1Dfetch(boardTex,n.bottomRightIdx) + tex1Dfetch(boardTex,n.bottomIdx);
			return cnt;
		}
		// top boundary of the board:
		if ((tid >= 0) && (tid <= (width - 1))) {
			cnt += tex1Dfetch(boardTex,n.leftIdx) + tex1Dfetch(boardTex,n.bottomLeftIdx)
				+ tex1Dfetch(boardTex,n.bottomIdx) + tex1Dfetch(boardTex,n.bottomRightIdx) + tex1Dfetch(boardTex,n.rightIdx);
			return cnt;
		}
		// right boundary of the board:
		if (((tid + 1) % width) == 0) {
			cnt += tex1Dfetch(boardTex,n.topIdx) + tex1Dfetch(boardTex,n.topLeftIdx)
				+ tex1Dfetch(boardTex,n.leftIdx) + tex1Dfetch(boardTex,n.bottomLeftIdx) + tex1Dfetch(boardTex,n.bottomIdx);
			return cnt;
		}
		// bottom boundary of the board:
		if ((tid >= (size - width)) && (tid <= (size - 1))) {
			cnt += tex1Dfetch(boardTex,n.leftIdx) + tex1Dfetch(boardTex,n.topLeftIdx)
				+ tex1Dfetch(boardTex,n.topIdx) + tex1Dfetch(boardTex,n.topRightIdx) + tex1Dfetch(boardTex,n.rightIdx);
			return cnt;
		}

		cnt += tex1Dfetch(boardTex,n.leftIdx) + tex1Dfetch(boardTex,n.topLeftIdx) + tex1Dfetch(boardTex,n.topIdx) + 
			tex1Dfetch(boardTex,n.topRightIdx) + tex1Dfetch(boardTex,n.rightIdx) + tex1Dfetch(boardTex,n.bottomRightIdx) + 
			tex1Dfetch(boardTex,n.bottomIdx)+ tex1Dfetch(boardTex,n.bottomLeftIdx);
		return cnt;
	}
	
}

__global__
void convay_kernel(int *board,bool readOnly, long long size) {
	unsigned int const tidx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int const tidy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int const tid = tidx + tidy * gridDim.x * blockDim.x;

	if (tid >= size)return;

	// width of the board:
	int const width = gridDim.x * blockDim.x;

	// compute indices of all neighbors:
	neighbours n;
	n.leftIdx = tid - 1;
	n.topIdx = tid - width;
	n.topLeftIdx = n.topIdx - 1;
	n.topRightIdx = n.topIdx + 1;
	n.rightIdx = tid + 1;
	n.bottomIdx = tid + width;
	n.bottomLeftIdx = n.bottomIdx - 1;
	n.bottomRightIdx = n.bottomIdx + 1;

	int cnt = getCount(readOnly, tid, width, size, n);
	int check{ 0 };

	if (readOnly) {
		check = tex1Dfetch(roBoardTex, tid);
	}
	else {
		check = tex1Dfetch(boardTex, tid);
	}

	if (check == 0)
		board[tid] = (int)(cnt == 3);
	else
		board[tid] = (int)((cnt == 2) || (cnt == 3));


}


__global__
void init_random(unsigned int seed, curandState_t* states, long size) {
	unsigned int const tidx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int const tidy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int const tid = tidx + tidy * gridDim.x * blockDim.x;
	if (tid >= size)return;

	curand_init(seed, tid, 0, &states[tid]);
}

__global__
void generateBoard(int* board, curandState_t* states, long size) {
	unsigned int const tidx = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int const tidy = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int const tid = tidx + tidy * gridDim.x * blockDim.x;

	if (tid >= size)return;

	board[tid] = curand(&states[tid]) % 2;

}


void ConvayGame::print(int* board, long counter) {
	std::size_t const totalSize = boardSize_ * boardSize_;
	std::cout << counter << ".generation\n| ";
	for (std::size_t t = 0; t < totalSize; ++t) {
		if ((t > 0) && (t % boardSize_) == 0)
			std::cout << "|\n| ";
		std::cout << board[t] << " ";
	}
	std::cout << "|\n";
}


void ConvayGame::initBoard(int* DBoard) {
	dim3 const blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 const gridSize = dim3((boardSize_ + blockSize.x - 1) / blockSize.x,
		(boardSize_ + blockSize.y - 1) / blockSize.y);
	long const totalSize = boardSize_ * boardSize_;

	curandState_t* states;
	cudaMalloc((void**)&states, sizeof(curandState_t) * totalSize);
	init_random << <gridSize, blockSize >> > (time(0), states, totalSize);
	generateBoard << <gridSize, blockSize >> > (DBoard, states, totalSize);
	cudaFree(states);
}

void ConvayGame::singleGeneration(int* board,bool readOnly) {
	dim3 const blockSize(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 const gridSize = dim3((boardSize_ + blockSize.x - 1) / blockSize.x,
		(boardSize_ + blockSize.y - 1) / blockSize.y);
	long const totalSize = boardSize_ * boardSize_;
	convay_kernel << <gridSize, blockSize >> > (board, readOnly, totalSize);
}



void ConvayGame::play() {
	int const totalSize = boardSize_ * boardSize_;

	int* hBoard = (int*)malloc(sizeof(int) * totalSize);
	int* dBoard;
	int* dROBoard;
	cudaMalloc((void**)&dBoard, sizeof(int) * totalSize);
	cudaMalloc((void**)&dROBoard, sizeof(int) * totalSize);

	cudaBindTexture(NULL, roBoardTex, dROBoard, totalSize);
	cudaBindTexture(NULL, boardTex, dBoard, totalSize);


	initBoard(dROBoard);
	cudaMemcpy(hBoard, dROBoard, sizeof(int) * totalSize,
		cudaMemcpyKind::cudaMemcpyDeviceToHost);
	print(hBoard, 0);
	volatile bool readOnly = true;
	for (std::size_t t = 1; t < generations_; ++t) {
		int* in, * out;
		if (readOnly) {
			in = dROBoard;
			out = dBoard;
		}
		else {
			in = dBoard;
			out = dROBoard;
		}
		singleGeneration(out, readOnly);
		cudaMemcpy(hBoard, dBoard, sizeof(int) * totalSize,
			cudaMemcpyKind::cudaMemcpyDeviceToHost);
		print(hBoard, t);
		readOnly != readOnly;
	}


	free(hBoard);
	cudaUnbindTexture(roBoardTex);
	cudaUnbindTexture(boardTex);
	cudaFree(dBoard);
	cudaFree(dROBoard);

}