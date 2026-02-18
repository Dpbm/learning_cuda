
#include <iostream>
#include <cuda/cmath>

#define SIZE 10000

__managed__ uint32_t data[SIZE]; // correct but old versions may not work well

__global__ void compute(uint32_t* data){
	// blockidx.x --> the current block i'm in
	// blockdim.x --> the amount of threads per block
	// threadidx.x --> the current thread id inside the block
	uint32_t index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < SIZE){
		data[index] = index;
	}
}

void show_data(uint32_t* data){
	for(size_t i = 0; i < SIZE; i++)
		std::cout << "i=" << i << "; data=" << data[i] << std::endl;
}

int main(){

	std::cout << "----Testing unified memory----" << std::endl;

	int threads = 256;
	int blocks = cuda::ceil_div(SIZE, threads); // is the same as (SIZE + threads - 1)/threads
	compute<<<blocks, threads>>>(data);
	cudaDeviceSynchronize();

	std::cout << "After computing: " << std::endl;
	show_data(data);
	
}
