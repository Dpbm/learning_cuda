#include <iostream>
#include <cuda/cmath> 

#define SIZE 10000

void initArray(float* data){
	for(size_t i = 0; i < SIZE; i++)
		data[i] = i+10.34f;
}

__global__ void compute(float* data){
	// blockidx.x --> the current block i'm in
	// blockdim.x --> the amount of threads per block
	// threadidx.x --> the current thread id inside the block
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if(index < SIZE){
		data[index] *= 3.3f;
	}
}

void show_data(float* data){
	for(size_t i = 0; i < SIZE; i++)
		std::cout << "i=" << i << "; data=" << data[i] << std::endl;
}

int main(){

	std::cout << "----Testing unified memory----" << std::endl;

	float* data;

	cudaMallocManaged(&data, SIZE*sizeof(float));
	initArray(data);

	std::cout << "Started array with: " << std::endl;
	show_data(data);

	int threads = 256;
	int blocks = cuda::ceil_div(SIZE, threads); // is the same as (SIZE + threads - 1)/threads
	compute<<<blocks, threads>>>(data);
	cudaDeviceSynchronize();

	std::cout << "After computing: " << std::endl;
	show_data(data);
	

	cudaFree(data);
}
