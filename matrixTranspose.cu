#include <random>
#include <chrono>
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "dev_array.h"
#include <math.h>
#include <fstream>

using namespace std;
using namespace std::chrono;

__global__ void transposeCoalescedKernel(float *odata, const float *idata, int N)
{
  const int size=64;
  int rows=N/size;
  __shared__ float tile[size][size];


  int x = blockIdx.x * size + threadIdx.x;
  int y = blockIdx.y * size + threadIdx.y;
  int width = gridDim.x * size;

  for (int j = 0; j < size; j += rows)
     tile[threadIdx.y+j][threadIdx.x] = idata[(y+j)*width + x];

  __syncthreads();

  x = blockIdx.y * size + threadIdx.x;  // transpose block offset
  y = blockIdx.x * size + threadIdx.y;

  for (int j = 0; j < size; j += rows)
     odata[(y+j)*width + x] = tile[threadIdx.x][threadIdx.y + j];
}

void transposeCoalesced(float *odata, const float *idata, int N){
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block. 
    // a maximum of 512 threads can be assigned to a block
    dim3 threadsPerBlock(N, N);
    dim3 blocksPerGrid(1, 1);
        if (N*N > 512){
            threadsPerBlock.x = 512;
            threadsPerBlock.y = 512;
            blocksPerGrid.x = ceil(double(N)/double(threadsPerBlock.x));
            blocksPerGrid.y = ceil(double(N)/double(threadsPerBlock.y));
        }
    transposeCoalescedKernel<<<blocksPerGrid,threadsPerBlock>>>(odata,idata, N);
}


int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N;
    cout<<"Enter the size of the arrays N:";
    cin>>N;
    int SIZE = N*N;
    char ch;
    cout<<"Do you want to perform computation on CPU also (y/n)";
    cin>>ch;
    bool flag=false;
    if(ch=='y') flag=true;

    // Using Uniform Random number generator to initialize the arrays.
  	default_random_engine generator(12312);
  	uniform_real_distribution<> distribution(-10.0,10.0);

  	// Making use of steady clock to measure the amount of time taken to compute the product of matrices.
  	steady_clock::time_point gpu_start;
    steady_clock::time_point gpu_end;
    steady_clock::time_point cpu_start;
    steady_clock::time_point cpu_end;
    duration<double> gpu_time_span;
    duration<double> cpu_time_span;

    // Allocate memory on the host
    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);

   

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = distribution(generator);
        }
    }

    // Allocate memory on the device
    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    d_A.set(&h_A[0], SIZE);



    gpu_start = steady_clock::now();
    transposeCoalesced(d_B.getData(), d_A.getData(),N);
    cudaDeviceSynchronize();
    
    d_B.get(&h_B[0], SIZE);
    cudaDeviceSynchronize();
    gpu_end = steady_clock::now();
    gpu_time_span = duration_cast<duration<double>>(gpu_end - gpu_start);

    cout<<"Time taken to compute the product on a GPU: "<<gpu_time_span.count()<<endl;
    double err = 0;
    float *cpu_B;
    if(flag){
	    cpu_B=new float[SIZE];

	    // Now do the matrix multiplication on the CPU
	    cpu_start = steady_clock::now();
	    for (int row=0; row<N; row++){
	        for (int col=0; col<N; col++){
	            cpu_B[row*N+col]=h_A[col*N+row];
	        }
	    }
	    cpu_end = steady_clock::now();
	    cpu_time_span = duration_cast<duration<double>>(cpu_end - cpu_start);
	
    
	    cout<<"Time taken to compute the product on a CPU: "<<cpu_time_span.count()<<endl;
	}

	// Writing the input matrices and output matrices into files
    std::ofstream matA("transpose/inputMatrix.txt"); 
    std::ofstream cpu("transpose/cpu.txt");
    std::ofstream gpu("transpose/gpu.txt");   

	for (int ROW=0; ROW < N; ROW++){
        for (int COL=0; COL < N; COL++){
    		matA<<h_A[ROW * N + COL]<"\t";
    		if(flag) cpu<<cpu_B[ROW * N + COL]<"\t";
            gpu<<h_B[ROW * N + COL]<<"\t";
        }
        if(flag) cpu<<endl;
        gpu<<endl;
        matA<<endl;
    }

    if(flag) cout << "Normalised Error: " << err/SIZE << endl;

    return 0;
}
