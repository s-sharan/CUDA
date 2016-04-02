#include <random>
#include <chrono>
#include <limits>
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

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N) {

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    float sum = 0;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            sum += A[ROW * N + i] * B[i * N + COL];
        }
    }
    C[ROW * N + COL] = sum;
}


void matrixMultiplication(float *A, float *B, float *C, int N){

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
    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A, B, C, N);
}


int main()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int N;
    cout<<"Enter the size of the arrays N:";
    cin>>N;
    int SIZE = N*N;

    // Using Uniform Random number generator to initialize the arrays.
    int max_int = numeric_limits<int>::max();
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
    vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<N; i++){
        for (int j=0; j<N; j++){
            h_A[i*N+j] = distribution(generator);
            h_B[i*N+j] = distribution(generator);
        }
    }

    // Allocate memory on the device
    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    dev_array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    gpu_start = steady_clock::now();
    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N);
    cudaDeviceSynchronize();
    

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();
    gpu_end = steady_clock::now();
    gpu_time_span = duration_cast<duration<double>>(gpu_end - gpu_start);

    cout<<"Time taken to compute the product on a GPU: "<<gpu_time_span.count()<<endl;

    float *cpu_C;
    cpu_C=new float[SIZE];

    // Now do the matrix multiplication on the CPU
    cpu_start = steady_clock::now();
    float sum;
    for (int row=0; row<N; row++){
        for (int col=0; col<N; col++){
            sum = 0.f;
            for (int n=0; n<N; n++){
                sum += h_A[row*N+n]*h_B[n*N+col];
            }
            cpu_C[row*N+col] = sum;
        }
    }
    cpu_end = steady_clock::now();
    cpu_time_span = duration_cast<duration<double>>(cpu_end - cpu_start);

    
    cout<<"Time taken to compute the product on a CPU: "<<cpu_time_span.count()<<endl;

    double err = 0;
    // Check the result and make sure it is correct
    for (int ROW=0; ROW < N; ROW++){
        for (int COL=0; COL < N; COL++){
            err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
        }
    }

	// Writing the input matrices and output matrices into files
    std::ofstream matA("matrixA.txt"); 
    std::ofstream matB("matrixB.txt");
    std::ofstream cpu("cpu.txt");
    std::ofstream gpu("gpu.txt");   

	for (int ROW=0; ROW < N; ROW++){
        for (int COL=0; COL < N; COL++){
    		matA<<h_A[ROW * N + COL]<"   ";
    		matB<<h_B[ROW * N + COL]<"   ";
    		cpu<<cpu_C[ROW * N + COL]<"   ";
            gpu<<h_C[ROW * N + COL]<<"   ";
        }
        cpu<<endl;
        gpu<<endl;
        matA<<endl;
        matB<<endl;
    }

    cout << "Normalised Error: " << err/SIZE << endl;

    return 0;
}
