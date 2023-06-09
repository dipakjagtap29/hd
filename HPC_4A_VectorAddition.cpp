Step 1 : !nvcc --version

Step 2: 
code = """

#include <stdio.h>
#include <stdlib.h>

#define N 1000000

__global__ void add(int *a, int *b, int *c) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;
    int size = N * sizeof(int);

    // Allocate memory on host
    a = (int*)malloc(size);
    b = (int*)malloc(size);
    c = (int*)malloc(size);

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = i * 2;
    }

    // Allocate memory on device
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 million threads
    add<<<(N + 255) / 256, 256>>>(d_a, d_b, d_c);

    // Copy result from device to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print first and last elements of result
    printf("c[0]=%d, c[%d] = %d",c[0],N-1,c[N-1]);

    // Free memory
    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
"""
  
  Step 3:
text_file = open("assign4.cu","w")
text_file.write(code)
text_file.close()
  
  Step 4:
!nvcc assign4.cu

Step 5:
!./a.out

Step 6:
!nvprof ./a.out

Step 7 : Now Final Output will be displayed
