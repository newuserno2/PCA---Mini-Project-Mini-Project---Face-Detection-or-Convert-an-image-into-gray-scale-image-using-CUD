#include <stdio.h>

__global__ void kernel(unsigned char* d_in, unsigned char* d_out)

{

    int idx = blockIdx.x;
    
	int idy = threadIdx.x;
	
	int gray_adr = idx*64 + idy; 
	
	// calculating address for writing grayscale value
	
	int clr_adr = 3*gray_adr;  
	
	// calculating address for reading RGB values

	if(gray_adr<(64*64))
	
		{
		
			double gray_val = 0.144*d_in[clr_adr] + 0.587*d_in[clr_adr+1] + 0.299*d_in[clr_adr+2];
			
			d_out[gray_adr] = (unsigned char)gray_val;
			
			//printf(" %d:%d=[%d,%d,%d,%d] \n", idx,idy,d_in[clr_adr],d_in[clr_adr+1],d_in[clr_adr+2],(int)gray_val);
			
		}
}

//   Kernel Calling Function

extern "C" void gray_parallel(unsigned char* h_in, unsigned char* h_out, int elems, int rows, int cols)

{

	unsigned char* d_in;
	
	unsigned char* d_out;
	
	cudaMalloc((void**) &d_in, elems);
	
	cudaMalloc((void**) &d_out, rows*cols);
	
	cudaMemcpy(d_in, h_in, elems*sizeof(unsigned char), cudaMemcpyHostToDevice);
	
    kernel<<<rows,cols>>>(d_out, d_in);

	cudaMemcpy(h_out, d_out, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
	
	cudaFree(d_in);
	
	cudaFree(d_out);
	
}
