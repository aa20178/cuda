/* Les 2 lignes suivantes permettent d'utiliser nvcc avec gcc 4.7 (sinon erreur de compilation) */
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

// Includes
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
//#include <cmath>
#include <ctime>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"

using namespace std;

/**
 * Fonction remplissant le vecteur data de données aléatoires
 * comprises entre 0 et 1
 */
void gendata(float *data, int n)
{
	for (int i = 0; i < n; i++)
		data[i] = rand()/((float) RAND_MAX);
}

/**
 * Calcul de c = a + b sur CPU
 */
void calc_cpu(float *a, float *b, float *c, int n)
{
	for (int i = 0; i < n; i++)
		c[i] = a[i] + b[i];
}

/**
 * Calcul de l'erreur relative entre le calcul GPU et le
 * calcul CPU
 */
float calcerr(float *gpu, float *cpu, int n)
{
	float err_max = 0.;
	for (int i = 0; i < n; i++)
	{
		const float res = cpu[i];
		float err;

		if (fabs(res) < 1e-7)
			err = fabs(res - gpu[i]);
		else
			err = fabs((res - gpu[i])/res);

		if (err > err_max)
			err_max = err;
	}
	return err_max;
}

/**
 * Kernel de calcul
 */
__global__ void vecadd_device(float *a, float *b, float *c, int n, int *tab)
//__global__ void vecadd_device(float *a, float *b, float *c, int n, int n_warp)
{ 
/*
	if (n_warp == 0)
	{
		const int tid = blockIdx.x*blockDim.x + threadIdx.x;
		if (tid<n)
		   	c[tid]=a[tid]+b[tid];
	}		   
	else
	{
		int offset = n_warp; const int tid = blockIdx.x*blockDim.x + threadIdx.x + offset;
		//const int tid = blockIdx.x* blockDim.x + blockDim.x - 1 - threadIdx.x;
		//const int tid = blockIdx.x*blockDim.x + threadIdx.x + n_warp*32;
		const int tidSup = n_warp*32 + 32;
		if (tid<n && tid<tidSup)
		//if (tid<n)
			c[tid]=a[tid]+b[tid];
	}
*/

		const int tid = blockIdx.x*blockDim.x + tab[threadIdx.x];
		if (tid<n)
		   	c[tid]=a[tid]+b[tid];
}

//void cleanup(float* h_A, float* h_B, float* h_C, float* h_res, 
			//float* d_A, float* d_B, float* d_C)
void cleanup(float* h_A, float* h_B, float* h_C, float* h_res, int* h_tab,
			float* d_A, float* d_B, float* d_C, int* d_tab)
{
	if (h_A)
		free(h_A);	
	if (h_B)
		free(h_B);	
	if (h_C)
		free(h_C);	
	if (h_res)
		free(h_res);		
	if (h_tab)
		free(h_tab);
	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);
	if (d_C)
		cudaFree(d_C);
	if (d_tab)
		cudaFree(d_tab);
	}


// Host code
int main(int argc, char** argv)
{
	
	if (argc != 2) {
		cout << "Usage: addition n\n";
		exit(-1);
	}

	printf("Addition vectorielle\n");
	// Récupération de la taille du vecteur dans argv
    int n  = atoi(argv[1]); // A compléter

	size_t size = n*sizeof(float);
	// cout << "memoire total pour des float = " << size << endl;
 
   	//int n_warp = 20000000;

	// Vecteurs CPU
	float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr;
	// Vecteur GPU
	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

	// Allocatation des vecteurs dans la mémoire CPU
	h_A = (float*) malloc(size);// A compléter
	h_B = (float*) malloc(size);
	h_C = (float*) malloc(size);
	float *h_res = (float*) malloc(size);
	
	

	// adressage indirecte
	size_t sizeInt = 1024*sizeof(int);
	int *h_tab = nullptr;
	h_tab = (int*) malloc(sizeInt);
	int *d_tab = nullptr;
	for (int i=0; i<1024; i++)
		h_tab[i]=i;

	//permutations aléatoires des 256 premiers 
	for (int i=0; i<256; i++) 
	{
		int r=rand()%(i+1);
		h_tab[i]=h_tab[r];
		h_tab[r]=i;
	}	
 

	// Allocation des vecteurs dans la mémoire GPU
	checkCudaErrors(cudaMalloc((void**) &d_A, size));// A compléter
	checkCudaErrors(cudaMalloc((void**) &d_B, size));
	checkCudaErrors(cudaMalloc((void**) &d_C, size));
	checkCudaErrors(cudaMalloc((void**) &d_tab, sizeInt));

	// Initialisation des vecteurs A et B
	srand(time(NULL));
	gendata(h_A, n);
	gendata(h_B, n);


	calc_cpu(h_A, h_B, h_C, n);


	// Copie des vecteur A et B dans la mémoire GPU
	checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tab, h_tab, sizeInt, cudaMemcpyHostToDevice));

	// Appel du kernel
	dim3 threadsPerBlock(1024);
	dim3 numBlocks(ceil(n/(float)threadsPerBlock.x));
	vecadd_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n, d_tab);
	//vecadd_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n, n_warp);
    checkCudaErrors(cudaPeekAtLastError());
	// Synchronisation
	checkCudaErrors(cudaDeviceSynchronize());

	// Copie du resultat de la memoire GPU vers la memoire CPU
	checkCudaErrors(cudaMemcpy(h_res, d_C, size, cudaMemcpyDeviceToHost));

/*
   for(int i=0; i<n; i++)
		cout << "tid = " << h_res[i] << endl;
*/

    // Comparaison des résultats calculés sur le GPU avec ceux calculésBp
	// sur le CPU et affichage de l'erreur
 	std::cout << "Erreur relative : " << calcerr(h_res, h_C, n) << std::endl;// A compléter

	// Timing
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	const int nb = 10;
	checkCudaErrors(cudaEventRecord(start, 0));
	for (int i = 0; i < nb; i++)
		//vecadd_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n, n_warp);	
		vecadd_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n, d_tab);
	checkCudaErrors(cudaEventRecord(stop, 0));	
	checkCudaErrors(cudaEventSynchronize(stop));
    // Calcul du temps d'eécution d'unn kernel
	float t_ms;
	checkCudaErrors(cudaEventElapsedTime(&t_ms, start, stop));
	t_ms /= nb;



	printf("Temps d'exécution du Kernel : %e (ms)\n", t_ms);
    // Affichage de la bande passante en GO/s
/*	
	if (n_warp == 0)
	{
		printf("Bande passante : %e (GO/s)\n", 3*size/1.024/1024/1024/t_ms); // A compléter);
	}		   
	else
	{
		//size_t sizeOffset = n_warp*sizeof(float);
		//printf("Bande passante : %e (GO/s)\n", 3*(size-sizeOffset)/1.024/1024/1024/t_ms); 
		printf("Bande passante : %e (GO/s)\n", 3*32*4/1.024/1024/1024/t_ms); // pour un warp 32 threads de 4 octets
	}
*/

	printf("Bande passante : %e (GO/s)\n", 3*size/1.024/1024/1024/t_ms); // A compléter);



	cleanup(h_A, h_B, h_C, h_res, h_tab, d_A, d_B, d_C, d_tab);
	//cleanup(h_A, h_B, h_C, h_res, d_A, d_B, d_C);
		
	return 0;
}
