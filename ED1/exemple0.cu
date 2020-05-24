/* Les 2 lignes suivantes permettent d'utiliser nvcc avec gcc 4.7 (sinon erreur de compilation) */
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <stdlib.h>
#include <cmath>
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_cuda.h"


/**
 * Fonction remplissant le vecteur data de donnees aleatoires
 * comprises entre 0 et 1
 */
void gendata(float *data, int n) {
	for (int i = 0; i < n; i++)
		data[i] = rand()/((float) RAND_MAX);
}


/**
 * Calcul de f(a) sur CPU
 */
void calc_cpu(float *data, int n) {
	for (int i = 0; i < n; i++) {
		const float x = data[i];
		data[i] = 3.*x*x*cos(x*x + x + 1);
	}
}


/**
 * Calcul de l'erreur relative entre le calcul GPU et le
 * calcul CPU
 */
float calcerr(float *gpu, float *cpu, int n) {
	float err_max = 0.;

	for (int i = 0; i < n; i++) {
		const float res = cpu[i];
		float err;
		if (res == 0)
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
__global__ void calc_device(float *data) {
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	const float x = data[tid];
	data[tid] = 3.*x*x*cos(x*x + x + 1);
}

/**
 * Fonction CPU appelant le kernle de calcul
 */
int main(void) {
	const int n = 128*1024;
	float *d_a;
	float h_a[n], h_res[n];

	// Le vecteur h_a est remplis de valeurs aleatoires
	gendata(h_a, n);
	// Allocation du tableau dans la memoire globale de la GPU
	checkCudaErrors(cudaMalloc((void**) &d_a, sizeof(float)*n));
	// Copie des donnees de la memoire CPU vers la memoire GPU
	checkCudaErrors(cudaMemcpy(d_a, h_a, sizeof(float)*n, cudaMemcpyHostToDevice));


	// Dimensions de la grille de thread
	dim3 threadsPerBlock(2048);
	dim3 numBlocks(n/threadsPerBlock.x);
	// Lancement du kernel
	calc_device<<<numBlocks, threadsPerBlock>>>(d_a);
    checkCudaErrors(cudaPeekAtLastError());
	// Synchronisation
	checkCudaErrors(cudaDeviceSynchronize());
	//CUT_CHECK_ERROR("Erreur lors du lancment du kernel");

	// Copie du resultat de la memoire GPU vers la memoire CPU
	checkCudaErrors(cudaMemcpy(h_res, d_a, sizeof(float)*n, cudaMemcpyDeviceToHost));

	// Comparaison des resultats calcules sur le GPU avec ceux calcules
	// sur le CPU
	calc_cpu(h_a, n);
	std::cout << "Erreur relative : " << calcerr(h_res, h_a, n) << std::endl;

	// Deallocation de la memoire GPU
	checkCudaErrors(cudaFree((void*) d_a));

	return 0;
}
