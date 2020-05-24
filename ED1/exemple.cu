/* Les 2 lignes suivantes permettent d'utiliser nvcc avec gcc 4.7 (sinon erreur de compilation) */
#undef _GLIBCXX_ATOMIC_BUILTINS
#undef _GLIBCXX_USE_INT128

#include <iostream>
#include <stdlib.h>
#include <cmath>
// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>

/**
 * Fonction remplissant le vecteur data de donnees aleatoires
 * comprises entre 0 et 1
 */
void gendata(double *data, int n) // je prends un ptr de double et un int, et je randomise le tableau pointé 
{
	for (int i = 0; i < n; i++)
		data[i] = rand() / ((double)RAND_MAX);
}

/**
 * Calcul de f(a) sur CPU
 */
void calc_cpu(double *data, int n) // je prends un tableau de doubles, je vais le remplir x -> f(x)
{
	for (int i = 0; i < n; i++)
	{
		const double x = data[i];
		data[i] = 3. * x * x * cos(x * x + x + 1);
	}
}

/**
 * Calcul de l'erreur relative entre le calcul GPU et le
 * calcul CPU
 */
double calcerr(double *gpu, double *cpu, int n)
{
	double err_max = 0.;

	for (int i = 0; i < n; i++)
	{
		const double res = cpu[i];
		double err;
		if (res == 0)
			err = fabs(res - gpu[i]);
		else
			err = fabs((res - gpu[i]) / res);

		if (err > err_max)
			err_max = err;
	}

	return err_max;
}

/**
 * Kernel de calcul
 */
__global__ void calc_device(double *data)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	const double x = data[tid];
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if(tid < n)
	data[tid] = 3. * x * x * cos(x * x + x + 1);
}

/**
 * Fonction CPU appelant le kernle de calcul
 */
int main(void)
{
	const int n = 128 * 1024;
	double *d_a;
	double h_a[n], h_res[n];

	// Le vecteur h_a est remplis de valeurs aleatoires
	gendata(h_a, n);
	// Allocation du tableau dans la memoire globale de la GPU
	checkCudaErrors(cudaMalloc((void **)&d_a, sizeof(double) * n));
	// Copie des donnees de la memoire CPU vers la memoire GPU
	checkCudaErrors(cudaMemcpy(d_a, h_a, sizeof(double) * n, cudaMemcpyHostToDevice));

	// Dimensions de la grille de thread
	dim3 threadsPerBlock(1024); // il fallait mettre 1024 au lieu de 2048
	dim3 numBlocks(n / threadsPerBlock.x);
	// Lancement du kernel
	calc_device<<<numBlocks, threadsPerBlock>>>(d_a);
	checkCudaErrors(cudaPeekAtLastError());
	// Synchronisation
	checkCudaErrors(cudaDeviceSynchronize());
	//CUT_CHECK_ERROR("Erreur lors du lancment du kernel");

	// Copie du resultat de la memoire GPU vers la memoire CPU
	checkCudaErrors(cudaMemcpy(h_res, d_a, sizeof(double) * n, cudaMemcpyDeviceToHost));

	// Comparaison des resultats calcules sur le GPU avec ceux calcules
	// sur le CPU
	calc_cpu(h_a, n);
	std::cout << "Erreur relative : " << calcerr(h_res, h_a, n) << std::endl;

	// Deallocation de la memoire GPU
	checkCudaErrors(cudaFree((void *)d_a));

	return 0;
}
