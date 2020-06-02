#include "matrice.h"

#define DIM_PORTION 2

// Code GPU

__global__ void mult_device(const float *A, const float *B, float *output, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ float B_shared[DIM_PORTION][DIM_PORTION];

	if (i < blockDim.x && j < blockDim.y)
	{

		B_shared[threadIdx.x][threadIdx.y] =B[j*n + i];
		__syncthreads();
		output[(j)*n + i] = B_shared[threadIdx.x][threadIdx.y];
		__syncthreads();
	}
/* 	else if (i < n && j < n)
	{
		output[(j)*n + i] = 5;
	} */
	else
		return;
}

int main(int argc, char **argv)
{

	int n(0);
	bool affiche(false);
	user_input(affiche, n, argc, argv);

	size_t size = n * n * sizeof(float);
	// Matrices CPU
	float *h_A = nullptr, *h_B = nullptr, *h_res = nullptr, *h_temoin = nullptr;
	// Matrices GPU
	float *d_A = nullptr, *d_B = nullptr, *d_res = nullptr;

	// Allocatation des vecteurs dans la mémoire CPU
	h_A = new float[n * n];
	h_B = new float[n * n];
	h_res = new float[n * n];
	h_temoin = new float[n * n];

	// Allocation des vecteurs dans la mémoire GPU
	checkCudaErrors(cudaMalloc((void **)&d_A, size));
	checkCudaErrors(cudaMalloc((void **)&d_B, size));
	checkCudaErrors(cudaMalloc((void **)&d_res, size));

	// Initialisation de la matrice A et B
	srand(time(NULL));
	genmat(h_A, n);
	genmat(h_B, n);
	multiplier_matrice(h_A, h_B, h_temoin, n);

	// Copie de la matrice A dans la mémoire GPU
	checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	// Definition de la taille des blocs et de la grille
	dim3 threadsPerBlock(DIM_PORTION, DIM_PORTION);
	dim3 numBlocks(ceil(n / (float)threadsPerBlock.x), ceil(n / (float)threadsPerBlock.x));
	std::cout << "bx: " << numBlocks.x << " by: " << numBlocks.y << "\n";

	mult_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_res, n);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Copie du résultat
	checkCudaErrors(cudaMemcpy(h_res, d_res, size, cudaMemcpyDeviceToHost));

	printf("Erreur max: %e\n", verify(h_res, h_temoin, n));

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	const int nb = 10;
	checkCudaErrors(cudaEventRecord(start, 0));
	for (int i = 0; i < nb; i++)
		mult_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_res, n);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float t_ms;
	checkCudaErrors(cudaEventElapsedTime(&t_ms, start, stop));
	t_ms /= nb;
	t_ms /= 1000;
	float octets_echanges(2 * size / pow(10, 9));
	multiplier_matrice(h_A, h_B, h_temoin, n);

	affichage_resultats_du_kernel(h_A, h_B, h_res, h_temoin, n, t_ms, octets_echanges, affiche);

	free_gpu(d_A);
	free_gpu(d_B);

	// Deallocation de la memoire CPU
	free_cpu(h_A);
	free_cpu(h_B);
	free_cpu(h_temoin);
	free_cpu(h_res);

	return 0;
}
