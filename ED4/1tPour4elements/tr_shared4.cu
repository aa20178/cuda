#include "../matrice.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define DIM_PORTION 32
#define LIGNES_BLOC 8

// Code GPU

__global__ void transpose_device(const float *input, float *output, int n)
{
	__shared__ float matrice_shared[DIM_PORTION][DIM_PORTION];

	int x_matrice = blockIdx.x * blockDim.x + threadIdx.x;
	int y_matrice = blockIdx.y * blockDim.y + threadIdx.y;

	for (int j = 0; j < DIM_PORTION; j += LIGNES_BLOC)
	{
		if (x_matrice < n && y_matrice < n)
		{
			matrice_shared[threadIdx.y][threadIdx.x] = input[(j + y_matrice) * n + x_matrice];
		}

		__syncthreads();

		if (x_matrice < n && y_matrice < n)
		{
			output[(y_matrice + j) * n + x_matrice] = matrice_shared[threadIdx.x][threadIdx.y];
		}
	}
}

int main(int argc, char **argv)
{

	int n(0);
	bool affiche(false);
		user_input(affiche,n,argc,argv);


	size_t size = n * n * sizeof(float);
	// Matrices CPU
	float *h_A = nullptr, *h_B = nullptr;
	// Matrices GPU
	float *d_A = nullptr, *d_B = nullptr;

	// Allocatation des vecteurs dans la mémoire CPU
	h_A = new float[n * n];
	h_B = new float[n * n];

	// Allocation des vecteurs dans la mémoire GPU
	checkCudaErrors(cudaMalloc((void **)&d_A, size));
	checkCudaErrors(cudaMalloc((void **)&d_B, size));

	// Initialisation de la matrice A
	srand(time(NULL));
	genmat(h_A, n);

	// Copie de la matrice A dans la mémoire GPU
	checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

	// Definition de la taille des blocs et de la grille
	dim3 threadsPerBlock(DIM_PORTION, DIM_PORTION);
	dim3 numBlocks(ceil(n / (float)threadsPerBlock.x), ceil(n / (float)threadsPerBlock.x));
	std::cout << "bx: " << numBlocks.x << " by: " << numBlocks.y << "\n";

	transpose_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, n);
	checkCudaErrors(cudaPeekAtLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Copie du résultat
	checkCudaErrors(cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost));
	printf("Erreur max: %e\n", verify(h_A, h_B, n));

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	const int nb = 10;
	checkCudaErrors(cudaEventRecord(start, 0));
	for (int i = 0; i < nb; i++)
		transpose_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, n);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float t_ms;
	checkCudaErrors(cudaEventElapsedTime(&t_ms, start, stop));
	t_ms /= nb;
	t_ms /= 1000;
	float octets_echanges(2 * size / pow(10, 9));

	affichage_resultats_du_kernel(h_A, h_B, n, t_ms, octets_echanges, affiche);

	free_gpu(d_A);
	free_gpu(d_B);

	// Deallocation de la memoire CPU
	free_cpu(h_A);
	free_cpu(h_B);

	return 0;
}
