#include "../matrice.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>

#define DIM_PORTION 32

// Code GPU

__global__ void mult_device(const float *A, const float *B, float *output, int n)
{
	__shared__ float shared_A[DIM_PORTION][DIM_PORTION];
	__shared__ float shared_B[DIM_PORTION][DIM_PORTION];

	int x_matrice = blockIdx.x * blockDim.x + threadIdx.x;
	int y_matrice = blockIdx.y * blockDim.y + threadIdx.y;

	if (x_matrice < n && y_matrice < n)
	{
		shared_A[threadIdx.y][threadIdx.x] = A[y_matrice * n + x_matrice];
	}

	__syncthreads();

	if (x_matrice < n && y_matrice < n)
	{
		output[y_matrice * n + x_matrice] = shared_A[threadIdx.x][threadIdx.y];
	}
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

	affichage_resultats_du_kernel(h_A, h_B, h_res, n, t_ms, octets_echanges, affiche);

	return 0;
}
