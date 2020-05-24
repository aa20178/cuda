#include <iostream>
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

// Code CPU
void afficher_matrice(float *A, int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			std::cout << A[i * n + j] << "  ";
		}
		std::cout << std::endl;
	}
}
 void multiplier_matrice(float *A, float *B, float *C, int n)
{
	for (int i = 0; i < n * n; i++)
	{
		for (int j = 0; j < n * n; j++)
		{
			C[i + j*n] = 0;
			for (int k = 0; k < n * n; k++)
			{
				C[i+j*n] += A[i+ k*n] * B[k+(j*n)];
			}
		}
	}
} 

void genmat(float *A, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			A[i * n + j] = rand() / (float)RAND_MAX;
}
float verify(const float *A, const float *B, int n)
{
	float error = 0;
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++)
			error = std::max(error, abs(A[i * n + j] - B[i * n + j]));

	return error;
}

int main(int argc, char **argv)
{

	int n = 0;
	bool affiche(false);

	if (argc < 2)
	{
		std::cout << argc << " il faut entrer un argument (taille matrice) " << std::endl;
		exit(-1);
	}
	if (argv[1] != NULL && atoi(argv[1]) > 1)
	{
		n = atoi(argv[1]);
	}
	if (argv[2] != NULL)
	{
		affiche = true;
	}

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
	multiplier_matrice(h_A, h_B, h_temoin,  n);

		// Copie de la matrice A dans la mémoire GPU
		checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	// Definition de la taille des blocs et de la grille
	dim3 threadsPerBlock(DIM_PORTION, DIM_PORTION);
	dim3 numBlocks(ceil(n / (float)threadsPerBlock.x), ceil(n / (float)threadsPerBlock.x));
	std::cout << "bx: " << numBlocks.x << " by: " << numBlocks.y << "\n";

	mult_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B,d_res, n);
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
		mult_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_res,n);
	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float t_ms;
	checkCudaErrors(cudaEventElapsedTime(&t_ms, start, stop));
	t_ms /= nb;
	t_ms /= 1000;
	float octets_echanges(2 * size / pow(10, 9));

	printf("Temps d'exécution du Kernel : %e (ms)\n", t_ms);
	printf("Bande passante GPU: %e GO/s\n", octets_echanges / t_ms);

	if (affiche == true)
	{

		std::cout << " A : " << std::endl;
		afficher_matrice(h_A, n);

		std::cout << " B : " << std::endl;
		afficher_matrice(h_B, n);

		std::cout << " B : " << std::endl;
		afficher_matrice(h_res, n);
	}

	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);

	// Deallocation de la memoire CPU
	if (h_A)
		delete[] h_A;
	if (h_B)
		delete[] h_B;
	return 0;
}
