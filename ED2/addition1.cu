// Includes
#include <stdio.h>
#include <iostream>
#include <vector>
// Helper functions and utilities to work with CUDA
#include <helper_cuda.h>

void gendata(float *data, int n)
{
	for (int i = 0; i < n; i++)
		data[i] = rand() / ((float)RAND_MAX);
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
			err = fabs((res - gpu[i]) / res);

		if (err > err_max)
			err_max = err;
	}

	return err_max;
}

/**
 * Kernel de calcul
 */
__global__ void vecadd_device(float *a, float *b, float *c, int n)
{
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
		c[tid] = a[tid] + b[tid];
}

void cleanup(float *h_A, float *h_B, float *h_C,
			 float *d_A, float *d_B, float *d_C)
{
	if (h_A)
		free(h_A);
	if (h_B)
		free(h_B);
	if (h_C)
		free(h_C);
	if (d_A)
		cudaFree(d_A);
	if (d_B)
		cudaFree(d_B);
	if (d_C)
		cudaFree(d_C);
}

// Host code
int main(int argc, char **argv)
{
	if ((argc != 2)) // || (atoi(argv[1]) < 1))
	{
		std::cout << " il faut un seul argument ! " << std::endl;
	}

	std::cout << "Addition vectorielle\n"
			  << std::endl;
	int n = 0; // 128 * 1024;
	n = atoi(argv[1]);
	std::cout << "n = " << n << std::endl;

	size_t size = n * sizeof(float);

	// Vecteurs CPU
	float *h_A = nullptr, *h_B = nullptr, *h_C = nullptr, *h_res_from_gpu = nullptr;

	// Vecteur GPU
	float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
	// Allocatation des vecteurs dans la mémoire CPU
	h_A = new float[n];
	h_B = new float[n];
	h_C = new float[n];
	h_res_from_gpu = new float[n];

	// Allocation des vecteurs dans la mémoire GPU
	// A compléter
	checkCudaErrors(cudaMalloc((float **)&d_A, size)); // pourquoi pas float* plutot ? parce qu'on donne l'adress du ptr
	checkCudaErrors(cudaMalloc((float **)&d_B, size));
	checkCudaErrors(cudaMalloc((float **)&d_C, size));

	// Initialisation des vecteurs A et B
	gendata(h_A, n);
	gendata(h_B, n);
	gendata(h_C, n);
	gendata(h_res_from_gpu, n); // au cas où

	// Copie des vecteur A et B dans la mémoire GPU
	checkCudaErrors(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

	cudaEvent_t start, stop;
	//cudaEvent_t *startp = &start;
	//cudaEvent_t *stopp = &stop;

	// Appel du kernel
	dim3 threadsPerBlock(512);							// arbitraire, à changer
	dim3 numBlocks(ceil(n / (float)threadsPerBlock.x)); // pourquoi ?

	vecadd_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n); // lancement du kernel

	checkCudaErrors(cudaEventCreate(&start)); // préparation du chronomètre
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	float elapsed(0);

	for (int j(0); j < 10; j++)
	{
		vecadd_device<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n); // lancement du kernel
		// on rattrape les exceptions
		checkCudaErrors(cudaPeekAtLastError());
		// Synchronisation
		checkCudaErrors(cudaDeviceSynchronize());
	}

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&elapsed, start, stop));

	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop)); // fin du chronometre

	// Comparaison des résultats calculés sur le GPU avec ceux calculés
	// sur le CPU et affichage de l'erreur
	// A compléter

	// Copie du resultat de la memoire GPU vers la memoire CPU
	checkCudaErrors(cudaMemcpy(h_res_from_gpu, d_C, size, cudaMemcpyDeviceToHost));

	// calcul côté cpu
	calc_cpu(h_A, h_B, h_C, n);

	//calcul de l'erreur
	std::cout << "Erreur relative : " << calcerr(h_res_from_gpu, h_C, n) << std::endl;
	elapsed = elapsed / 1000; // pour avoir le tps en sec
	// Timing: calcul du temps d'eécution d'unn kernel : déclaration
	float octets_echanges(3 * size / pow(10, 9));
	float temps_moyen(elapsed / 10);
	std::cout << "temps d'execution moyen  : " << temps_moyen << " sec " << std::endl
			  << "nombre d'octets lis/ecrits obtenue ( 2 lis, un écrit, pour chaque thread )  : "
			  << octets_echanges << "Go" << std::endl
			  << " bande passante :" << octets_echanges / temps_moyen << " GO/s" << std::endl; //total

	// A compléter
	// Calcul du temps d'eécution d'unn kernel

	// Affichage du temps d'exécution en ms et de de la bande passante en GO/s
	// A compléter

	cleanup(h_A, h_B, h_C, d_A, d_B, d_C);

	return 0;
}