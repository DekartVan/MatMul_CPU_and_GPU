#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <chrono>
#include <math.h> 
#include <cstdlib> 

#define B_size 16
#define N 2048


// Функция заполнения матриц псевдо-рандомными значениями
double* matrix_random(double* matrix, double size) {
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        matrix[i] = rand() % 100 - round(100 / 2);
    }
    return matrix;
}

// Функция перемножения матриц на GPU
__global__ void matmul_gpu(double* a, double* b, double* c, int n) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float sum = 0.0f;
    int ia = n * B_size * by + n * ty;
    int ib = B_size * bx + tx;

    for (int k = 0; k < n; k++)
    {
        sum += a[ia + k] * b[ib + k * n];
    }

    int ic = n * B_size * by + B_size * bx;
    c[ic + n * ty + tx] = sum;
}

// Функция перемножения двух матриц на CPU
void matrix_mul_CPU(double* aMatrix, double* bMatrix, double* cMatrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int sum = 0;
            for (int k = 0; k < size; k++)
                sum = sum + aMatrix[i * size + k] * bMatrix[k * size + j];
            cMatrix[i * size + j] = sum;
        }
    }
}

int main(int agrs, char * argv[])
{
    setlocale(LC_ALL, "ru"); // Задание языка
    std::ofstream out("out_data.txt", std::ios::app); // Для записи данных о времени выполнения в файл
    typedef std::chrono::microseconds us; // Сокращение для удобства использования
    std::chrono::high_resolution_clock::time_point start, end; 

    // Выделение памяти на хосте
    double* aMatrix = new double[N * N];
    double* bMatrix = new double[N * N];
    double* cMatrix = new double[N * N];

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
        {
            int k = N * i + j;
            aMatrix[k] = 0.0;
            bMatrix[k] = 0.0;
        }

    // Перемножение матриц различных размерностей
    int l = 0;
    for (int i = 16; l < 2048 * 2048; i *= 2) {
        l = pow(i, 2);

        // Заполнение матриц aMatrix и bMatrix случайными значениями 
        matrix_random(aMatrix, l);
        matrix_random(bMatrix, l);

        // Перемножение матриц и фиксирование времени выполнения
        start = std::chrono::high_resolution_clock::now();
        matrix_mul_CPU(aMatrix, bMatrix, cMatrix, sqrt(l));
        end = std::chrono::high_resolution_clock::now();

        us uSec;
        std::chrono::duration<double> duration = end - start;
        uSec = std::chrono::duration_cast<us>(duration);

        if (out.is_open())
        {
            out << "Размерность матриц: " << i << "x" << i << "; CPU_Time: " << uSec.count() << " us" << std::endl;
            std::cout << "Размерность матриц: " << i << "x" << i << "; CPU_Time: " << uSec.count() << " us" << std::endl;
        }
    }
    
    // Работа с CUDA
    // Нужно заметить, что скорость вычисления на GPU будет ограничена не возможностями видеокарты, а еёскоростью доступа 
    // к глобальной памяти. В следующей версии произведу оптимизацию за счёт разбиения матриц.

    int n_Bytes = N*N * sizeof(double);

    double* aMatrDev = NULL;
    double* bMatrDev = NULL;
    double* cMatrDev = NULL;

    cudaMalloc((void**)&aMatrDev, n_Bytes);
    cudaMalloc((void**)&bMatrDev, n_Bytes);
    cudaMalloc((void**)&cMatrDev, n_Bytes);

    // Конфигурирование ядра

    dim3 threads(B_size, B_size);
    dim3 blocks(N / threads.x, N / threads.y);

    // Создаём обработчик событий CUDA
    float gpuTime = 0.0f;

    l = 0;
    for (int i = 16; l < 2048 * 2048; i *= 2) {
        l = pow(i, 2);
        n_Bytes = l * sizeof(double);

        // Заполнение матриц aMatrix и bMatrix случайными значениями 
        matrix_random(aMatrix, l);
        matrix_random(bMatrix, l);

        cudaEvent_t begin, stop;
        cudaEventCreate(&begin);
        cudaEventCreate(&stop);

        // Копируем данные с хоста на девайс

        cudaEventRecord(begin, 0);
        cudaMemcpy(aMatrDev, aMatrix, n_Bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(bMatrDev, bMatrix, n_Bytes, cudaMemcpyHostToDevice);

        // Запускаем ядра CUDA и выполняем вычисления

        matmul_gpu << < blocks, threads >> > (aMatrDev, bMatrDev, cMatrDev, sqrt(l));

        cudaMemcpy(cMatrix, cMatrDev, n_Bytes, cudaMemcpyDeviceToHost);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuTime, begin, stop);

        out << "Размерность матриц: " << i << "x" << i << "; CPU_Time: " << gpuTime * 1000 << " us" << std::endl;
        std::cout << "Размерность матриц: " << i << "x" << i << "; CPU_Time: " << gpuTime * 1000 << " us" << std::endl;

        cudaEventDestroy(begin);
        cudaEventDestroy(stop);
    }
    cudaFree(aMatrDev);
    cudaFree(bMatrDev);
    cudaFree(cMatrDev);
    out.close();
    return 0;
}



