# MatMul_CPU_and_GPU
Программа, реализующая перемножения двух квадратных матриц на CPU и GPU

Было реализовано автоматическое заполнение матриц. 
В цикле происходило постепенное увеличение размера матриц с 16x16 до 2048x2048 и их перемножение сначала на CPU, потом на GPU. 

С помощью библиотеки chrono фиксировалось время выполнения операции перемножения матриц.
Время выполнения в us и размерности матриц записывались в файл, откуда позже была взята информация и при помощи Python были построены графики:

![image](https://github.com/DekartVan/MatMul_CPU_and_GPU/assets/60447026/1044b72b-9aef-408c-9bbb-998821b4ba81)

![загружено](https://github.com/DekartVan/MatMul_CPU_and_GPU/assets/60447026/a05de92d-9de3-4c65-9d4b-0033cbad3e3f)

CPU изначально выигрывает в скорости обработки матриц небольшого размера, так как при вычислениях данных матриц на GPU значительное время расходуется на пересылку информации из глобальной памяти на видеодрайвер. 

![загружено (1)](https://github.com/DekartVan/MatMul_CPU_and_GPU/assets/60447026/dda09a80-d015-42cd-bb9d-3b7c09a5bfdc)

Но при увеличении размерности матриц -> GPU справляется заметно быстрее. При размерности 2048x2048 производительность выше в 187 раз.
