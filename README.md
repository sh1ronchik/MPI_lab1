# Состав команды

Работу выполнили студенты группы **22ПИ1**:

| Участник | Вклад в проект |
|---------|----------------|
| **Овсянников Артём Сергеевич** | Реализовал задания **1** и **2**: вычисление числа π методом Монте-Карло с использованием MPI и параллельное умножение матрицы на вектор тремя способами разбиения данных. |
| **Шейх Руслан Халедович** | Внёс доработки и оптимизации в решения заданий **1** и **2**, а также реализовал параллельный алгоритм умножения матриц по **алгоритму Кэннона** (задание 3). |


# Руководство по настройке MPI проекта

## 1. Установка (Ubuntu)
```bash
sudo apt update
sudo apt install -y openmpi-bin openmpi-doc libopenmpi-dev python3 python3-pip
python3 -m pip install --user matplotlib
```

## 2. Структура проекта
task1/ \
├── scripts/ \
│   ├── task1.c \
│   └── task1_plot.py \
├── data/ \
│    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── csv/             # Вывод в CSV \
│    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── plot/             # Графики в PNG \
task2/ \
├── scripts/ \
│   ├── task2.c \
│   └── task2_plot.py \
├── data/ \
│    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── csv/             # Вывод в CSV \
│    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── plot/             # Графики в PNG \
task3/ \
├── scripts/ \
│   ├── task3.c \
│   └── task3_plot.py \
├── data/ \
│    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── csv/             # Вывод в CSV \
│    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── plot/             # Графики в PNG \

## 3. Компиляция
Task 1
```bash
mpicc -O2 -std=c11 task1/scripts/task1.c -o task1/scripts/task1
```
Task 2
```bash
mpicc -O2 -std=c11 task2/scripts/task2.c -o task2/scripts/task2
```
Task 3
```bash
mpicc -O2 -std=c11 task3/scripts/task3.c -o task3/scripts/task3 -lm
```

## 4. Запуск
Task 1: 
```bash
mpiexec -n 4 ./task1/scripts/task1 10000000
```
Task 2:
```bash
mpiexec -n 4 ./task2/scripts/task2 "100, 500, 1000, 5000, 10000"
```
Task 3:
```bash
mpiexec -n 4 ./task3/scripts/task3 512,1024,2048 task3
```

## 5. Форматы CSV
Task 1:
```
procs,total_points,overall_time,pi
```

Task 2:
```
procs,rows,cols,overall
```

Task 3:
```
procs,N,overall
```


## 6. Построение графиков
Task 1
```bash
python3 task1/scripts/task1_plot.py task1
```

Task 2
```bash
python3 task2/scripts/task2_plot.py --prefix task2
```

Task 3
```bash
python3 task3/scripts/task3_plot.py --prefix task3
```
