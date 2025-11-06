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
mpicc -O2 -std=c11 task3/scripts/task3.c -o task3/scripts/task3
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
mpiexec -n 4 ./task3/scripts/task3 1024 task3
```

## 5. Форматы CSV
Task 1:
```
procs,total_points,overall_time,comp_max,comm_max,pi
```

Task 2:
```
procs,rows,cols,overall,comp_max,comm_max
```

Task 3:
```
procs,N,overall,comp_max,comm_max
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
