#include <mpi.h>
#include <algorithm>    
#include <cmath>       
#include <cstdint>
#include <iomanip>      //точность вывода чисел
#include <iostream>  
#include <limits>     
#include <stdexcept>   
#include <string>      
#include <vector>      

//генерация положительного элемента матрицы A по его индексам
static double entry(std::uint64_t i, std::uint64_t j,
                    std::uint64_t n) {
    //кэффициенты для генерации элементов по индексам
    double value = 0.1 + ((i * 15 + j * 31) % 100) / 100.0; 
    //доп. увелич. элементы на главной диагонали
    if (i == j) {
        value += 0.05 * n;
    }
    return value;
}

static void usage(int rank, const char* exe) {
    if (rank == 0) { //проверяем главный MIP процесс
        std::cerr << "Использование: " << exe << " N iterations tolerance\n"
                  << "  N > 0, iterations > 0, tolerance >= 0\n"
                  //при tolerance=0 выполн. фиксир. число итераций
                  << "  tolerance=0: выполнить ровно 'iterations' итераций (режим измерения времени)\n"
                  //при (+) пороге провер. точность нормировки
                  << "  tolerance>0: остановка, когда максимальное отклонение суммы строк или столбцов от 1 не превышает tolerance\n";
    }
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);//иниц. среды MPI

    //rank- номер текущего процесса
    //processes -общее количество MPI-проц.
    int rank = 0, processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    //проверка на кол-во аргументов
    if (argc != 4) {
        usage(rank, argv[0]); 
        MPI_Finalize(); //заверш. MPI
        return 1; //ошибка
    }

    //размер матрицы NхN
    std::size_t n = 0;
    int max_iters = 0;
    //заданная допустимая погрешность
    double tolerance = 0.0;

    try {//проверка на введённые значения

        //размер матрицы(перв. аргемент)
        const std::string ns(argv[1]);
        const unsigned long long parsed_n = std::stoull(ns);
        //проверяем размер матрицы: положит. и удовлетвор. MPI
        if (ns.empty() || ns[0] == '-' || parsed_n == 0 ||
            parsed_n > static_cast<unsigned long long>(
                std::numeric_limits<int>::max())) {
            throw std::invalid_argument("invalid N");
        }
        n = static_cast<std::size_t>(parsed_n);

        //кол-во итераций(2-й аргмент)
        const std::string is(argv[2]);
        const long long parsed_iters = std::stoll(is);
        if (parsed_iters < 1 ||
            parsed_iters > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("invalid iterations");
        }
        max_iters = static_cast<int>(parsed_iters);

        tolerance = std::stod(argv[3]);
        //точность конечна и неотрицательна
        if (!std::isfinite(tolerance) || tolerance < 0.0) {
            throw std::invalid_argument("invalid tolerance");
        }

    } catch (const std::exception& e) {
        //сообщение об ошибке
        if (rank == 0)
            std::cerr << "Invalid arguments: " << e.what() << "\n";
        usage(rank, argv[0]); 
        MPI_Finalize();
        return 1; 
    }

    // число MPI-процессов и ранг текущего процесса
    //в типе size_t для вычисления индексов строк
    const std::size_t p = static_cast<std::size_t>(processes);
    const std::size_t k = static_cast<std::size_t>(rank);
    //базовое число строк для кажд. процесса
    const std::size_t base = n / p;
    const std::size_t extra = n % p; //остаток от распред.
    //количество строк матрицы, выделенных тек. проц.
    const std::size_t local_rows = base + (k < extra ? 1 : 0);
    // номер первой строки матрицы, которую обр. тек. проц.
    const std::size_t first_row = k * base + std::min(k, extra);
    //проверка на переполненние
    if (local_rows != 0 &&
        n > std::numeric_limits<std::size_t>::max() / local_rows) {
        if (rank == 0)
            std::cerr << "Matrix size overflow\n";
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    //a - локальная часть исходной матрицы
    //r -коэффициенты масштабирования локальных строк
    //c - коэффициенты масштабирования столбцов
    //partial -локальные суммы по столбцам
    //sums -глобальные суммы по столбцам
    std::vector<double> a, r, c, partial, sums;
    try {
        a.resize(local_rows * n);
        r.resize(local_rows, 1.0);
        c.resize(n, 1.0);
        partial.resize(n);
        sums.resize(n);

    } catch (const std::bad_alloc&) {
        std::cerr << "Rank " << rank << ": нехватка памяти\n";
        MPI_Abort(MPI_COMM_WORLD, 3);
    }

    //заполнение локального блока исходной матрицы
    for (std::size_t i = 0; i < local_rows; ++i)
        for (std::size_t j = 0; j < n; ++j)
            a[i * n + j] = entry(first_row + i, j, n);

    MPI_Barrier(MPI_COMM_WORLD);//синхронизация процессов

    //фиксир. нач. время выполн.
    const double t0 = MPI_Wtime();
    //счётчик итераций
    int executed = 0;
    //значение ошибки(бескон.)
    double error = std::numeric_limits<double>::infinity();


    //алгоритм Синкхорн—Кноппа
    for (int it = 0; it < max_iters; ++it) {

        //[ЭТАП 1]!! вычисл. коэфф. масштабирования строк
        //r_i = 1 / sum_j(A_ij * c_j)
        for (std::size_t i = 0; i < local_rows; ++i) {
            double sum = 0.0;
            for (std::size_t j = 0; j < n; ++j)
                sum += a[i * n + j] * c[j];
            r[i] = 1.0 / sum;
        }

        //[ЭТАП 2]!!вычисл. лок. сумм столбцов
        //partial_j = sum_i(r_i * A_ij).
        std::fill(partial.begin(), partial.end(), 0.0);
        for (std::size_t i = 0; i < local_rows; ++i)
            for (std::size_t j = 0; j < n; ++j)
                partial[j] += r[i] * a[i * n + j];

        // {ЭТАП 3]!! глоб. суммир. по столбцам
        MPI_Allreduce(//собирает суммы со всех процессов
            partial.data(),      //что передаём
            sums.data(),          // куда записываем результат
            static_cast<int>(n),  //кол-во чисел
            MPI_DOUBLE,            // тип чисел
            MPI_SUM,              //складываем значения
            MPI_COMM_WORLD         // все MPI-процессы
        );

        //[ЭТАП 4]!! обновление коэфф. столбцов
        for (std::size_t j = 0; j < n; ++j)
            //нормировка столбца c_j = 1/sums_j.
            c[j] = 1.0 / sums[j];
        //обнов. числа выполненных итерац.
        executed = it + 1;

        //проверяем точность, если tolerance > 0
        if (tolerance > 0.0) {
            //вычисл. макс. отклонение суммы строки
            double local_error = 0.0;
            for (std::size_t i = 0; i < local_rows; ++i) {
                double row_sum = 0.0;
                for (std::size_t j = 0; j < n; ++j)
                    //суммир. A_ij*c_j.
                    row_sum += a[i * n + j] * c[j];
                //учит. коэф. масштабир. строки
                row_sum *= r[i];
              //максим. откл. суммы строки от 1
                local_error = std::max(
                    local_error,
                    std::abs(row_sum - 1.0)
                );
            }

            //ищем макс. ошибку среди всех процессов
            MPI_Allreduce(
                &local_error,     //ошибка тек. процесса
                &error,           //записываем результат
                1,                
                MPI_DOUBLE,      
                MPI_MAX,         
                MPI_COMM_WORLD   
            );

            //досрочное заверш. при достиж. точности
            if (error <= tolerance)
                break;
        }
    }

    //синхронизация процессов
    MPI_Barrier(MPI_COMM_WORLD);
    //вычисление времени тек. процесса
    const double local_seconds = MPI_Wtime() - t0;
    //время выполнения
    double seconds = 0.0;
    //выбир макс. вреия среди всех процессов и передаём процессу 0
    MPI_Reduce(
        &local_seconds,  
        &seconds,      
        1,              
        MPI_DOUBLE,      
        MPI_MAX,        
        0,               //получатель-процесс 0
        MPI_COMM_WORLD   
    );

    //проверка точности после измерения времени
    // считаем суммы строк и столбцов масштабированной матрицы
    double local_row_error = 0.0;
    //обнул. лок. суммы столбцов
    std::fill(partial.begin(), partial.end(), 0.0);

    for (std::size_t i = 0; i < local_rows; ++i) {
        double row_sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            //элемент матрицы после масштаб-ния
            const double bij = r[i] * a[i * n + j] * c[j];
            row_sum += bij;
            partial[j] += bij;
        }
        //максим. откл. суммы строки от 1-цы
        local_row_error = std::max(
            local_row_error,
            std::abs(row_sum - 1.0)
        );
    }

    //глоб. ошибка нормир. строк
    double row_error = 0.0;
    MPI_Reduce(//самая большая ошибка нормир. строк, берём максим.
        &local_row_error,
        &row_error,        
        1,              
        MPI_DOUBLE,        
        MPI_MAX,        
        0,                
        MPI_COMM_WORLD 
    );

    MPI_Allreduce(
        partial.data(),      
        sums.data(),         
        static_cast<int>(n), 
        MPI_DOUBLE,          
        MPI_SUM,             
        MPI_COMM_WORLD       
    );

    //финальные расчёты и вывод выполняет процесс 0
    if (rank == 0) {
        double col_error = 0.0;
        for (std::size_t j = 0; j < n; ++j)
            col_error = std::max(
                col_error,
                std::abs(sums[j] - 1.0)
            );

        //общая ошибка(по строкам и столбцам)
        const double max_error = std::max(row_error, col_error);
        std::cout << std::setprecision(12)
                  << "N=" << n
                  << " P=" << processes //кол-во MPI процессов
                  << " iterations=" << executed
                  << " time_sec=" << seconds
                  << " row_error=" << row_error
                  << " col_error=" << col_error
                  << " max_error=" << max_error
                  //флаг точности: 1 - достигнута, 0 - не дост.
                  << " converged="
                  << ((tolerance > 0.0 &&
                       max_error <= tolerance) ? 1 : 0)
                  << '\n';
    }
    MPI_Finalize();
    return 0;
}
