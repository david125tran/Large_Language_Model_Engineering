#include <iostream>
#include <iomanip>
#include <chrono>

double calculate(int64_t iterations, double param1, double param2) {
    double result = 1.0;
    for (int64_t i = 1; i <= iterations; ++i) {
        double j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }
    return result;
}

int main() {
    auto start_time = std::chrono::high_resolution_clock::now();
    double result = calculate(100'000'000, 4, 1) * 4;
    auto end_time = std::chrono::high_resolution_clock::now();

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Result: " << result << std::endl;
    std::chrono::duration<double> execution_time = end_time - start_time;
    std::cout << "Execution Time: " << execution_time.count() << " seconds" << std::endl;

    return 0;
}
