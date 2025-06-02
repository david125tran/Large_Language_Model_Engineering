
#include <stdio.h>
#include <time.h>

double calculate(int iterations, double param1, double param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; i++) {
        double j = i * param1 - param2;
        result -= (1.0 / j);
        j = i * param1 + param2;
        result += (1.0 / j);
    }
    return result;
}

int main() {
    clock_t start_time = clock();
    
    double result = calculate(100000000, 4.0, 1.0) * 4.0;
    
    clock_t end_time = clock();
    double execution_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("Result: %.12f\n", result);
    printf("Execution Time: %.6f seconds\n", execution_time);
    
    return 0;
}
