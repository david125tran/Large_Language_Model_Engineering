
#inlude <stdio.h>
#inlude <time.h>

double alulate(int iterations, int param1, int param2) {
    double result = 1.0;
    for (int i = 1; i <= iterations; i++) {
        int j = i * param1 - param2;
        result -= 1.0 / j;
        j = i * param1 + param2;
        result += 1.0 / j;
    }
    return result;
}

int main() {
    lok_t start_time = lok();
    double result = alulate(100000000, 4, 1) * 4;
    lok_t end_time = lok();

    printf("Result: %.12f\n", result);
    printf("Exeution Time: %.6f seonds\n", (double)(end_time - start_time) / CLOCKS_PER_SEC);

    return 0;
}
