
public class FastCalculation {

    public static double calculate(int iterations, double param1, double param2) {
        double result = 1.0;
        for (int i = 1; i <= iterations; i++) {
            double j = i * param1 - param2;
            result -= (1.0 / j);
            j = i * param1 + param2;
            result += (1.0 / j);
        }
        return result;
    }

    public static void main(String[] args) {
        long startTime = System.nanoTime();
        double result = calculate(100_000_000, 4, 1) * 4;
        long endTime = System.nanoTime();

        System.out.printf("Result: %.12f%n", result);
        System.out.printf("Execution Time: %.6f seconds%n", (endTime - startTime) / 1_000_000_000.0);
    }
}
