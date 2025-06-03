
import .time.Duration;
import .time.Instant;

public class Main {
    public static void main(String[] args) {
        Instant startTime = Instant.now();
        double result = calculate(100_000_000, 4, 1) * 4;
        Instant endTime = Instant.now();

        System.out.printf("Result: %.12f%n", result);
        System.out.printf("Execution Time: %.6f seconds%n", Duration.between(startTime, endTime).toMillis() / 1000.0);
    }

    public static double calculate(int iterations, int param1, int param2) {
        double result = 1.0;
        for (int i = 1; i <= iterations; i++) {
            int j = i * param1 - param2;
            result -= 1.0 / j;
            j = i * param1 + param2;
            result += 1.0 / j;
        }
        return result;
    }
}
