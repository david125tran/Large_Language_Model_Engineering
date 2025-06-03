
using System;
using System.Numerics;

class Program
{
    static void Main(string[] args)
    {
        BigInteger iterations = 100_000_000;
        BigInteger param1 = 4;
        BigInteger param2 = 1;

        BigInteger result = 1.0;
        for (BigInteger i = 1; i <= iterations; i++)
        {
            BigInteger j = i * param1 - param2;
            result -= 1 / (BigInteger)j;

            j = i * param1 + param2;
            result += 1 / (BigInteger)j;
        }

        result *= 4;
        Console.WriteLine($"Result: {result:F12}");

        TimeSpan executionTime = TimeSpan.FromSeconds(DateTime.Now.Ticks / TimeSpan.TicksPerSecond - DateTime.UtcNow.Ticks / TimeSpan.TicksPerSecond);
        Console.WriteLine($"Execution Time: {executionTime.TotalSeconds:F6} seconds");
    }
}
