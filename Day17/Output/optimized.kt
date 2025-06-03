
import .system.measureTimeMillis

fun calculate(iterations: Int, param1: Int, param2: Int): Double {
    var result = 1.0
    for (i in 1..iterations) {
        val j1 = i * param1 - param2
        result -= 1.0 / j1
        val j2 = i * param1 + param2
        result += 1.0 / j2
    }
    return result
}

fun main() {
    val startTime = System.currentTimeMillis()
    val result = calculate(100_000_000, 4, 1) * 4
    val endTime = System.currentTimeMillis()

    println("Result: ${"%.12f".format(result)}")
    println("Execution Time: ${(endTime - startTime).toDouble() / 1000} seconds")
}
