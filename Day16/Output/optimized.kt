
fun calculate(iterations: Int, param1: Double, param2: Double): Double {
    var result = 1.0
    for (i in 1..iterations) {
        val j1 = i * param1 - param2
        result -= 1 / j1
        val j2 = i * param1 + param2
        result += 1 / j2
    }
    return result
}

fun main() {
    val startTime = System.nanoTime()
    val result = calculate(100_000_000, 4.0, 1.0) * 4
    val endTime = System.nanoTime()

    println("Result: %.12f".format(result))
    println("Execution Time: %.6f seconds".format((endTime - startTime) / 1_000_000_000.0))
}
