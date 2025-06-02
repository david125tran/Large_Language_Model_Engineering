
import Foundation

func calculate(iterations: Int, param1: Double, param2: Double) -> Double {
    var result = 1.0
    for i in 1...iterations {
        let j1 = Double(i) * param1 - param2
        let j2 = Double(i) * param1 + param2
        result -= 1.0 / j1
        result += 1.0 / j2
    }
    return result
}

let startTime = CFAbsoluteTimeGetCurrent()
let result = calculate(iterations: 100_000_000, param1: 4.0, param2: 1.0) * 4.0
let endTime = CFAbsoluteTimeGetCurrent()

print(String(format: "Result: %.12f", result))
print(String(format: "Execution Time: %.6f seconds", endTime - startTime))
