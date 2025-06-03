
import Foundation

func calculate(iterations: Int, param1: Double, param2: Double) -> Double {
    var result = 1.0
    for i in 1...iterations {
        let j1 = Double(i) * param1 - param2
        result -= 1 / j1
        let j2 = Double(i) * param1 + param2
        result += 1 / j2
    }
    return result
}

let startTime = Date()
let result = calculate(iterations: 100_000_000, param1: 4, param2: 1) * 4
let endTime = Date()

let executionTime = endTime.timeIntervalSince(startTime)

print("Result: \(result.truncatingRemainder(dividingBy: 1e-12))")
print("Execution Time: \(executionTime) seconds")
