
package main

import (
	"fmt"
	"time"
)

func calculate(iterations int, param1, param2 float64) float64 {
	result := 1.0
	for i := 1; i <= iterations; i++ {
		j1 := float64(i)*param1 - param2
		result -= 1 / j1
		j2 := float64(i)*param1 + param2
		result += 1 / j2
	}
	return result
}

func main() {
	startTime := time.Now()
	result := calculate(100_000_000, 4, 1) * 4
	elapsedTime := time.Since(startTime)

	fmt.Printf("Result: %.12f\n", result)
	fmt.Printf("Execution Time: %.6f seconds\n", elapsedTime.Seconds())
}
