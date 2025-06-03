
package main

import (
	"fmt"
	"time"
)

func calculate(iterations int, param1 int, param2 int) float64 {
	result := 1.0
	for i := 1; i <= iterations; i++ {
		j := i*param1 - param2
		result -= 1.0 / float64(j)
		j = i*param1 + param2
		result += 1.0 / float64(j)
	}
	return result
}

func main() {
	startTime := time.Now()
	result := calculate(100000000, 4, 1) * 4
	endTime := time.Now()

	fmt.Printf("Result: %.12f\n", result)
	fmt.Printf("Execution Time: %.6f seconds\n", endTime.Sub(startTime).Seconds())
}
