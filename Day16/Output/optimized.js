
function calculate(iterations, param1, param2) {
    let result = 1.0;
    for (let i = 1; i <= iterations; i++) {
        let j1 = i * param1 - param2;
        result -= 1 / j1;
        let j2 = i * param1 + param2;
        result += 1 / j2;
    }
    return result;
}

const start_time = performance.now();
const result = calculate(100_000_000, 4, 1) * 4;
const end_time = performance.now();

console.log(`Result: ${result.toFixed(12)}`);
console.log(`Execution Time: ${(end_time - start_time) / 1000.0.toFixed(6)} seconds`);
