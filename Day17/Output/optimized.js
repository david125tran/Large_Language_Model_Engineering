
function calculate(iterations, param1, param2) {
    let result = 1.0;
    for (let i = 1; i <= iterations; i++) {
        let j = i * param1 - param2;
        result -= 1 / j;
        j = i * param1 + param2;
        result += 1 / j;
    }
    return result;
}

let startTime = performance.now();
let result = calculate(100000000, 4, 1) * 4;
let endTime = performance.now();

console.log(`Result: ${result.toFixed(12)}`);
console.log(`Execution Time: ${(endTime - startTime).toFixed(6)} milliseconds`);
