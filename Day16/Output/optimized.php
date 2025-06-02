


function calculate($iterations, $param1, $param2) {
    $result = 1.0;
    for ($i = 1; $i <= $iterations; $i++) {
        $j1 = $i * $param1 - $param2;
        $result -= (1 / $j1);
        $j2 = $i * $param1 + $param2;
        $result += (1 / $j2);
    }
    return $result;
}

$start_time = microtime(true);
$result = calculate(100000000, 4, 1) * 4;
$end_time = microtime(true);

printf("Result: %.12f\n", $result);
printf("Execution Time: %.6f seconds\n", $end_time - $start_time);
