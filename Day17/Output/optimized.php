
<?
function calculate($iterations, $param1, $param2) {
    $result = 1.0;
    for ($i = 1; $i <= $iterations; $i++) {
        $j = $i * $param1 - $param2;
        $result -= (1/$j);
        $j = $i * $param1 + $param2;
        $result += (1/$j);
    }
    return $result;
}

$start_time = microtime(true);
$result = calculate(100000000, 4, 1) * 4;
$end_time = microtime(true);

echo "Result: " . number_format($result, 12);
echo "Execution Time: " . number_format($end_time - $start_time, 6) . " seconds";
?>
