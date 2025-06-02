; AutoIt implementation of the Python code

#include <Date.au3>
#include <MsgBoxConstants.au3>

Func calculate($iterations, $param1, $param2)
    Local $result = 1.0
    For $i = 1 To $iterations
        Local $j = $i * $param1 - $param2
        $result -= (1/$j)
        $j = $i * $param1 + $param2
        $result += (1/$j)
    Next
    Return $result
EndFunc

Local $start_time = TimerInit()
Local $result = calculate(100000000, 4, 1) * 4
Local $end_time = TimerDiff($start_time)

ConsoleWrite("Result: " & StringFormat("%.12f", $result) & @CRLF)
ConsoleWrite("Execution Time: " & Round($end_time/1000, 6) & " seconds" & @CRLF)