
#include <math.h>

Func Calculate(ByVal iterations As Integer, ByVal param1 As Double, ByVal param2 As Double)
    Local $result = 1.0
    For $i = 1 To iterations + 1
        Local $j = $i * $param1 - $param2
        $result -= 1 / $j
        $j = $i * $param1 + $param2
        $result += 1 / $j
    Next
    Return $result * 4
EndFunc

Local $startTime = DllCall("kernel32.dll", "GetTickCount64", "")["return"]
Local $result = Calculate(100000000, 4, 1)
Local $endTime = DllCall("kernel32.dll", "GetTickCount64", "")["return"]

MsgBox(0, "Result", Format($result, ".12f"))
MsgBox(0, "Execution Time", Format(($endTime - $startTime) / 1000, ".6f") & " seconds")
