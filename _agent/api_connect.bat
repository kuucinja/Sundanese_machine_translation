@echo off
setlocal

set API_KEY=sk_ber_3p7oSMXGsls4fyOT1sG7ovRaec7oWbMEEenqt_06d0f90501dcadac

echo Using key: %API_KEY%


curl -X POST "https://api.berget.ai/v1/chat/completions" ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer %API_KEY%" ^
  --data-binary "@request.json"

pause
endlocal
