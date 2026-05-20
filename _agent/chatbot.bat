@echo off
setlocal EnableDelayedExpansion

set API_KEY=sk_ber_3p7oSMXGsls4fyOT1sG7ovRaec7oWbMEEenqt_06d0f90501dcadac

:chat
set /p USER_INPUT=You: 

if "%USER_INPUT%"=="exit" goto end

curl -s -X POST "https://api.berget.ai/v1/chat/completions" ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer %API_KEY%" ^
  -d "{\"model\":\"meta-llama/Llama-3.1-8B-Instruct\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},{\"role\":\"user\",\"content\":\"%USER_INPUT%\"}],\"temperature\":0.7,\"max_tokens\":200}"

echo.
goto chat

:end
echo Bye!
pause