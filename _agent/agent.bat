@echo off
setlocal EnableDelayedExpansion

set API_KEY=sk_ber_3p7oSMXGsls4fyOT1sG7ovRaec7oWbMEEenqt_06d0f90501dcadac

:menu
echo.
echo ===== AI FILE AGENT =====
echo 1. List files
echo 2. Read file and ask AI
echo 3. Exit
set /p choice=Choose:

if "%choice%"=="1" goto list
if "%choice%"=="2" goto read
if "%choice%"=="3" exit

goto menu

:list
dir /b
goto menu

:read
set /p filename=Enter file name:

set /p question=What do you want to ask about it?

set /p filecontent=<%filename%

curl -s -X POST "https://api.berget.ai/v1/chat/completions" ^
  -H "Content-Type: application/json" ^
  -H "Authorization: Bearer %API_KEY%" ^
  -d "{\"model\":\"meta-llama/Llama-3.1-8B-Instruct\",\"messages\":[{\"role\":\"system\",\"content\":\"You analyze files.\"},{\"role\":\"user\",\"content\":\"File content: %filecontent% QUESTION: %question%\"}],\"temperature\":0.3,\"max_tokens\":300}"

goto menu