[CmdletBinding()]
param(
    [string]$Python = "python",
    [switch]$CollectCoverage,
    [switch]$OpenReport
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$EvidenceRoot = Join-Path $ProjectRoot "evidence"
$RunDir = Join-Path $EvidenceRoot $Stamp

New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

$JUnit = Join-Path $RunDir "junit.xml"
$HTML = Join-Path $RunDir "pytest_report.html"
$Console = Join-Path $RunDir "test_results.txt"
$GitBefore = Join-Path $RunDir "git_status_before.txt"
$GitAfter = Join-Path $RunDir "git_status_after.txt"
$Summary = Join-Path $RunDir "summary.txt"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "AI TRADER - PHASE 1 TEST RUNNER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Project : $ProjectRoot"
Write-Host "Evidence: $RunDir"
Write-Host ""

# Capture repository state BEFORE tests.
git status --short | Out-File -FilePath $GitBefore -Encoding utf8
$AppBefore = @(git status --short -- app)

# Verify pytest is available.
& $Python -m pytest --version
if ($LASTEXITCODE -ne 0) {
    throw "pytest is not available. Install requirements-test.txt first."
}

$pytestArgs = @(
    "-m", "pytest",
    "-v",
    "--junitxml=$JUnit",
    "--html=$HTML",
    "--self-contained-html"
)

if ($CollectCoverage) {
    $pytestArgs += @(
        "--cov=app",
        "--cov-report=term-missing",
        "--cov-report=html:$RunDir\coverage-html"
    )
}

Write-Host ""
Write-Host "Running tests..." -ForegroundColor Yellow

& $Python @pytestArgs 2>&1 | Tee-Object -FilePath $Console
$TestExitCode = $LASTEXITCODE

# Capture repository state AFTER tests.
git status --short | Out-File -FilePath $GitAfter -Encoding utf8

# Safety check: Phase 1 must not create/modify files under app/.
$AfterApp = @(git status --short -- app)

# The runner itself does not alter app/. A difference here means something
# executed during the test run changed production files.
$AppChanged = ($AppBefore -join "`n") -ne ($AfterApp -join "`n")

$Status = if ($TestExitCode -eq 0 -and -not $AppChanged) {
    "PASS"
} else {
    "FAIL"
}

@"
AI TRADER - PHASE 1 TEST SUMMARY
================================

Timestamp : $Stamp
Status    : $Status
Pytest RC : $TestExitCode
App changed during run : $AppChanged

Evidence:
  Console : $Console
  JUnit   : $JUnit
  HTML    : $HTML
  Git     : $GitAfter
"@ | Out-File -FilePath $Summary -Encoding utf8

Write-Host ""
Write-Host "============================================================"
Write-Host "RESULT: $Status" -ForegroundColor $(if ($Status -eq "PASS") {"Green"} else {"Red"})
Write-Host "Evidence: $RunDir"
Write-Host "============================================================"

if ($OpenReport -and (Test-Path $HTML)) {
    Start-Process $HTML
}

if ($Status -ne "PASS") {
    exit 1
}

exit 0
