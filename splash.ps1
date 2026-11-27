param(
    [ValidateSet('boot', 'open')]
    [string]$Mode = 'boot'
)

$ErrorActionPreference = 'Stop'
try { $Host.UI.RawUI.WindowTitle = 'Audio Data Transcription Toolkit' } catch {}

function Write-C([string]$Text, [string]$Color = 'Cyan') {
    Write-Host $Text -ForegroundColor $Color
}

function Show-Spin([string]$Label, [int]$Ms = 550) {
    $frames = @('|', '/', '-', '\')
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $i = 0
    while ($sw.ElapsedMilliseconds -lt $Ms) {
        Write-Host -NoNewline ("`r    [{0}]  {1}   " -f $frames[$i % 4], $Label) -ForegroundColor Yellow
        Start-Sleep -Milliseconds 65
        $i++
    }
    Write-Host ("`r    [OK]  {0}          " -f $Label) -ForegroundColor Green
}

function Show-Bar([int]$Ms = 700) {
    $width = 28
    $delay = [Math]::Max(15, [int]($Ms / $width))
    for ($n = 0; $n -le $width; $n++) {
        $pct = [int](100 * $n / $width)
        $bar = ('#' * $n) + ('-' * ($width - $n))
        Write-Host -NoNewline ("`r    [" + $bar + "] " + $pct + "%  ") -ForegroundColor Cyan
        Start-Sleep -Milliseconds $delay
    }
    Write-Host ""
}

function Show-Type([string]$Text, [string]$Color = 'Yellow', [int]$Ms = 20) {
    foreach ($ch in $Text.ToCharArray()) {
        Write-Host -NoNewline $ch -ForegroundColor $Color
        Start-Sleep -Milliseconds $Ms
    }
    Write-Host ""
}

if ($Mode -eq 'open') {
    Write-Host ""
    Write-C "    +------------------------------------------------------------------+" 'DarkCyan'
    Write-C "    |        LOCAL ASR ENGINE  -  MULTI-FORMAT MEDIA  -  READY         |" 'DarkCyan'
    Write-C "    +------------------------------------------------------------------+" 'DarkCyan'
    Write-Host ""
    Write-Host -NoNewline "    " -ForegroundColor Yellow
    Show-Type "Launching Audio Data Transcription Toolkit GUI ..." 'Yellow' 20
    Write-Host ""
    Show-Bar 650
    Write-C "    [OK]  Engine initialized. Opening Desktop / Web UI ..." 'Green'
    Write-Host ""
    exit 0
}

Clear-Host
Write-Host ""
Write-C "          _   _   _ ____ ___ ___    ____    _  _____  _                " 'Cyan'
Write-C "         / \ | | | |  _ \_ _/ _ \  |  _ \  / \|_   _|/ \               " 'Cyan'
Write-C "        / _ \| | | | | | | | | | | | | | |/ _ \ | | / _ \              " 'Cyan'
Write-C "       / ___ \ |_| | |_| | | |_| | | |_| / ___ \| |/ ___ \             " 'Cyan'
Write-C "      /_/   \_\___/|____/___\___/  |____/_/   \_\_/_/   \_\            " 'Cyan'
Write-Host ""
Write-C "           T R A N S C R I P T I O N   &   R E P O R T   T O O L K I T " 'Yellow'
Write-C "           A U D I O   &   V I D E O   ( . W A V / . M P 3 / . M P 4 ) " 'White'
Write-Host ""
Write-C "    +------------------------------------------------------------------+" 'DarkCyan'
Write-C "    |  100% Local Whisper ASR   Multi-CPU / NVIDIA CUDA / AMD ROCm     |" 'DarkGray'
Write-C "    |  WAV / MP3 / MP4 / M4A / FLAC / MKV  ->  .TXT / .DOCX / .SRT     |" 'DarkGray'
Write-C "    +------------------------------------------------------------------+" 'DarkCyan'
Write-Host ""
Show-Spin "Probing CPU & GPU (CUDA / ROCm / DirectML) topology ..." 550
Show-Spin "Initializing High-Pass + Block-RMS AGC DSP pipeline ..." 500
Show-Spin "Loading Word (.docx) Conversation Report generator ..." 500
Write-Host ""
Write-C "    > ENGINE   faster-whisper (CTranslate2) + PyAV + python-docx" 'DarkGray'
Write-C "    > PRIVACY  100% offline local inference (no cloud upload)" 'DarkGray'
Write-C "    > MEDIA    WAV (16/24-bit & 32-bit float), MP3, MP4, M4A, FLAC, MKV, MOV" 'DarkGray'
Write-Host ""

