# TEST1

Source : 27 fichiers WAV 
Durée totale : 8'22"

Best avg-gen-loss=1.036 
Epoch 159 
Step : 19-33-52-73-75-159


# TEST2

Source : 53 fichiers WAV 
Durée totale : 16'04"

Best avg-gen-loss=0.819
Epoch 182
Step : 16-25-31-33-42-70-182

# TEST3

Source : 75 fichiers WAV 
Durée totale : 24'27"

Best avg-gen-loss=0.709
Epoch  
Step : 15-22-27-32-66-83


```powershell
    # Utilise le répertoire où tu es déjà
    $folder = Get-Location

    # Charger les fichiers WAV du dossier courant
    $files = Get-ChildItem $folder -Filter *.wav

    $duration = 0
    foreach ($f in $files) {
        $shell = New-Object -ComObject Shell.Application
        $folderObj = $shell.Namespace($f.DirectoryName)
        $fileObj = $folderObj.ParseName($f.Name)

        # Colonne 27 = durée
        $length = $folderObj.GetDetailsOf($fileObj, 27)

        if ($length) {
            $ts = [TimeSpan]::Parse($length)
            $duration += $ts.Ticks
        }
    }

    $total = New-Object TimeSpan $duration
    Write-Output "Durée totale = $($total.ToString())"

```

    !find /content -type f -iname "*.pth"
    !find /content -type f -iname "*.index"

/content/HRVC/models/rvc/training/bollo/bollo.index

    from google.colab import files
    files.download("/content/HRVC/models/rvc/training/bollo/bollo_best.pth")