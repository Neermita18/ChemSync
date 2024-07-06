# Parameters
$BATCH_SIZE = 300  # Number of files per commit
$BRANCH_NAME = "main"

# Get the list of all files
$FILES = git ls-files -o --exclude-standard

# Counter for files
$count = 0

# Array to hold files for a single commit
$batch = @()

# Function to commit and push the current batch of files
function Commit-And-Push {
    param (
        [array]$batch
    )
    
    if ($batch.Count -ne 0) {
        git add $batch
        git commit -m "Batch commit of $($BATCH_SIZE) files"
        git push origin $BRANCH_NAME
        $batch = @()  # Reset batch array
    }
}

# Loop through all files and batch them
foreach ($file in $FILES) {
    $batch += $file
    $count++

    if ($count -ge $BATCH_SIZE) {
        Commit-And-Push -batch $batch
        $count = 0
    }
}

# Commit any remaining files
Commit-And-Push -batch $batch

Write-Output "All batches committed and pushed successfully."