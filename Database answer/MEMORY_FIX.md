# Fix Memory Allocation Error

## Problem
You're getting: `memory allocation of 8388608 bytes failed`

This happens when the system runs out of memory while loading the embedding model or processing PDFs.

## Solutions

### Solution 1: Close Other Applications
Close unnecessary applications to free up RAM before starting the server.

### Solution 2: Use Lazy Loading (Recommended)
Modify the code to load the embedding model only when needed, not at startup.

### Solution 3: Reduce Batch Size
Process PDFs in smaller batches to reduce memory usage.

### Solution 4: Use a Smaller Model
Switch to an even smaller embedding model (though `all-MiniLM-L6-v2` is already quite small).

### Solution 5: Process One PDF at a Time
Upload and process PDFs one at a time instead of multiple simultaneously.

## Quick Fix: Restart and Try Again

Sometimes memory gets fragmented. Try:

1. **Close the server** (Ctrl+C)
2. **Restart your computer** (if possible)
3. **Close other applications** (browsers, IDEs, etc.)
4. **Start the server again**

## Check Available Memory

On Windows PowerShell:
```powershell
Get-CimInstance Win32_OperatingSystem | Select-Object FreePhysicalMemory, TotalVisibleMemorySize
```

This shows available vs total memory in KB.

## Alternative: Use Cloud/Remote Processing

If memory is consistently an issue, consider:
- Running the backend on a machine with more RAM
- Using cloud services (AWS, Google Cloud, etc.)
- Processing PDFs in smaller chunks

## Code Changes Applied

I've updated the code to:
1. Force CPU usage (avoids GPU memory issues)
2. Add better error handling
3. Add memory clearing after model load
4. Add initialization progress messages

Try running the server again with these changes.

