Remove-Item "kshiked\ui\kshield\causal.py" -ErrorAction SilentlyContinue
Remove-Item "kshiked\ui\kshield\terrain.py" -ErrorAction SilentlyContinue
Remove-Item "kshiked\ui\kshield\simulation.py" -ErrorAction SilentlyContinue
Remove-Item "kshiked\ui\kshield\impact.py" -ErrorAction SilentlyContinue
Remove-Item "kshiked\ui\whatif_workbench.py" -ErrorAction SilentlyContinue
Remove-Item "kshiked\ui\sentinel\policy_chat.py" -ErrorAction SilentlyContinue
Remove-Item "kshiked\ui\kshield\sim" -Recurse -Force -ErrorAction SilentlyContinue
Write-Host "Cleanup complete. Please restart the Sentinel Dashboard."
