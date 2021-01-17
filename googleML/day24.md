### day 24 static vs dynamic training

- Static model : Offline訓練，訓練一次，然後一直用他去predict，建立簡單、測試方便。也因此我們可以一直tune到我們覺得完美為止。
    - Static model比起dynamic來講還少需要監控，而且可以在上線前先驗證正確性。
- Dynamic model : Online訓練，資料會持續灌入model，也必須持續update model。也因此要持續觀察input資料，以免因小失大、以偏概全。
    - Dynamic model要持續為了新資料更新，而且會花大量時間在監控train跟監控input data上