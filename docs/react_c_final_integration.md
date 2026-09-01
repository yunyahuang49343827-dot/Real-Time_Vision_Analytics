# React C｜Responsive、UX Polish 與最終整合

## 範圍

本階段只調整 React Dashboard、顯示用 API metadata 與 Supervision visualization layer。YOLO、既有 ByteTrack、Trajectory、Event、Evidence 與 Analytics 核心邏輯均未修改。

## Responsive 策略

- Desktop / Laptop 使用寬版主內容區；KPI 使用可換行 grid。
- 主影片採 `aspect-video`、`width: 100%` 與 `object-fit: contain`，避免變形和水平溢出。
- 分析摘要在寬螢幕位於影片右側，窄畫面移至影片下方。
- Analytics cards 與 charts 使用 responsive grid 與 `min-width: 0`。
- Event Review 在寬螢幕為左右 workspace，窄畫面為上下排列。
- 窄畫面以功能選單取代固定側欄。

## 狀態與安全呈現

介面提供「尚未分析、分析中、無交通資料、無事件、無 Evidence、Backend unavailable、API timeout、Artifact unavailable、Job FAILED」的繁體中文狀態。後端錯誤只顯示安全的使用者訊息，不顯示 traceback。

INFO、WARNING、CRITICAL、REVIEW_REQUIRED、DETECTED、CONFIRMED 使用一致的 badge 色彩語意。Tracking / Heatmap 使用 segmented control，切換既有 artifact，不建立新 Job，也不重新執行推論。

## 工程資訊

工程資訊集中顯示 Job ID、runtime model、SHA256、profile、imgsz、confidence、device、FPS、resolution、frame count、codec、processing time 與 API status。一般 Overview 不重複顯示這些技術欄位。

## Zone / ROI

Zone Engine 未修改。Tracking visualization 只在 rendering layer 將 polygon 改為細線、低彩度並以低透明度混合，讓 bbox、Track ID 與 trajectory 保持主視覺。

## 驗證界線

瀏覽器 smoke test 驗證 Upload、standard / aerial profile、progress、完成狀態、Tracking / Heatmap、Analytics、Events / Evidence empty handling、新增分析、寬窄視窗與 console。畫面驗證不改變任何 CV 或事件規則結果。
