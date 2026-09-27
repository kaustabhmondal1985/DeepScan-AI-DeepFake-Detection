import type { AnalysisResult } from '@/lib/deepscan-api'

function escapeHtml(value: unknown): string {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;')
}

function formatPercent(value: number | undefined | null): string {
  return `${(Number(value ?? 0) * 100).toFixed(2)}%`
}

function formatSeconds(value: number | undefined | null): string {
  const seconds = Number(value ?? 0)

  if (!Number.isFinite(seconds)) {
    return '—'
  }

  const minutes = Math.floor(seconds / 60)
  const remainder = seconds - minutes * 60

  if (minutes === 0) {
    return `${remainder.toFixed(2)} sec`
  }

  return `${minutes}m ${remainder.toFixed(2)}s`
}

function assessmentClass(assessment: string): string {
  const normalized = assessment.toLowerCase()

  if (normalized.includes('manipulated')) {
    return 'assessment-red'
  }

  if (normalized.includes('authentic')) {
    return 'assessment-green'
  }

  return 'assessment-amber'
}

export function exportDeepScanReport(result: AnalysisResult): void {
  if (typeof window === 'undefined') {
    return
  }

  const reportWindow = window.open(
    '',
    '_blank',
    'width=1100,height=900',
  )

  if (!reportWindow) {
    window.alert(
      'DeepScan could not open the report window. Please allow pop-ups for this site and try again.',
    )
    return
  }

  const generatedAt = new Date().toLocaleString()

  const assessment = result.assessment ?? 'Inconclusive'
  const confidence = formatPercent(result.confidence)

  const realProbability = formatPercent(
    result.probabilities?.real ?? 0,
  )

  const fakeProbability = formatPercent(
    result.probabilities?.fake ?? 0,
  )

  const video = result.video

  const predictions = Array.isArray(result.frame_predictions)
    ? result.frame_predictions
    : []

  const limitations = Array.isArray(result.limitations)
    ? result.limitations
    : []

  const model = result.model

  const reportId =
    (result as AnalysisResult & { analysis_id?: string }).analysis_id ??
    'DeepScan Analysis'

  const predictionRows =
    predictions.length > 0
      ? predictions
          .map((frame, index) => {
            const timestamp =
              typeof frame.timestamp === 'number'
                ? formatSeconds(frame.timestamp)
                : '—'

            const frameReal = formatPercent(
              frame.probabilities?.real ?? 0,
            )

            const frameFake = formatPercent(
              frame.probabilities?.fake ?? 0,
            )

            const prediction =
              frame.prediction ??
              frame.assessment ??
              '—'

            return `
              <tr>
                <td>${index + 1}</td>
                <td>${escapeHtml(timestamp)}</td>
                <td>${escapeHtml(prediction)}</td>
                <td>${frameReal}</td>
                <td>${frameFake}</td>
              </tr>
            `
          })
          .join('')
      : `
        <tr>
          <td colspan="5" class="empty">
            No frame-level predictions were returned.
          </td>
        </tr>
      `

  const limitationRows =
    limitations.length > 0
      ? limitations
          .map(
            (item) => `
              <li>${escapeHtml(item)}</li>
            `,
          )
          .join('')
      : `
          <li>No additional limitations were returned by the analysis.</li>
        `

  const reason = result.reason ?? ''

  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta
  name="viewport"
  content="width=device-width, initial-scale=1.0"
/>

<title>DeepScan Forensic Analysis Report</title>

<style>
  * {
    box-sizing: border-box;
  }

  body {
    margin: 0;
    padding: 32px;
    background: #f3f4f6;
    color: #111827;
    font-family:
      Inter,
      ui-sans-serif,
      system-ui,
      -apple-system,
      BlinkMacSystemFont,
      "Segoe UI",
      sans-serif;
    line-height: 1.5;
  }

  .toolbar {
    width: 210mm;
    max-width: 100%;
    margin: 0 auto 20px;
    display: flex;
    justify-content: flex-end;
  }

  .print-button {
    border: none;
    border-radius: 8px;
    padding: 10px 16px;
    background: #111827;
    color: white;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
  }

  .report {
    width: 210mm;
    min-height: 297mm;
    max-width: 100%;
    margin: 0 auto;
    background: white;
    padding: 18mm;
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.08);
  }

  .header {
    display: flex;
    justify-content: space-between;
    gap: 24px;
    border-bottom: 2px solid #111827;
    padding-bottom: 18px;
    margin-bottom: 24px;
  }

  .brand {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -0.5px;
  }

  .subtitle {
    margin-top: 3px;
    color: #6b7280;
    font-size: 13px;
  }

  .report-meta {
    text-align: right;
    font-size: 12px;
    color: #6b7280;
  }

  .report-meta strong {
    color: #111827;
  }

  .section {
    margin-top: 28px;
    break-inside: avoid;
  }

  .section-title {
    margin: 0 0 12px;
    padding-bottom: 7px;
    border-bottom: 1px solid #e5e7eb;
    font-size: 16px;
    font-weight: 750;
  }

  .assessment-box {
    border: 1px solid #d1d5db;
    border-radius: 12px;
    padding: 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
  }

  .assessment-label {
    color: #6b7280;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.7px;
  }

  .assessment {
    margin-top: 4px;
    font-size: 25px;
    font-weight: 800;
  }

  .assessment-red {
    color: #b91c1c;
  }

  .assessment-green {
    color: #047857;
  }

  .assessment-amber {
    color: #b45309;
  }

  .confidence {
    text-align: right;
  }

  .confidence-value {
    font-size: 26px;
    font-weight: 800;
  }

  .metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-top: 14px;
  }

  .metric {
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 14px;
  }

  .metric-label {
    color: #6b7280;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }

  .metric-value {
    margin-top: 4px;
    font-size: 20px;
    font-weight: 750;
  }

  .summary {
    background: #f9fafb;
    border-left: 4px solid #374151;
    padding: 14px 16px;
    border-radius: 6px;
    font-size: 13px;
  }

  .info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    overflow: hidden;
  }

  .info-item {
    padding: 11px 13px;
    border-bottom: 1px solid #e5e7eb;
  }

  .info-item:nth-child(odd) {
    border-right: 1px solid #e5e7eb;
  }

  .info-label {
    color: #6b7280;
    font-size: 11px;
  }

  .info-value {
    margin-top: 2px;
    font-size: 13px;
    font-weight: 600;
    word-break: break-word;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 11px;
  }

  th {
    text-align: left;
    padding: 9px;
    background: #f3f4f6;
    border-bottom: 1px solid #d1d5db;
    font-weight: 700;
  }

  td {
    padding: 8px 9px;
    border-bottom: 1px solid #e5e7eb;
  }

  .empty {
    text-align: center;
    color: #6b7280;
    padding: 20px;
  }

  .limitations {
    margin: 0;
    padding-left: 20px;
    font-size: 12px;
  }

  .limitations li {
    margin-bottom: 6px;
  }

  .footer {
    margin-top: 36px;
    padding-top: 12px;
    border-top: 1px solid #d1d5db;
    display: flex;
    justify-content: space-between;
    gap: 16px;
    font-size: 10px;
    color: #6b7280;
  }

  .disclaimer {
    margin-top: 18px;
    padding: 12px 14px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    border-radius: 8px;
    font-size: 11px;
    color: #78350f;
  }

  @page {
    size: A4;
    margin: 14mm;
  }

  @media print {
    body {
      padding: 0;
      background: white;
    }

    .toolbar {
      display: none;
    }

    .report {
      width: auto;
      min-height: auto;
      max-width: none;
      padding: 0;
      box-shadow: none;
    }
  }

  @media (max-width: 800px) {
    body {
      padding: 12px;
    }

    .metric-grid {
      grid-template-columns: 1fr;
    }

    .info-grid {
      grid-template-columns: 1fr;
    }

    .info-item:nth-child(odd) {
      border-right: none;
    }

    .header {
      flex-direction: column;
    }

    .report-meta {
      text-align: left;
    }

    .assessment-box {
      flex-direction: column;
      align-items: flex-start;
    }

    .confidence {
      text-align: left;
    }
  }
</style>
</head>

<body>

<div class="toolbar">
  <button class="print-button" onclick="window.print()">
    Save / Print PDF
  </button>
</div>

<main class="report">

  <header class="header">
    <div>
      <div class="brand">DeepScan</div>
      <div class="subtitle">
        AI-Assisted Deepfake Video Analysis Report
      </div>
    </div>

    <div class="report-meta">
      <div>
        <strong>Report ID:</strong>
        ${escapeHtml(reportId)}
      </div>

      <div>
        <strong>Generated:</strong>
        ${escapeHtml(generatedAt)}
      </div>
    </div>
  </header>

  <section class="section">
    <h2 class="section-title">
      Analysis Assessment
    </h2>

    <div class="assessment-box">
      <div>
        <div class="assessment-label">
          Overall assessment
        </div>

        <div class="assessment ${assessmentClass(assessment)}">
          ${escapeHtml(assessment)}
        </div>
      </div>

      <div class="confidence">
        <div class="assessment-label">
          Confidence
        </div>

        <div class="confidence-value">
          ${confidence}
        </div>
      </div>
    </div>

    <div class="metric-grid">

      <div class="metric">
        <div class="metric-label">
          Real probability
        </div>

        <div class="metric-value">
          ${realProbability}
        </div>
      </div>

      <div class="metric">
        <div class="metric-label">
          Fake probability
        </div>

        <div class="metric-value">
          ${fakeProbability}
        </div>
      </div>

      <div class="metric">
        <div class="metric-label">
          Frames analyzed
        </div>

        <div class="metric-value">
          ${escapeHtml(result.frames_analyzed ?? 0)}
        </div>
      </div>

    </div>
  </section>

  ${
    reason
      ? `
  <section class="section">
    <h2 class="section-title">
      Analysis Summary
    </h2>

    <div class="summary">
      ${escapeHtml(reason)}
    </div>
  </section>
  `
      : ''
  }

  <section class="section">
    <h2 class="section-title">
      Source Video
    </h2>

    <div class="info-grid">

      <div class="info-item">
        <div class="info-label">
          Duration
        </div>

        <div class="info-value">
          ${formatSeconds(video?.duration_seconds)}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Resolution
        </div>

        <div class="info-value">
          ${escapeHtml(video?.resolution ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Frame rate
        </div>

        <div class="info-value">
          ${
            typeof video?.fps === 'number'
              ? `${video.fps.toFixed(2)} FPS`
              : '—'
          }
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Total frames
        </div>

        <div class="info-value">
          ${escapeHtml(video?.total_frames ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Codec
        </div>

        <div class="info-value">
          ${escapeHtml(video?.codec ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Faces detected
        </div>

        <div class="info-value">
          ${escapeHtml(result.preprocessing?.faces_detected ?? 0)}
        </div>
      </div>

    </div>
  </section>

  <section class="section">
    <h2 class="section-title">
      Methodology & Model
    </h2>

    <div class="info-grid">

      <div class="info-item">
        <div class="info-label">
          Model
        </div>

        <div class="info-value">
          ${escapeHtml(model?.name ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Architecture
        </div>

        <div class="info-value">
          ${escapeHtml(model?.architecture ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Input size
        </div>

        <div class="info-value">
          ${escapeHtml(model?.input_size ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Embedding dimension
        </div>

        <div class="info-value">
          ${escapeHtml(model?.embedding_dimension ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Frames per video
        </div>

        <div class="info-value">
          ${escapeHtml(model?.frames_per_video ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Aggregation
        </div>

        <div class="info-value">
          ${escapeHtml(model?.aggregation ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Checkpoint
        </div>

        <div class="info-value">
          ${escapeHtml(model?.checkpoint ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Checkpoint epoch
        </div>

        <div class="info-value">
          ${escapeHtml(model?.checkpoint_epoch ?? '—')}
        </div>
      </div>

      <div class="info-item">
        <div class="info-label">
          Validation balanced accuracy
        </div>

        <div class="info-value">
          ${
            typeof model?.checkpoint_val_balanced_accuracy === 'number'
              ? formatPercent(
                  model.checkpoint_val_balanced_accuracy,
                )
              : '—'
          }
        </div>
      </div>

    </div>
  </section>

  <section class="section">
    <h2 class="section-title">
      Frame-Level Evidence
    </h2>

    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Timestamp</th>
          <th>Prediction</th>
          <th>Real</th>
          <th>Fake</th>
        </tr>
      </thead>

      <tbody>
        ${predictionRows}
      </tbody>
    </table>
  </section>

  <section class="section">
    <h2 class="section-title">
      Limitations
    </h2>

    <ul class="limitations">
      ${limitationRows}
    </ul>

    <div class="disclaimer">
      DeepScan provides AI-assisted forensic screening and should not
      be treated as definitive proof of authenticity or manipulation.
      Results should be interpreted together with the available
      evidence and appropriate human review.
    </div>
  </section>

  <footer class="footer">
    <span>
      DeepScan — AI-Assisted Media Forensics
    </span>

    <span>
      Generated ${escapeHtml(generatedAt)}
    </span>
  </footer>

</main>

</body>
</html>
`

  reportWindow.document.open()
  reportWindow.document.write(html)
  reportWindow.document.close()

  setTimeout(() => {
    reportWindow.focus()
    reportWindow.print()
  }, 500)
}