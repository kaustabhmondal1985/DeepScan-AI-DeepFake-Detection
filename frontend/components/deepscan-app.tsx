'use client'

import {
  useState,
  type Dispatch,
  type SetStateAction,
} from 'react'

import {
  Activity,
  AlertTriangle,
  ArrowRight,
  AudioLines,
  BarChart3,
  Check,
  ChevronRight,
  CircleDot,
  Clock3,
  Download,
  FileVideo,
  Fingerprint,
  History,
  House,
  Info,
  Layers3,
  Menu,
  Mic2,
  MoreHorizontal,
  Pause,
  Play,
  Plus,
  Radar,
  ScanLine,
  Search,
  ShieldCheck,
  Sparkles,
  UploadCloud,
  UserRound,
  Video,
  Volume2,
  X,
} from 'lucide-react'

import {
  analyzeVideo,
  formatTimestamp,
  percentage,
  type AnalysisResult,
  type StatusResponse,
} from '@/lib/deepscan-api'

import { exportDeepScanReport } from '@/lib/deepscan-report'


// ============================================================
// TYPES
// ============================================================

type PageKey =
  | 'dashboard'
  | 'analyze'
  | 'history'
  | 'methodology'

type AnalysisState =
  | 'idle'
  | 'progress'
  | 'results'


// ============================================================
// APP
// ============================================================

export function DeepScanApp() {
  const [page, setPage] =
    useState<PageKey>('dashboard')

  const [analysis, setAnalysis] =
    useState<AnalysisState>('idle')

  const [file, setFile] =
    useState<File | null>(null)

  const [result, setResult] =
    useState<AnalysisResult | null>(null)

  const [status, setStatus] =
    useState<StatusResponse | null>(null)

  const [error, setError] =
    useState<string | null>(null)

  const [menuOpen, setMenuOpen] =
    useState(false)


  const go = (next: PageKey) => {
    setPage(next)
    setMenuOpen(false)
  }


  const startAnalysis = async () => {
    if (!file) {
      return
    }

    setError(null)
    setResult(null)
    setStatus({
      analysis_id: '',
      status: 'uploading',
      progress: 0,
      message: 'Uploading video...',
    })

    setAnalysis('progress')

    try {
      const analysisResult =
        await analyzeVideo(
          file,
          (nextStatus) => {
            setStatus(nextStatus)
          }
        )

      setResult(analysisResult)
      setAnalysis('results')
    } catch (err) {
      console.error(
        '[DeepScan] Analysis failed:',
        err
      )

      const message =
        err instanceof Error
          ? err.message
          : 'Video analysis failed.'

      setError(message)
      setAnalysis('idle')
    }
  }


  const resetAnalysis = () => {
    setFile(null)
    setResult(null)
    setStatus(null)
    setError(null)
    setAnalysis('idle')
  }


  return (
    <div className="ds-shell">

      {/* ======================================================
          HEADER
      ====================================================== */}

      <header className="topbar">

        <button
          className="brand"
          onClick={() => go('dashboard')}
          aria-label="DeepScan dashboard"
        >
          <span className="brand-mark">
            <Radar
              size={21}
              strokeWidth={1.6}
            />
            <span />
          </span>

          <span>
            <strong>DeepScan</strong>
            <small>MEDIA FORENSICS</small>
          </span>
        </button>


        <nav
          className="main-nav"
          aria-label="Main navigation"
        >
          {(
            [
              [
                'dashboard',
                'Dashboard',
                House,
              ],
              [
                'analyze',
                'Analyze',
                ScanLine,
              ],
              [
                'history',
                'History',
                History,
              ],
              [
                'methodology',
                'Methodology',
                Layers3,
              ],
            ] as const
          ).map(
            ([
              key,
              label,
              Icon,
            ]) => (
              <button
                key={key}
                className={
                  page === key
                    ? 'nav-item active'
                    : 'nav-item'
                }
                onClick={() =>
                  go(key)
                }
              >
                <Icon size={15} />
                {label}
              </button>
            )
          )}
        </nav>


        <div className="top-actions">

          <span className="system-status">
            <i />
            System operational
          </span>

          <button
            className="avatar"
            aria-label="Account menu"
            onClick={() =>
              setMenuOpen(
                !menuOpen
              )
            }
          >
            <span>KT</span>
            <ChevronRight size={14} />
          </button>

          <button
            className="mobile-menu"
            onClick={() =>
              setMenuOpen(
                !menuOpen
              )
            }
            aria-label="Open menu"
          >
            <Menu size={20} />
          </button>

        </div>


        {menuOpen && (
          <div className="account-popover">
            <div className="account-name">
              Kaustabh T.
            </div>

            <div className="muted">
              Analyst workspace
            </div>

            <button
              onClick={() =>
                setMenuOpen(false)
              }
            >
              Close menu
            </button>
          </div>
        )}

      </header>


      {/* ======================================================
          PAGES
      ====================================================== */}

      {page === 'dashboard' && (
        <Dashboard
          onAnalyze={() =>
            go('analyze')
          }
        />
      )}


      {page === 'analyze' && (
        <Analyze
          file={file}
          setFile={setFile}
          analysis={analysis}
          setAnalysis={setAnalysis}
          status={status}
          result={result}
          error={error}
          onStart={startAnalysis}
          onNew={resetAnalysis}
        />
      )}


      {page === 'history' && (
        <HistoryPage
          onOpen={() => {
            setPage('analyze')
          }}
        />
      )}


      {page === 'methodology' && (
        <Methodology />
      )}

    </div>
  )
}


// ============================================================
// DASHBOARD
// ============================================================

function Dashboard({
  onAnalyze,
}: {
  onAnalyze: () => void
}) {
  return (
    <main>

      <section className="hero page-pad">

        <div className="hero-copy">

          <div className="eyebrow">
            <span className="eyebrow-line" />
            MULTIMODAL MEDIA FORENSICS
          </div>

          <h1>
            See beyond
            <br />
            <em>the deepfake.</em>
          </h1>

          <p>
            DeepScan analyzes facial and
            temporal signals to identify
            potentially manipulated media
            and provide traceable evidence
            behind its assessment.
          </p>

          <div className="hero-actions">

            <button
              className="primary-btn"
              onClick={onAnalyze}
            >
              <ScanLine size={17} />
              Analyze a video
              <ArrowRight size={16} />
            </button>

            <button
              className="quiet-btn"
              onClick={() =>
                document
                  .getElementById(
                    'method'
                  )
                  ?.scrollIntoView({
                    behavior:
                      'smooth',
                  })
              }
            >
              Explore methodology
              <ChevronRight size={15} />
            </button>

          </div>

          <div className="hero-note">
            <ShieldCheck size={15} />
            AI-assisted analysis ·
            Designed for investigative review
          </div>

        </div>

        <ForensicPreview />

      </section>


      <section className="stats-strip page-pad">

        <div>
          <strong>04</strong>
          <span>evidence channels</span>
        </div>

        <div>
          <strong>10</strong>
          <span>pipeline stages</span>
        </div>

        <div>
          <strong>01</strong>
          <span>explainable report</span>
        </div>

        <div className="stats-caption">
          WHAT IS SUSPICIOUS
          <br />
          <b>WHERE &amp; WHEN</b>
          <br />
          did it occur?
        </div>

      </section>


      <section className="section page-pad">

        <div className="section-head">

          <div>
            <div className="eyebrow">
              01 / CAPABILITIES
            </div>

            <h2>
              One signal is never enough.
            </h2>
          </div>

          <p>
            DeepScan processes multiple
            evidence channels instead of
            relying on a single visual
            classifier.
          </p>

        </div>


        <div className="cap-grid">

          <Capability
            icon={Fingerprint}
            num="01"
            title="Visual intelligence"
            model="RETINAFACE + XCEPTION"
            text="Analyze facial regions and visual manipulation artifacts."
          />

          <Capability
            icon={Activity}
            num="02"
            title="Temporal analysis"
            model="FRAME-LEVEL AGGREGATION"
            text="Compare predictions across sampled video frames."
          />

          <Capability
            icon={AudioLines}
            num="03"
            title="Voice intelligence"
            model="AUDIO ANALYSIS"
            text="Reserved for the multimodal extension of DeepScan."
          />

          <Capability
            icon={Layers3}
            num="04"
            title="Cross-modal analysis"
            model="MULTIMODAL FUSION"
            text="Planned extension combining independent evidence channels."
          />

        </div>

      </section>


      <section
        id="method"
        className="process-section page-pad"
      >

        <div className="section-head">

          <div>
            <div className="eyebrow">
              02 / PROCESS
            </div>

            <h2>
              From upload to evidence.
            </h2>
          </div>

          <p>
            Every assessment is built from
            a traceable sequence of analysis
            stages.
          </p>

        </div>


        <div className="process-line">

          {[
            'Upload',
            'Extract',
            'Analyze',
            'Correlate',
            'Explain',
            'Report',
          ].map(
            (x, i) => (
              <div
                className="process-step"
                key={x}
              >
                <span>
                  0{i + 1}
                </span>

                <b>{x}</b>

                {i < 5 && <i />}
              </div>
            )
          )}

        </div>

      </section>


      <section className="trust-section page-pad">

        <div className="trust-copy">

          <div className="eyebrow">
            03 / TRUST &amp; LIMITATIONS
          </div>

          <h2>
            Your media is evidence.
            <br />
            <em>Treat it accordingly.</em>
          </h2>

          <p>
            DeepScan is an AI-assisted
            screening system. Results may
            contain false positives and false
            negatives.
          </p>

          <span className="trust-label">
            <Info size={14} />
            Not definitive proof of manipulation
          </span>

        </div>


        <div className="comparison">

          <div>
            <span>
              TRADITIONAL DETECTOR
            </span>

            <strong>
              Is it fake?
            </strong>
          </div>

          <div className="compare-arrow">
            <ArrowRight />
          </div>

          <div className="highlight">
            <span>DEEPSCAN</span>

            <strong>
              What is suspicious?
            </strong>

            <small>
              Where · When · Why
            </small>
          </div>

        </div>

      </section>

    </main>
  )
}


// ============================================================
// FORENSIC PREVIEW
// ============================================================

function ForensicPreview() {
  return (
    <div className="forensic-preview">

      <div className="preview-top">
        <span>
          <CircleDot size={12} />
          LIVE EVIDENCE VIEW
        </span>

        <span>
          DS / DEMO
        </span>
      </div>


      <div className="video-frame">

        <div className="frame-grid" />

        <div className="scan-beam" />

        <div className="face-box">
          <span>
            FACE 01 · DETECTED
          </span>
          <i />
        </div>

        <div className="frame-meta">
          <span>FRAME 00428</span>
          <span>00:14.72</span>
        </div>

        <div className="frame-label label-one">
          VISUAL ANALYSIS
          <b>ACTIVE</b>
        </div>

        <div className="frame-label label-two">
          FACE DETECTED
          <b>01</b>
        </div>

        <div className="subject">
          <div className="subject-head" />
          <div className="subject-body" />
        </div>

      </div>


      <div className="preview-timeline">

        <span>00:00</span>

        <div>
          <i />
          <b />
          <b className="warning" />
          <b />
          <b className="warning second" />
        </div>

        <span>01:24</span>

      </div>


      <div className="preview-foot">

        <span>
          <i className="cyan-dot" />
          FACE DETECTION
        </span>

        <span>
          <i className="red-dot" />
          FRAME ANALYSIS
        </span>

      </div>

    </div>
  )
}


// ============================================================
// CAPABILITY
// ============================================================

function Capability({
  icon: Icon,
  num,
  title,
  model,
  text,
}: {
  icon: typeof Fingerprint
  num: string
  title: string
  model: string
  text: string
}) {
  return (
    <div className="cap-card">

      <div className="cap-top">
        <span className="icon-box">
          <Icon size={18} />
        </span>

        <span>{num}</span>
      </div>

      <h3>{title}</h3>

      <code>{model}</code>

      <p>{text}</p>

      <ArrowRight
        size={16}
        className="card-arrow"
      />

    </div>
  )
}


// ============================================================
// ANALYZE PAGE
// ============================================================

function Analyze({
  file,
  setFile,
  analysis,
  setAnalysis,
  status,
  result,
  error,
  onStart,
  onNew,
}: {
  file: File | null
  setFile: (f: File | null) => void
  analysis: AnalysisState
  setAnalysis: (
    s: AnalysisState
  ) => void
  status: StatusResponse | null
  result: AnalysisResult | null
  error: string | null
  onStart: () => void
  onNew: () => void
}) {

  if (
    analysis === 'progress'
  ) {
    return (
      <ProgressScreen
        file={file}
        status={status}
      />
    )
  }


  if (
    analysis === 'results' &&
    result
  ) {
    return (
      <Results
        result={result}
        onNew={onNew}
      />
    )
  }


  return (
    <main className="page-pad app-page">

      <div className="page-title">

        <div>

          <div className="eyebrow">
            ANALYSIS WORKSPACE / NEW SCAN
          </div>

          <h1>
            Start a <em>DeepScan.</em>
          </h1>

          <p>
            Upload a video to begin
            AI-assisted forensic analysis.
          </p>

        </div>


        <span className="scan-id">
          <span className="status-dot" />
          READY TO RECEIVE
        </span>

      </div>


      {error && (
        <div
          className="trust-banner"
          style={{
            marginBottom:
              '20px',
          }}
        >
          <AlertTriangle
            size={17}
          />

          <div>
            <b>
              Analysis failed
            </b>

            <span>
              {error}
            </span>
          </div>
        </div>
      )}


      <div className="upload-layout">

        <div>

          <UploadDropzone
            file={file}
            setFile={setFile}
          />

          <div className="format-note">
            <FileVideo size={14} />

            MP4 / MOV / AVI / WEBM

            <span>·</span>

            Maximum file size configured
            by workspace
          </div>

        </div>


        <div className="scan-options">

          <div className="eyebrow">
            SELECT SCAN DEPTH
          </div>

          <ScanOption
            title="Quick scan"
            text="Visual frame analysis"
            meta="Rapid signal check"
            icon={ScanLine}
          />

          <ScanOption
            selected
            title="Deep forensic scan"
            text="RetinaFace + fine-tuned Xception"
            meta="Current backend model"
            icon={Radar}
          />

        </div>

      </div>


      <div className="trust-banner">

        <ShieldCheck size={17} />

        <div>

          <b>
            AI-assisted analysis
          </b>

          <span>
            Results are probabilistic and
            should not be used as the sole
            basis for legal or high-stakes
            decisions.
          </span>

        </div>

      </div>


      <button
        className="primary-btn analyze-btn"
        disabled={!file}
        onClick={onStart}
      >
        <ScanLine size={17} />

        Begin deep forensic scan

        <ArrowRight size={16} />

      </button>

    </main>
  )
}


// ============================================================
// UPLOAD DROPZONE
// ============================================================

function UploadDropzone({
  file,
  setFile,
}: {
  file: File | null
  setFile: (
    f: File | null
  ) => void
}) {

  const [drag, setDrag] =
    useState(false)


  const handleFile = (
    selectedFile: File | null
  ) => {

    if (!selectedFile) {
      return
    }

    setFile(selectedFile)
  }


  return (
    <div
      className={
        drag
          ? 'upload-card drag'
          : 'upload-card'
      }

      onDragOver={(e) => {
        e.preventDefault()
        setDrag(true)
      }}

      onDragLeave={() =>
        setDrag(false)
      }

      onDrop={(e) => {
        e.preventDefault()
        setDrag(false)

        handleFile(
          e.dataTransfer.files[0] ||
            null
        )
      }}
    >

      <input
        id="video-upload"
        type="file"
        accept="video/mp4,video/quicktime,video/x-msvideo,video/webm"
        onChange={(e) =>
          handleFile(
            e.target.files?.[0] ||
              null
          )
        }
      />


      <div className="upload-orbit">
        <UploadCloud size={25} />
      </div>


      {file ? (
        <>

          <h3>
            {file.name}
          </h3>

          <p>
            {(
              file.size /
              1024 /
              1024
            ).toFixed(1)}
            {' MB · Ready for analysis'}
          </p>

          <button
            className="remove-file"
            onClick={(e) => {
              e.stopPropagation()
              setFile(null)
            }}
          >
            <X size={14} />
            Remove file
          </button>

        </>
      ) : (
        <>

          <h3>
            Drop video evidence here
          </h3>

          <p>
            or select a file from your
            device
          </p>

          <label
            htmlFor="video-upload"
            className="browse-btn"
          >
            Browse files
            <ArrowRight size={14} />
          </label>

        </>
      )}

    </div>
  )
}


// ============================================================
// SCAN OPTION
// ============================================================

function ScanOption({
  title,
  text,
  meta,
  icon: Icon,
  selected,
}: {
  title: string
  text: string
  meta: string
  icon: typeof Radar
  selected?: boolean
}) {
  return (
    <div
      className={
        selected
          ? 'scan-option selected'
          : 'scan-option'
      }
    >

      <span className="radio">
        {selected && <i />}
      </span>

      <Icon size={18} />

      <div>
        <b>{title}</b>
        <span>{text}</span>
        <small>{meta}</small>
      </div>

      {selected && (
        <Check
          size={15}
          className="option-check"
        />
      )}

    </div>
  )
}


// ============================================================
// PROGRESS SCREEN
// ============================================================

function ProgressScreen({
  file,
  status,
}: {
  file: File | null
  status: StatusResponse | null
}) {

  const progress =
    typeof status?.progress === 'number'
      ? Math.min(
          100,
          Math.max(
            0,
            status.progress
          )
        )
      : 0


  const currentMessage =
    status?.message ||
    'Preparing video analysis...'


  return (
    <main className="page-pad app-page progress-page">

      <div className="page-title">

        <div>

          <div className="eyebrow">
            ANALYSIS JOB
            {status?.analysis_id
              ? ` / ${status.analysis_id}`
              : ''}
          </div>

          <h1>
            Analyzing <em>evidence.</em>
          </h1>

          <p>
            {file?.name ||
              'Selected video'}
            {' · Deep forensic scan'}
          </p>

        </div>


        <span className="scan-id">

          <span className="pulse-dot" />

          PROCESSING

        </span>

      </div>


      <div className="progress-layout">

        <div className="processing-visual">

          <ForensicPreview />


          <div className="progress-readout">

            <div>

              <span>
                OVERALL PROGRESS
              </span>

              <strong>
                {progress}
                <small>%</small>
              </strong>

            </div>


            <div className="progress-bar">

              <i
                style={{
                  width:
                    `${progress}%`,
                }}
              />

            </div>


            <p>
              {currentMessage}
            </p>

          </div>

        </div>


        <div className="stage-list">

          <div className="stage-head">

            <span>
              PIPELINE STATUS
            </span>

            <b>
              {progress}%
            </b>

          </div>


          {[
            'Video preprocessing',
            'Face detection',
            'Visual analysis',
            'Temporal aggregation',
            'Result generation',
          ].map(
            (stage, index) => {

              const completed =
                progress >=
                ((index + 1) /
                  5) *
                  100

              const current =
                !completed &&
                progress >=
                  (index /
                    5) *
                    100

              return (
                <div
                  className={
                    completed
                      ? 'stage done'
                      : current
                        ? 'stage current'
                        : 'stage'
                  }
                  key={stage}
                >

                  <span className="stage-index">

                    {completed ? (
                      <Check
                        size={12}
                      />
                    ) : (
                      String(
                        index + 1
                      ).padStart(
                        2,
                        '0'
                      )
                    )}

                  </span>

                  <b>{stage}</b>

                  <small>
                    {completed
                      ? 'Completed'
                      : current
                        ? 'Processing'
                        : 'Pending'}
                  </small>

                </div>
              )
            }
          )}

        </div>

      </div>

    </main>
  )
}


// ============================================================
// RESULTS
// ============================================================

function Results({
  result,
  onNew,
}: {
  result: AnalysisResult
  onNew: () => void
}) {

  const [playing, setPlaying] =
    useState(false)


  const real =
    percentage(
      result.probabilities.real
    )

  const fake =
    percentage(
      result.probabilities.fake
    )

  const confidence =
    percentage(
      result.confidence
    )


  const assessmentClass =
    result.assessment
      .toLowerCase()
      .includes('manipulated')
      ? 'red'
      : result.assessment
          .toLowerCase()
          .includes('authentic')
        ? 'green'
        : 'amber'


  return (
    <main className="page-pad results-page">

      {/* ====================================================
          RESULT HEADER
      ==================================================== */}

      <div className="result-head">

        <div>

          <div className="eyebrow">
            ANALYSIS RESULT
            {' / '}
            {result.analysis_id}
          </div>

          <h1>
            {result.assessment}
          </h1>

          <p>
            AI-assisted model assessment ·
            Not definitive proof of
            manipulation
          </p>

        </div>


        <div className="result-actions">

          <button
            className="quiet-btn"
            onClick={onNew}
          >
            <Plus size={15} />
            New analysis
          </button>

          <button
            className="primary-btn"
            onClick={() =>
              exportDeepScanReport(result)
            }
          >
            <Download size={15} />
            Export report
          </button>

        </div>

      </div>


      {/* ====================================================
          SCORE OVERVIEW
      ==================================================== */}

      <div className="score-overview">

        <div className="overall-score">

          <span>
            MODEL CONFIDENCE
          </span>

          <strong>
            {confidence}
            <small>%</small>
          </strong>

          <div className="meter">

            <i
              style={{
                width:
                  `${confidence}%`,
              }}
            />

          </div>

          <b
            className={`${assessmentClass}-text`}
          >
            {result.assessment}
          </b>

        </div>


        <Modality
          title="Real probability"
          value={`${real}%`}
          status="Model probability"
          color="green"
          icon={ShieldCheck}
        />


        <Modality
          title="Fake probability"
          value={`${fake}%`}
          status="Model probability"
          color={
            fake >= 50
              ? 'red'
              : 'amber'
          }
          icon={AlertTriangle}
        />


        <Modality
          title="Frames analyzed"
          value={String(
            result.frames_analyzed
          )}
          status="Sampled video frames"
          color="cyan"
          icon={Video}
        />


        <Modality
          title="Faces detected"
          value={String(
            result.preprocessing
              .faces_detected
          )}
          status="RetinaFace detections"
          color="cyan"
          icon={Fingerprint}
        />

      </div>


      {/* ====================================================
          RESULTS GRID
      ==================================================== */}

      <div className="results-grid">

        <div className="evidence-main">

          <div className="result-card viewer-card">

            <div className="card-heading">

              <div>

                <span className="eyebrow">
                  ANALYSIS SUMMARY
                </span>

                <h2>
                  Video evidence
                </h2>

              </div>

              <span className="timecode">
                {result.video.duration_seconds.toFixed(
                  2
                )}
                {' sec'}
              </span>

            </div>


            <div className="large-viewer">

              <div className="frame-grid" />

              <div className="scan-beam" />

              <div className="viewer-face">

                <span>
                  FACE 01 · DETECTED
                </span>

              </div>

              <div className="viewer-subject">
                <div className="subject-head" />
                <div className="subject-body" />
              </div>

              <span className="viewer-timestamp">
                Analysis complete
              </span>

            </div>


            <div className="viewer-controls">

              <button
                onClick={() =>
                  setPlaying(
                    !playing
                  )
                }
                aria-label={
                  playing
                    ? 'Pause'
                    : 'Play'
                }
              >
                {playing ? (
                  <Pause size={16} />
                ) : (
                  <Play
                    size={16}
                    fill="currentColor"
                  />
                )}
              </button>

              <div className="seek">

                <i
                  style={{
                    width:
                      `${confidence}%`,
                  }}
                />

              </div>

              <span>
                {formatTimestamp(
                  result.video
                    .duration_seconds
                )}
              </span>

              <Volume2 size={15} />

              <MoreHorizontal
                size={18}
              />

            </div>


            <div className="overlay-toggles">

              <label>
                <input
                  type="checkbox"
                  defaultChecked
                />
                <span>
                  Face detection
                </span>
              </label>

              <label>
                <input
                  type="checkbox"
                  defaultChecked
                />
                <span>
                  Frame predictions
                </span>
              </label>

            </div>

          </div>


          <EvidenceTimeline
            result={result}
          />

        </div>


        <div className="side-stack">

          <WhyFlagged
            result={result}
          />

          <Faces
            result={result}
          />

          <ModelCard
            result={result}
          />

        </div>

      </div>


      {/* ====================================================
          LOWER GRID
      ==================================================== */}

      <div className="lower-grid">

        <FramePredictionCard
          result={result}
        />

        <VideoMetadataCard
          result={result}
        />

      </div>


      {/* ====================================================
          LIMITATIONS
      ==================================================== */}

      <div className="report-cta">

        <div>

          <span className="eyebrow">
            REVIEW NOTES
          </span>

          <h2>
            Assessment ready for review.
          </h2>

          <p>
            {result.reason ||
              'Prediction generated from sampled face crops using the fine-tuned Xception model.'}
          </p>

          <div
            style={{
              marginTop:
                '12px',
              display:
                'flex',
              flexDirection:
                'column',
              gap: '5px',
            }}
          >

            {result.limitations.map(
              (limitation) => (
                <small
                  key={limitation}
                >
                  • {limitation}
                </small>
              )
            )}

          </div>

        </div>


        <button
          className="primary-btn"
          onClick={() =>
            exportDeepScanReport(result)
          }
        >
          <Download size={16} />
          Export report
        </button>

      </div>

    </main>
  )
}


// ============================================================
// MODALITY CARD
// ============================================================

function Modality({
  title,
  value,
  status,
  color,
  icon: Icon,
}: {
  title: string
  value: string
  status: string
  color: string
  icon: typeof Activity
}) {

  return (
    <div className="modality">

      <Icon
        size={17}
        className={`modality-icon ${color}`}
      />

      <span>{title}</span>

      <strong>{value}</strong>

      <small>{status}</small>

    </div>
  )
}


// ============================================================
// EVIDENCE TIMELINE
// ============================================================

function EvidenceTimeline({
  result,
}: {
  result: AnalysisResult
}) {

  const predictions =
    result.frame_predictions


  return (
    <div className="timeline-card">

      <div className="card-heading">

        <div>

          <span className="eyebrow">
            FRAME-LEVEL SIGNALS
          </span>

          <h2>
            Sampled prediction timeline
          </h2>

        </div>

        <span className="legend">
          <i />
          sampled frame
        </span>

      </div>


      <div className="timeline">

        <span>
          00:00
        </span>

        <div className="timeline-track">

          {predictions.map(
            (prediction, index) => {

              const duration =
                Math.max(
                  result.video
                    .duration_seconds,
                  1
                )

              const left =
                Math.min(
                  96,
                  Math.max(
                    2,
                    (
                      prediction.timestamp /
                      duration
                    ) *
                      100
                  )
                )

              return (
                <i
                  key={`${prediction.frame_index}-${index}`}
                  className={
                    prediction
                      .probabilities
                      .fake >= 0.5
                      ? 'event one'
                      : 'event'
                  }
                  style={{
                    left:
                      `${left}%`,
                  }}
                  title={`${formatTimestamp(
                    prediction.timestamp
                  )} · Fake ${percentage(
                    prediction
                      .probabilities
                      .fake
                  )}%`}
                />
              )
            }
          )}

        </div>

        <span>
          {formatTimestamp(
            result.video
              .duration_seconds
          )}
        </span>

      </div>


      <div className="event-list">

        {predictions.map(
          (prediction) => {

            const fake =
              percentage(
                prediction
                  .probabilities
                  .fake
              )

            return (
              <div
                className="event-row"
                key={
                  prediction.frame_index
                }
              >

                <span
                  className={
                    fake >= 50
                      ? 'event-time red'
                      : 'event-time amber'
                  }
                >
                  {formatTimestamp(
                    prediction.timestamp
                  )}
                </span>


                <div>

                  <b>
                    Frame{' '}
                    {prediction.frame_index}
                  </b>

                  <small>
                    Face confidence:{' '}
                    {percentage(
                      prediction
                        .face_confidence
                    )}
                    %
                  </small>

                </div>


                <strong>
                  Fake {fake}%
                </strong>

                <ChevronRight
                  size={15}
                />

              </div>
            )
          }
        )}

      </div>

    </div>
  )
}


// ============================================================
// WHY FLAGGED
// ============================================================

function WhyFlagged({
  result,
}: {
  result: AnalysisResult
}) {

  const highestFake =
    [...result.frame_predictions]
      .sort(
        (a, b) =>
          b.probabilities.fake -
          a.probabilities.fake
      )
      .slice(0, 3)


  return (
    <div className="result-card why-card">

      <div className="card-heading">

        <div>

          <span className="eyebrow">
            MODEL EVIDENCE
          </span>

          <h2>
            Highest fake signals
          </h2>

        </div>

        <Sparkles
          size={17}
          className="cyan"
        />

      </div>


      {highestFake.map(
        (frame) => {

          const fake =
            percentage(
              frame
                .probabilities
                .fake
            )

          return (
            <div
              className="evidence-item"
              key={
                frame.frame_index
              }
            >

              <span
                className={
                  fake >= 50
                    ? 'severity red'
                    : 'severity amber'
                }
              />

              <div>

                <b>
                  Frame{' '}
                  {frame.frame_index}
                </b>

                <small>
                  {formatTimestamp(
                    frame.timestamp
                  )}
                  {' · '}
                  Fake probability
                  {' '}
                  {fake}%
                </small>

                <p>
                  The model assigned
                  this sampled face
                  crop a fake
                  probability of{' '}
                  {fake}%.
                </p>

              </div>

            </div>
          )
        }
      )}


      <div className="text-btn">
        Frame-level model
        evidence
        <ArrowRight size={14} />
      </div>

    </div>
  )
}


// ============================================================
// FACES
// ============================================================

function Faces({
  result,
}: {
  result: AnalysisResult
}) {

  const faceCount =
    result.preprocessing
      .faces_detected


  return (
    <div className="result-card faces-card">

      <div className="card-heading">

        <div>

          <span className="eyebrow">
            FACE DETECTION
          </span>

          <h2>
            Detected faces
          </h2>

        </div>

        <span className="muted">
          {faceCount} detections
        </span>

      </div>


      <div className="face-row">

        <div className="face-thumb">
          <UserRound size={22} />
        </div>

        <div>

          <b>
            Primary face
          </b>

          <small>
            RetinaFace detections
          </small>

        </div>

        <strong>
          {faceCount}
        </strong>

      </div>


      <div className="face-row">

        <div className="face-thumb">
          <Fingerprint size={22} />
        </div>

        <div>

          <b>
            Frames analyzed
          </b>

          <small>
            Sampled for Xception
          </small>

        </div>

        <strong>
          {result.frames_analyzed}
        </strong>

      </div>

    </div>
  )
}


// ============================================================
// MODEL CARD
// ============================================================

function ModelCard({
  result,
}: {
  result: AnalysisResult
}) {

  return (
    <div className="result-card cross-card">

      <div className="card-heading">

        <div>

          <span className="eyebrow">
            MODEL
          </span>

          <h2>
            Analysis model
          </h2>

        </div>

        <Layers3
          size={17}
          className="cyan"
        />

      </div>


      <div className="cross-score">

        <strong>
          {result.model
            .checkpoint_val_balanced_accuracy *
            100
          }
          <small>%</small>
        </strong>

        <span>
          Validation balanced accuracy
        </span>

      </div>


      <div className="signal-path">

        <span>
          RetinaFace
        </span>

        <i />

        <span>
          Xception
        </span>

        <i />

        <span>
          MLP classifier
        </span>

        <i />

        <span>
          Mean probability
        </span>

      </div>


      <div className="mismatch">

        <Info size={14} />

        {result.model.name}

      </div>

    </div>
  )
}


// ============================================================
// FRAME PREDICTIONS
// ============================================================

function FramePredictionCard({
  result,
}: {
  result: AnalysisResult
}) {

  return (
    <div className="result-card audio-card">

      <div className="card-heading">

        <div>

          <span className="eyebrow">
            MODEL OUTPUT
          </span>

          <h2>
            Frame predictions
          </h2>

        </div>

        <strong className="audio-score">
          {result.frames_analyzed}
        </strong>

      </div>


      <div
        style={{
          display:
            'flex',
          flexDirection:
            'column',
          gap: '10px',
          marginTop:
            '18px',
        }}
      >

        {result.frame_predictions.map(
          (frame) => {

            const fake =
              percentage(
                frame
                  .probabilities
                  .fake
              )

            const real =
              percentage(
                frame
                  .probabilities
                  .real
              )

            return (
              <div
                key={
                  frame.frame_index
                }
                style={{
                  display:
                    'grid',
                  gridTemplateColumns:
                    '55px 70px 1fr 80px',
                  alignItems:
                    'center',
                  gap:
                    '12px',
                }}
              >

                <span>
                  {formatTimestamp(
                    frame.timestamp
                  )}
                </span>

                <small>
                  F{frame.frame_index}
                </small>

                <div
                  className="progress-bar"
                >
                  <i
                    style={{
                      width:
                        `${fake}%`,
                    }}
                  />
                </div>

                <strong>
                  {fake}% fake
                </strong>

              </div>
            )
          }
        )}

      </div>


      <div className="audio-labels">

        <span>
          Real probability:
          {' '}
          {percentage(
            result.probabilities.real
          )}
          %
        </span>

        <span>
          Fake probability:
          {' '}
          {percentage(
            result.probabilities.fake
          )}
          %
        </span>

      </div>

    </div>
  )
}


// ============================================================
// VIDEO METADATA
// ============================================================

function VideoMetadataCard({
  result,
}: {
  result: AnalysisResult
}) {

  return (
    <div className="result-card heatmap-card">

      <div className="card-heading">

        <div>

          <span className="eyebrow">
            VIDEO METADATA
          </span>

          <h2>
            Source information
          </h2>

        </div>

        <FileVideo size={17} />

      </div>


      <div
        style={{
          display:
            'grid',
          gridTemplateColumns:
            '1fr 1fr',
          gap:
            '14px',
          marginTop:
            '20px',
        }}
      >

        <MetadataItem
          label="Resolution"
          value={
            result.video.resolution
          }
        />

        <MetadataItem
          label="Duration"
          value={`${result.video.duration_seconds}s`}
        />

        <MetadataItem
          label="FPS"
          value={String(
            result.video.fps
          )}
        />

        <MetadataItem
          label="Total frames"
          value={String(
            result.video.total_frames
          )}
        />

        <MetadataItem
          label="Codec"
          value={
            result.video.codec
          }
        />

        <MetadataItem
          label="Frames analyzed"
          value={String(
            result.frames_analyzed
          )}
        />

      </div>


      <p>
        Metadata is extracted from the
        uploaded source video. No identity
        information is inferred.
      </p>

    </div>
  )
}


function MetadataItem({
  label,
  value,
}: {
  label: string
  value: string
}) {

  return (
    <div>

      <small
        style={{
          display:
            'block',
          marginBottom:
            '4px',
        }}
      >
        {label}
      </small>

      <strong>
        {value}
      </strong>

    </div>
  )
}


// ============================================================
// HISTORY
// ============================================================

function HistoryPage({
  onOpen,
}: {
  onOpen: () => void
}) {

  const rows = [
    [
      'DS-2026-041',
      'interview_sample.mp4',
      '21 Sep 2026',
      'Potentially Manipulated',
      '89.4%',
      '01:24',
    ],
    [
      'DS-2026-038',
      'press_statement.mov',
      '19 Sep 2026',
      'Likely Authentic',
      '12.8%',
      '00:48',
    ],
    [
      'DS-2026-034',
      'conference_clip.mp4',
      '16 Sep 2026',
      'Inconclusive',
      '51.2%',
      '02:07',
    ],
    [
      'DS-2026-029',
      'voice_note.webm',
      '12 Sep 2026',
      'Potentially Manipulated',
      '76.0%',
      '00:36',
    ],
  ]


  return (
    <main className="page-pad app-page">

      <div className="page-title">

        <div>

          <div className="eyebrow">
            ARCHIVE / ANALYSIS HISTORY
          </div>

          <h1>
            Evidence <em>archive.</em>
          </h1>

          <p>
            Review previous scans and
            generated forensic reports.
          </p>

        </div>

        <button
          className="primary-btn"
          onClick={onOpen}
        >
          <Plus size={16} />
          New analysis
        </button>

      </div>


      <div className="history-toolbar">

        <div className="filter-tabs">

          <button className="selected">
            All scans
            <b>24</b>
          </button>

          <button>
            Potentially manipulated
            <b>8</b>
          </button>

          <button>
            Likely authentic
            <b>12</b>
          </button>

          <button>
            Inconclusive
            <b>4</b>
          </button>

        </div>


        <label className="search-box">

          <Search size={15} />

          <input
            placeholder="Search scan ID or file"
          />

        </label>

      </div>


      <div className="history-table">

        <div className="table-head">

          <span>SCAN ID</span>
          <span>FILE</span>
          <span>DATE</span>
          <span>ASSESSMENT</span>
          <span>CONFIDENCE</span>
          <span>DURATION</span>
          <span />

        </div>


        {rows.map(
          (r, i) => (

            <button
              className="table-row"
              key={r[0]}
              onClick={
                i === 0
                  ? onOpen
                  : undefined
              }
            >

              <span className="mono">
                {r[0]}
              </span>

              <span className="file-cell">
                <FileVideo size={15} />
                {r[1]}
              </span>

              <span>{r[2]}</span>

              <span>

                <i
                  className={`status-pill ${
                    i === 1
                      ? 'green'
                      : i === 2
                        ? 'amber'
                        : 'red'
                  }`}
                />

                {r[3]}

              </span>

              <span className="confidence">
                {r[4]}
              </span>

              <span>
                {r[5]}
              </span>

              <span>
                <MoreHorizontal
                  size={17}
                />
              </span>

            </button>
          )
        )}

      </div>


      <div className="empty-hint">

        <Clock3 size={16} />

        Showing recent analyses ·
        Retention follows workspace policy

      </div>

    </main>
  )
}


// ============================================================
// METHODOLOGY
// ============================================================

function Methodology() {

  const cards = [
    [
      'RetinaFace',
      'Face detection & alignment',
      'Detects faces across sampled frames without identifying people.',
    ],
    [
      'Xception',
      'Spatial feature extraction',
      'Analyzes facial crops for visual manipulation signals.',
    ],
    [
      'Frame aggregation',
      'Video-level prediction',
      'Aggregates predictions from sampled face crops.',
    ],
    [
      'Audio',
      'Planned extension',
      'Audio analysis is part of the planned multimodal extension.',
    ],
    [
      'Multimodal fusion',
      'Planned extension',
      'Future version will combine independent modality signals.',
    ],
    [
      'Explainability',
      'Evidence reporting',
      'Frame-level predictions provide traceable model evidence.',
    ],
  ]


  return (
    <main className="page-pad methodology-page">

      <div className="page-title">

        <div>

          <div className="eyebrow">
            SYSTEM / METHODOLOGY
          </div>

          <h1>
            How DeepScan <em>thinks.</em>
          </h1>

          <p>
            A transparent view of the
            processing architecture behind
            each assessment.
          </p>

        </div>

        <span className="scan-id">
          <Info size={14} />
          FINE-TUNED XCEPTION
        </span>

      </div>


      <div className="architecture">

        <div className="arch-column">

          <span className="arch-label">
            VIDEO PATH
          </span>

          {[
            'INPUT VIDEO',
            'FRAME EXTRACTION',
            'RETINAFACE',
            'XCEPTION',
            'FRAME AGGREGATION',
          ].map(
            (x, i) => (
              <div
                className="arch-node"
                key={x}
              >
                <b>{x}</b>
                {i < 4 && <i />}
              </div>
            )
          )}

        </div>


        <div className="arch-column audio-path">

          <span className="arch-label">
            FUTURE AUDIO PATH
          </span>

          {[
            'INPUT VIDEO',
            'AUDIO EXTRACTION',
            'AUDIO FEATURES',
            'AUDIO MODEL',
            'AUDIO EMBEDDING',
          ].map(
            (x, i) => (
              <div
                className="arch-node"
                key={x}
              >
                <b>{x}</b>
                {i < 4 && <i />}
              </div>
            )
          )}

        </div>


        <div className="fusion-node">

          <Layers3 size={22} />

          <span>
            MULTIMODAL FUSION
          </span>

          <i />

          <b>
            AUDIO-VISUAL CONSISTENCY
          </b>

          <i />

          <strong>
            EVIDENCE → REPORT
          </strong>

        </div>

      </div>


      <div className="section-head model-head">

        <div>

          <div className="eyebrow">
            MODEL COMPONENTS
          </div>

          <h2>
            Signals, not verdicts.
          </h2>

        </div>

        <p>
          Model outputs are evidence
          signals. No individual component
          is treated as definitive.
        </p>

      </div>


      <div className="model-grid">

        {cards.map(
          ([title, sub, text], i) => (

            <div
              className="model-card"
              key={title}
            >

              <span>
                0{i + 1}
              </span>

              <div className="model-icon">

                {i === 0 ? (
                  <Fingerprint />
                ) : i === 1 ? (
                  <ScanLine />
                ) : i === 2 ? (
                  <Activity />
                ) : i === 3 ? (
                  <Mic2 />
                ) : i === 4 ? (
                  <Layers3 />
                ) : (
                  <Sparkles />
                )}

              </div>

              <h3>{title}</h3>

              <code>{sub}</code>

              <p>{text}</p>

            </div>
          )
        )}

      </div>


      <div className="method-note">

        <ShieldCheck size={18} />

        <div>

          <b>
            Interpretability is a
            first-class output.
          </b>

          <span>
            DeepScan surfaces frame-level
            model evidence alongside
            confidence so reviewers can
            understand what influenced an
            assessment.
          </span>

        </div>

      </div>

    </main>
  )
}