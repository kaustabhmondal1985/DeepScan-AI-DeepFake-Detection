const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ||
  "http://127.0.0.1:8000"


// ============================================================
// TYPES
// ============================================================

export interface UploadResponse {
  analysis_id: string
  status: string
  message?: string
}


export interface StatusResponse {
  analysis_id: string
  status: string
  progress?: number
  message?: string
  error?: string
}


export interface FramePrediction {
  frame_index: number
  timestamp: number
  crop_path: string
  face_confidence: number
  probabilities: {
    real: number
    fake: number
  }
}


export interface TrackPrediction {
  track_id: number
  frames_used: number
  timestamps: number[]
  prediction: {
    real: number
    fake: number
  }
}


export interface AnalysisResult {
  assessment: string

  confidence: number

  probabilities: {
    real: number
    fake: number
  }

  frames_analyzed: number

  tracks_analyzed: number

  tracks?: TrackPrediction[]

  model: {
    name: string
    architecture: string
    input_size: string
    embedding_dimension: number
    frames_per_video: number
    aggregation: string
    checkpoint: string
    checkpoint_epoch: number
    checkpoint_val_balanced_accuracy: number
  }

  reason?: string

  frame_predictions: FramePrediction[]

  analysis_id: string

  video: {
    duration_seconds: number
    fps: number
    width: number
    height: number
    resolution: string
    total_frames: number
    codec: string
  }

  preprocessing: {
    frames_analyzed: number
    faces_detected: number
  }

  limitations: string[]
}


export interface AnalysisResultResponse {
  analysis_id: string
  status: string
  result?: AnalysisResult
  error?: string
}


// ============================================================
// UPLOAD VIDEO
// ============================================================

export async function uploadVideo(
  file: File
): Promise<UploadResponse> {

  const formData =
    new FormData()

  formData.append(
    "file",
    file
  )


  const response =
    await fetch(
      `${API_BASE_URL}/api/analysis/analyze`,
      {
        method: "POST",
        body: formData,
      }
    )


  if (!response.ok) {

    let message =
      "Failed to upload video."

    try {

      const errorData =
        await response.json()

      if (errorData?.detail) {

        message =
          typeof errorData.detail ===
          "string"
            ? errorData.detail
            : JSON.stringify(
                errorData.detail
              )
      }

    } catch {
      // Keep default error.
    }

    throw new Error(
      message
    )
  }


  return response.json()
}


// ============================================================
// GET ANALYSIS STATUS
// ============================================================

export async function getAnalysisStatus(
  analysisId: string
): Promise<StatusResponse> {

  const response =
    await fetch(
      `${API_BASE_URL}/api/analysis/${analysisId}/status`,
      {
        method: "GET",
        cache: "no-store",
      }
    )


  if (!response.ok) {

    throw new Error(
      `Failed to get analysis status: ${response.status}`
    )
  }


  return response.json()
}


// ============================================================
// GET ANALYSIS RESULT
// ============================================================

export async function getAnalysisResult(
  analysisId: string
): Promise<AnalysisResultResponse> {

  const response =
    await fetch(
      `${API_BASE_URL}/api/analysis/${analysisId}/result`,
      {
        method: "GET",
        cache: "no-store",
      }
    )


  if (!response.ok) {

    throw new Error(
      `Failed to get analysis result: ${response.status}`
    )
  }


  const data =
    await response.json()


  // ========================================================
  // IMPORTANT
  //
  // The DeepScan backend currently returns the analysis
  // result DIRECTLY:
  //
  // {
  //   "assessment": "...",
  //   "confidence": ...,
  //   "probabilities": {...},
  //   ...
  // }
  //
  // It does NOT necessarily return:
  //
  // {
  //   "status": "completed",
  //   "result": {...}
  // }
  //
  // So support both formats.
  // ========================================================


  // Format 1:
  //
  // {
  //   analysis_id,
  //   status,
  //   result: {...}
  // }

  if (
    data &&
    data.result
  ) {

    return {
      analysis_id:
        data.analysis_id ||
        analysisId,

      status:
        data.status ||
        "completed",

      result:
        data.result,

      error:
        data.error,
    }
  }


  // Format 2:
  //
  // Direct AnalysisResult
  //
  // {
  //   assessment,
  //   confidence,
  //   probabilities,
  //   ...
  // }

  if (
    data &&
    typeof data.assessment ===
      "string"
  ) {

    return {
      analysis_id:
        data.analysis_id ||
        analysisId,

      status:
        "completed",

      result:
        data,
    }
  }


  // Unknown response format.

  return {
    analysis_id:
      analysisId,

    status:
      data?.status ||
      "unknown",

    error:
      data?.error ||
      "Backend returned an unexpected result format.",
  }
}


// ============================================================
// ANALYZE VIDEO
// ============================================================

export async function analyzeVideo(
  file: File,

  onProgress?: (
    status: StatusResponse
  ) => void,

  pollingInterval = 1500
): Promise<AnalysisResult> {


  // ----------------------------------------------------------
  // STEP 1: Upload
  // ----------------------------------------------------------

  const uploadResult =
    await uploadVideo(
      file
    )


  const analysisId =
    uploadResult.analysis_id


  // ----------------------------------------------------------
  // STEP 2: Poll status
  // ----------------------------------------------------------

  while (true) {

    const status =
      await getAnalysisStatus(
        analysisId
      )


    onProgress?.(
      status
    )


    // --------------------------------------------------------
    // Completed
    // --------------------------------------------------------

    if (
      status.status ===
      "completed"
    ) {

      break
    }


    // --------------------------------------------------------
    // Failed
    // --------------------------------------------------------

    if (
      status.status ===
        "failed" ||
      status.status ===
        "error"
    ) {

      throw new Error(
        status.error ||
          status.message ||
          "Video analysis failed."
      )
    }


    // --------------------------------------------------------
    // Wait before polling again
    // --------------------------------------------------------

    await new Promise(
      (resolve) =>
        setTimeout(
          resolve,
          pollingInterval
        )
    )
  }


  // ----------------------------------------------------------
  // STEP 3: Fetch final result
  // ----------------------------------------------------------

  const resultResponse =
    await getAnalysisResult(
      analysisId
    )


  // ----------------------------------------------------------
  // STEP 4: Validate final result
  // ----------------------------------------------------------

  if (
    !resultResponse.result
  ) {

    throw new Error(
      resultResponse.error ||
        "Analysis completed but no result was returned."
    )
  }


  return resultResponse.result
}


// ============================================================
// PERCENTAGE
// ============================================================

export function percentage(
  value: number
): number {

  return Number(
    (
      value * 100
    ).toFixed(2)
  )
}


// ============================================================
// FORMAT TIMESTAMP
// ============================================================

export function formatTimestamp(
  seconds: number
): string {

  const totalSeconds =
    Math.max(
      0,
      Math.floor(
        seconds
      )
    )


  const minutes =
    Math.floor(
      totalSeconds / 60
    )


  const remainingSeconds =
    totalSeconds % 60


  return `${String(
    minutes
  ).padStart(
    2,
    "0"
  )}:${String(
    remainingSeconds
  ).padStart(
    2,
    "0"
  )}`
}