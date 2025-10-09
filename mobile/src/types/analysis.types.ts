export interface AnalysisRequest {
  imageUri: string;
  notes?: string;
}

export interface AnalysisResult {
  id: string;
  prediction: 'BENIGN' | 'MALIGNANT';
  confidence: number;
  processingTime: number;
  analysisDate: string;
  imageInfo: ImageInfo;
  userNotes?: string;
  isBookmarked: boolean;
  tags: string[];
}

export interface ImageInfo {
  originalName: string;
  dimensions: {
    width: number;
    height: number;
  };
  fileSize: number;
  mimeType: string;
}

export interface MLResults {
  prediction: 'BENIGN' | 'MALIGNANT';
  confidence: number;
  processingTime: number;
  modelVersion: string;
  features: number[];
  rawOutput: number;
}

export interface AnalysisHistory {
  analyses: AnalysisResult[];
  totalCount: number;
  page: number;
  pageSize: number;
}
