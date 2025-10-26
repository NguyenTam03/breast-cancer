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
  imageUrl?: string;  // URL to fetch the image
  userNotes?: string;
  isBookmarked: boolean;
  tags: string[];
  analysisType?: 'image' | 'features';  // Type of analysis
  featuresInfo?: {  // Information about features used (for feature-based analysis)
    featureCount: number;
    selectedFeatures: Array<{
      index: number;
      name: string;
      description: string;
      display_order: number;
    }>;
    featureValues: number[];
  };
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
