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
  featureSelection?: FeatureSelection;
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

export interface FeatureInfo {
  name: string;
  displayName: string;
  category: string;
  importance: number;
}

export interface FeatureSelection {
  algorithm: string;
  modelType: string;
  selectedFeatures: FeatureInfo[];
  totalFeatures: number;
  selectedCount: number;
  selectionRatio: number;
}

export interface FeatureComparison {
  commonFeatures: FeatureInfo[];
  rfOnlyFeatures: FeatureInfo[];
  svmOnlyFeatures: FeatureInfo[];
  totalCommon: number;
  rfTotal: number;
  svmTotal: number;
}
