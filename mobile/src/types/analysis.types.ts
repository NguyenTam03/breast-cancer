export interface AnalysisRequest {
  imageUri: string;
  notes?: string;
}

export interface FeatureAnalysisRequest {
  featureData: Record<string, number>;
  useGWO?: boolean;
  notes?: string;
}

export interface AnalysisResult {
  id: string;
  prediction: 'BENIGN' | 'MALIGNANT';
  confidence: number;
  processingTime: number;
  analysisDate: string;
  imageInfo?: ImageInfo;
  imageUrl?: string;  // URL to fetch the image
  userNotes?: string;
  isBookmarked: boolean;
  tags: string[];
  // New fields for feature analysis
  method?: string;
  featuresUsed?: number;
  inputFeatures?: Record<string, number>;
  useGWO?: boolean;
  analysisType?: 'image' | 'features';
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

export interface FeatureInfo {
  name: string;
  description: string;
  default_value: number;
  is_required: boolean;
}

export interface FeaturesInfoResponse {
  success: boolean;
  features: FeatureInfo[];
  totalFeatures: number;
  description: string;
}

export interface ComparisonResult {
  success: boolean;
  analysisDate: string;
  results: {
    image_prediction?: {
      prediction: 'BENIGN' | 'MALIGNANT';
      confidence: number;
      processing_time: number;
      method: string;
      error?: string;
    };
    feature_prediction?: {
      prediction: 'BENIGN' | 'MALIGNANT';
      confidence: number;
      processing_time: number;
      method: string;
      features_used: number;
      error?: string;
    };
    comparison?: {
      agreement: boolean;
      confidence_difference: number;
      average_confidence: number;
    };
  };
  notes?: string;
}

export interface AnalysisHistory {
  analyses: AnalysisResult[];
  totalCount: number;
  page: number;
  pageSize: number;
}
