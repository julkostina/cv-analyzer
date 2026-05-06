export type ExperienceItem = {
  employer?: string | null;
  title?: string | null;
  duration?: string | null;
};

export type CertificateItem = {
  name?: string | null;
  institution?: string | null;
  year?: string | null;
};

export type EducationItem = {
  degree?: string | null;
  institution?: string | null;
  year?: string | null;
};

export type ProjectItem = {
  name?: string | null;
  description?: string | null;
  link?: string | null;
};

export type CVAnalysisResponse = {
  success: boolean;
  extracted_text?: string | null;
  analysis?: Record<string, unknown> | null;
  skills?: string[] | null;
  experience?: ExperienceItem[] | null;
  certificates?: CertificateItem[] | null;
  education?: EducationItem[] | null;
  projects?: ProjectItem[] | null;
  match_score?: number | null;
  match_score_reasoning?: string | null;
  recommendations?: string[] | null;
  matched_competencies?: string[] | null;
  missing_competencies?: string[] | null;
  semantic_breakdown?: Record<string, number> | null;
  error?: string | null;
};
