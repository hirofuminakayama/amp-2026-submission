from pydantic import BaseModel, ConfigDict, Field


class NumericSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    count: int
    mean: float
    standard_deviation: float
    minimum: float
    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    maximum: float


class CandidateEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    rank: int = Field(ge=1)
    candidate_id: str
    sequence: str
    final_score: float
    apex_vote16: float = Field(ge=0, le=1)
    apex_consensus16: float = Field(ge=0, le=1)
    gram_negative_success16: float = Field(ge=0, le=1)
    gram_positive_success16: float = Field(ge=0, le=1)
    mdr_proxy: float = Field(ge=0, le=1)
    apex_broad_mean_mic_u_m: float = Field(gt=0)
    apex_model_disagreement: float = Field(ge=0)
    embedding_cluster: int
    pep_hard_filter_pass: bool
    spps_difficulty_score: float = Field(ge=0)
    hemopi2_hc50_u_m: float | None = Field(default=None, gt=0)
    hemopi2_hemolytic: bool | None = None
    selectivity_proxy: float | None = Field(default=None, gt=0)


class Random25Row(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str
    draw: int = Field(ge=1)
    selected_ranks: tuple[int, ...]
    apex_vote16: float
    apex_consensus16: float
    gram_negative_success16: float
    gram_positive_success16: float
    mdr_proxy: float
    apex_broad_mean_mic_u_m: float
    apex_model_disagreement: float
    embedding_cluster_coverage: int
    pep_hard_filter_pass_fraction: float
    spps_favorable_fraction: float
    hemopi2_non_hemolytic_fraction: float | None
    hemopi2_hc50_median_u_m: float | None
    selectivity_proxy_median: float | None


class RandomDrawEvaluation(BaseModel):
    model_config = ConfigDict(frozen=True)

    variant: str
    draws: int
    sample_size: int
    seed: int
    rows: tuple[Random25Row, ...]
    summary: dict[str, NumericSummary]


class HemoPI2Prediction(BaseModel):
    model_config = ConfigDict(frozen=True)

    candidate_id: str
    sequence: str
    hc50_u_m: float = Field(gt=0)
    hemolytic: bool


class EvaluationCoverage(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    row_count: int = 0
    detail: str


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    run_dir: str
    counts: dict[str, int]
    input_sha256: dict[str, str]
    compliance: dict[str, object]
    coverage: dict[str, EvaluationCoverage]
    library_metrics: dict[str, NumericSummary]
    pathogen_groups: dict[str, NumericSummary]
    seqme_metrics: dict[str, float]
    top_metrics: dict[str, NumericSummary]
    random25: dict[str, dict[str, NumericSummary]]
    rerank: dict[str, object]
    limitations: tuple[str, ...]
    output_sha256: dict[str, str]
