package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;

/**
 * Root response object returned by the FloraScan AI prediction endpoint (/api/v1/predict).
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
@JsonInclude(JsonInclude.Include.NON_NULL)
public class CropPredictionResponse {

    /**
     * Diagnostic triage status:
     * - "success": High confidence diagnosis (above threshold).
     * - "low_confidence": Ambiguous pathology flagged for agronomist review.
     * - "not_a_leaf": Input rejected by Botanical Gatekeeper (wallpaper, car, sketch, etc.).
     */
    @JsonProperty("status")
    private String status;

    @JsonProperty("is_leaf")
    private boolean isLeaf;

    @JsonProperty("prediction")
    private PredictionItem prediction;

    @JsonProperty("top_k")
    private List<TopKItem> topK;

    @JsonProperty("low_confidence")
    private boolean lowConfidence;

    @JsonProperty("threshold")
    private double threshold;

    @JsonProperty("message")
    private String message;

    @JsonProperty("model_version")
    private String modelVersion;

    @JsonProperty("inference_time_ms")
    private double inferenceTimeMs;

    @JsonProperty("stage1_leaf")
    private Stage1Leaf stage1Leaf;

    @JsonProperty("stage2_disease")
    private Stage2Disease stage2Disease;

    @JsonProperty("crop_conditioned")
    private String cropConditioned;

    @JsonProperty("disclaimer")
    private String disclaimer;
}
