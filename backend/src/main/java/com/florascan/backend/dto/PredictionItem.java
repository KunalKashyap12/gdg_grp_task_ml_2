package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Top ranked diagnosis item returned by the ML inference engine.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class PredictionItem {

    @JsonProperty("class_id")
    private int classId;

    @JsonProperty("crop")
    private String crop;

    @JsonProperty("disease")
    private String disease;

    @JsonProperty("label")
    private String label;

    @JsonProperty("confidence")
    private double confidence;
}
