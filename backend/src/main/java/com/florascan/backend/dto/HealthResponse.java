package com.florascan.backend.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Health check response from the Python ML service.
 */
@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class HealthResponse {

    @JsonProperty("status")
    private String status;

    @JsonProperty("model_loaded")
    private boolean modelLoaded;

    @JsonProperty("num_classes")
    private int numClasses;

    @JsonProperty("supported_crops_count")
    private int supportedCropsCount;

    @JsonProperty("calibration_active")
    private boolean calibrationActive;

    @JsonProperty("temperature")
    private double temperature;

    @JsonProperty("disclaimer")
    private String disclaimer;
}
